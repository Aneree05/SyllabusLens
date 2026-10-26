"""
Two-stage upload: (1) preview + suggestion (nothing stored), (2) explicit confirmation per file.
An automatic suggestion is advisory; the subject and type the user selects are authoritative.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import streamlit as st

import config
from components import state, ui
from services.errors import DocumentValidationError, DuplicateDocumentError, NotFoundError, SyllabusLensError

PREVIEW_CHARS = 3000
_SUGGEST_TONE = {"suggested": "success", "ambiguous": "warning", "insufficient": "info", "no_subjects": "info"}
_SUGGEST_NAME = {"suggested": "Suggested", "ambiguous": "Ambiguous", "insufficient": "Not enough evidence", "no_subjects": "No subjects"}


def _nonce(sid: str) -> int:
    return st.session_state.setdefault(state.sk(sid, "upload_nonce"), 0)


def _cached(sid: str, name: str, compute):
    key = state.sk(sid, name)
    if key not in st.session_state:
        st.session_state[key] = compute()
    return st.session_state[key]


def _suggestion_panel(svc, sug, widget_key: str, current_subject_id: str) -> None:
    ui.render(f'<div class="sl-card-meta" style="margin:.2rem 0">{ui.badge("Automatic suggestion - not confirmed", "neutral")} '
              f'{ui.badge(_SUGGEST_NAME[sug.status], _SUGGEST_TONE[sug.status])}</div>')
    st.caption(sug.message)
    if sug.suggestions and sug.status in ("suggested", "ambiguous"):
        rows = [f"**{s.name}** - similarity {s.score:.2f}" + (f" (matching terms: {', '.join(s.evidence_terms[:4])})" if s.evidence_terms else "")
                for s in sug.suggestions]
        st.markdown("  \n".join(rows).replace("$", "\\$"))
        st.caption("Similarity is a TF-IDF cosine score between the file and each subject's profile. It is not a probability.")
        top = sug.top
        if top.subject_id != current_subject_id:
            st.button(f"Use suggested subject: {top.name}", key=f"{widget_key}_use",
                      on_click=lambda: st.session_state.__setitem__(widget_key, top.subject_id))


def _file_panel(svc, active, uploaded, kind_choice: str, nonce: int) -> Dict[str, Any]:
    """Render one file's review panel; return the decision to apply when the user clicks Save."""
    data = uploaded.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    pv = _cached(active.id, f"preview:{nonce}:{sha}", lambda: svc.library.preview_upload(uploaded.name, data))
    base = f"up_{nonce}_{sha[:12]}"
    decision: Dict[str, Any] = {"name": pv.filename, "ok": False}
    with st.expander(f"{pv.filename}  ({ui.human_size(len(data))})", expanded=True):
        if pv.error:
            ui.notice("error", "This file cannot be uploaded", pv.error)
            return decision
        unit = "slides" if pv.ext == "pptx" else "pages"
        count_badge = ui.badge(f"{pv.page_count} {unit}") if pv.page_count else ""
        ui.render(f"{ui.status_badge(pv.status)} {ui.badge('.' + pv.ext)} {count_badge}")
        if pv.status != "processed" or pv.status_detail:
            ui.notice("warning" if pv.status == "processed" else "error",
                      "Extraction needs attention" if pv.status == "processed" else "No usable text was extracted", pv.status_detail
                      or "The file could not be read. Try re-exporting it as a text-based PDF, PPTX or TXT.")
        if pv.text.strip():
            shown = pv.text[:PREVIEW_CHARS]
            st.text_area("Extracted text preview", shown, height=150, disabled=True, key=f"{base}_prev",
                         help=f"Showing the first {len(shown):,} of {len(pv.text):,} characters.")
        subjects = svc.subjects.list()
        ids = [s.id for s in subjects]
        names = {s.id: s.name for s in subjects}
        sug = _cached(active.id, f"suggest:{nonce}:{sha}", lambda: svc.classifier.suggest_for_text(pv.text, pv.filename)) if pv.status == "processed" else None
        subj_key = f"{base}_subj"
        if subj_key not in st.session_state:
            st.session_state[subj_key] = active.id
        if sug is not None:
            _suggestion_panel(svc, sug, subj_key, st.session_state.get(subj_key, active.id))
        target_id = st.selectbox("Save to subject (your choice is final)", ids, format_func=names.get, key=subj_key)
        kinds = list(config.DOCUMENT_KINDS)
        default_kind = kind_choice if kind_choice != "auto" else pv.kind_suggestion[0]
        kind = st.selectbox("Document type", kinds, index=kinds.index(default_kind), key=f"{base}_kind",
                            format_func=ui.kind_label)
        if kind_choice == "auto" and pv.kind_suggestion:
            st.caption(f"Type suggestion: {ui.kind_label(pv.kind_suggestion[0])}. {pv.kind_suggestion[1]}")
        if kind == config.QUESTION_PAPER_KIND:
            st.caption(f"{pv.question_count} question(s) detected automatically. You can review and correct them before analysis.")
        decision.update(data=data, subject_id=target_id, kind=kind, pv=pv)
        dups = svc.library.duplicates_of(pv.file_hash)
        same = [d for d in dups if d["subject_id"] == target_id]
        other = [d for d in dups if d["subject_id"] != target_id]
        allow_dup = False
        if same:
            ui.notice("warning", "Already in this subject", f"Identical to '{same[0]['filename']}'. It will be skipped unless you allow a copy.")
            allow_dup = st.checkbox("Upload a duplicate copy anyway", key=f"{base}_dup")
        if other:
            st.caption("Also present in: " + ", ".join(f"{d['filename']} ({d['subject_name']})" for d in other) + ". Subjects keep separate copies.")
        save_anyway = True
        if pv.status != "processed":
            save_anyway = st.checkbox("Save anyway (listed as failed and excluded from analysis)", key=f"{base}_anyway")
        decision.update(ok=(not same or allow_dup) and save_anyway, allow_dup=allow_dup)
    return decision


def _apply(svc, decisions: List[Dict[str, Any]], meta: Dict[str, Any]) -> List[Dict[str, str]]:
    names = {s.id: s.name for s in svc.subjects.list()}
    out = []
    for d in decisions:
        if not d.get("ok"):
            out.append(dict(name=d["name"], outcome="skipped", message="Not saved (error, duplicate or not confirmed)."))
            continue
        try:
            res = svc.library.ingest(d["subject_id"], d["name"], d["data"], doc_kind=d["kind"], allow_duplicate=d["allow_dup"], **meta)
            msg = f"Saved to {names[d['subject_id']]} as {ui.kind_label(d['kind'])}."
            out.append(dict(name=d["name"], outcome="saved", message=msg + (" " + " ".join(res.warnings) if res.warnings else "")))
        except (DuplicateDocumentError, DocumentValidationError, NotFoundError, SyllabusLensError) as exc:
            out.append(dict(name=d["name"], outcome="error", message=str(exc)))
    return out


def render_upload_tab(svc, active) -> None:
    sid = active.id
    summary = st.session_state.pop(state.sk(sid, "upload_summary"), None)
    if summary:
        saved = [s for s in summary if s["outcome"] == "saved"]
        if saved:
            ui.notice("success", f"{len(saved)} document(s) saved", " ".join(f"{s['name']}: {s['message']}" for s in saved))
        for s in summary:
            if s["outcome"] != "saved":
                ui.notice("error" if s["outcome"] == "error" else "warning", f"{s['name']} was not saved", s["message"])
    nonce = _nonce(sid)
    mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    c1, c2 = st.columns([2, 3])
    kind_choice = c1.selectbox("Document type for this upload", ["auto"] + list(config.DOCUMENT_KINDS), key=f"up_kind_{sid}",
                               format_func=lambda k: "Detect automatically (you confirm)" if k == "auto" else ui.kind_label(k))
    c2.caption(f"Supported: PDF (text-based), PPTX, TXT, MD. Maximum {mb} MB per file. Scanned/image-only PDFs have no OCR and will be flagged.")
    files = st.file_uploader("Choose one or more files", type=[e.lstrip(".") for e in config.ALLOWED_UPLOAD_EXTENSIONS],
                             accept_multiple_files=True, key=f"uploader_{sid}_{nonce}")
    if not files:
        ui.empty_state("No files selected yet", "Choose or drop files in the box above: lecture notes, slides, question papers or a syllabus. You review what was extracted and confirm the subject before anything is saved.")
        return
    with st.expander("Optional details for all files in this upload"):
        a, b, c, d = st.columns(4)
        tags = a.text_input("Tags (comma separated)", key=f"up_tags_{sid}_{nonce}")
        year = b.text_input("Academic year", key=f"up_year_{sid}_{nonce}")
        sem = c.text_input("Semester", key=f"up_sem_{sid}_{nonce}")
        sess = d.text_input("Exam session", key=f"up_sess_{sid}_{nonce}")
    decisions = [_file_panel(svc, active, f, kind_choice, nonce) for f in files]
    ready = [d for d in decisions if d.get("ok")]
    st.markdown("&nbsp;")
    if st.button(f"Save {len(ready)} document(s)", type="primary", key=f"up_save_{sid}_{nonce}", disabled=not ready):
        meta = dict(tags=[t for t in tags.split(",") if t.strip()], academic_year=year, semester=sem, exam_session=sess)
        with st.status("Saving and indexing documents...", expanded=False) as status:
            result = _apply(svc, decisions, meta)
            status.update(label="Finished", state="complete")
        st.session_state[state.sk(sid, "upload_summary")] = result
        st.session_state[state.sk(sid, "upload_nonce")] = nonce + 1
        st.rerun()
    elif not ready:
        st.caption("Nothing to save yet: resolve the items flagged above, or confirm them.")
