"""Document Library: upload, search/filter, preview, reclassify, move, replace, remove."""
from __future__ import annotations

import streamlit as st

import config
from components import state, ui
from nlp.document_segmentation import clean_heading
from pages_content.upload_flow import render_upload_tab
from services.errors import SyllabusLensError

PAGE_SIZE = 8
PREVIEW_CHARS = 4000


def _filters(svc, sid: str):
    c1, c2, c3 = st.columns([3, 2, 2])
    query = c1.text_input("Search", placeholder="Title, filename or tag", key=f"lib_q_{sid}")
    kind = c2.selectbox("Type", [""] + list(config.DOCUMENT_KINDS), key=f"lib_kind_{sid}",
                        format_func=lambda k: "All types" if not k else ui.kind_label(k))
    status = c3.selectbox("Status", ["", "processed", "failed", "no_text"], key=f"lib_status_{sid}",
                          format_func=lambda k: {"": "Any status", "processed": "Processed", "failed": "Extraction failed", "no_text": "No text found"}[k])
    d1, d2, d3, d4 = st.columns([2, 2, 2, 2])
    ext = d1.selectbox("File type", ["", "pdf", "pptx", "txt", "md"], key=f"lib_ext_{sid}", format_func=lambda e: "All files" if not e else "." + e)
    sort = d2.selectbox("Sort by", ["uploaded_desc", "uploaded_asc", "title", "filename", "size_desc"], key=f"lib_sort_{sid}",
                        format_func={"uploaded_desc": "Newest first", "uploaded_asc": "Oldest first", "title": "Title A-Z",
                                     "filename": "Filename A-Z", "size_desc": "Largest first"}.get)
    content = d3.checkbox("Search extracted text too", key=f"lib_content_{sid}")
    allsub = d4.checkbox("Show all subjects", key=f"lib_all_{sid}", help="Browsing only. Analyses never read other subjects unless you enable cross-subject mode.")
    return dict(subject_id=None if allsub else sid, kind=kind or None, status=status or None, ext=ext or None,
                query=query, search_content=content, sort=sort, limit=1000), allsub


def _preview(svc, doc) -> None:
    text = svc.library.get_text(doc.id)
    if doc.status != "processed":
        ui.notice("error" if doc.status == "failed" else "warning", "No usable text", doc.status_detail)
        return
    if doc.is_question_paper:
        qs = svc.library.questions_of(doc.id)
        st.caption(f"{len(qs)} question(s) detected" + (" (user-corrected)." if doc.has_corrections else ". Review and correct them on the Question Analysis page."))
        for q in qs[:6]:
            ui.render(f'<div class="sl-card-meta"><strong>{ui.esc(q.question_id)}</strong> {ui.esc(q.clean_text[:160])}</div>')
        if len(qs) > 6:
            st.caption(f"... and {len(qs) - 6} more.")
    else:
        secs = svc.library.sections_of(doc.id)
        if secs:
            n = st.number_input(f"Section (1-{len(secs)})", 1, len(secs), 1, key=f"sec_nav_{doc.id}",
                                help="Navigate the indexed sections of this document.")
            s = secs[int(n) - 1]
            ui.render(f'<div class="sl-card-meta"><strong>{ui.esc(clean_heading(s["heading"]))}</strong><span>{ui.esc(s["page_or_slide"])}</span><span>{s["word_count"]} words</span></div>')
            st.text_area("Section text", s["text"][:PREVIEW_CHARS], height=170, disabled=True, key=f"sec_txt_{doc.id}_{int(n)}", label_visibility="collapsed")
    st.text_area("Full extracted text", text[:PREVIEW_CHARS], height=140, disabled=True, key=f"txt_{doc.id}",
                 help=f"First {min(len(text), PREVIEW_CHARS):,} of {len(text):,} characters.")


def _actions(svc, doc, subjects) -> None:
    t1, t2, t3, t4 = st.tabs(["Rename & type", "Move to subject", "Details & file", "Remove"])
    with t1:
        new_title = st.text_input("Title", doc.title, key=f"ttl_{doc.id}")
        if st.button("Save title", key=f"ttl_btn_{doc.id}") and new_title != doc.title:
            try:
                svc.library.rename(doc.id, new_title); st.rerun()
            except SyllabusLensError as exc:
                ui.notice("error", "Not renamed", str(exc))
        kinds = list(config.DOCUMENT_KINDS)
        new_kind = st.selectbox("Document type", kinds, index=kinds.index(doc.doc_kind), format_func=ui.kind_label, key=f"knd_{doc.id}")
        if st.button("Change type", key=f"knd_btn_{doc.id}", disabled=new_kind == doc.doc_kind):
            svc.library.set_kind(doc.id, new_kind)
            st.toast("Type changed. Saved analyses keep their original inputs; new runs use the new type.")
            st.rerun()
    with t2:
        if doc.status == "processed":
            sug = svc.classifier.suggest_for_document(doc.id)
            ui.render(f'{ui.badge("Automatic suggestion - not confirmed")}')
            st.caption(sug.message + (f" (similarity {sug.top.score:.2f})" if sug.top else ""))
        others = [s for s in subjects if s.id != doc.subject_id]
        if not others:
            st.caption("Create another subject to move documents between subjects.")
        else:
            dest = st.selectbox("Move to", [s.id for s in others], format_func={s.id: s.name for s in others}.get, key=f"mv_{doc.id}")
            ok = st.checkbox("I understand the document will leave this subject's analyses", key=f"mv_ok_{doc.id}")
            if st.button("Move document", key=f"mv_btn_{doc.id}", disabled=not ok):
                svc.library.move_to_subject(doc.id, dest)
                st.toast("Moved. Both subjects' corpora changed; saved analyses were not altered.")
                st.rerun()
    with t3:
        tags = st.text_input("Tags (comma separated)", ", ".join(doc.tags), key=f"tg_{doc.id}")
        a, b, c = st.columns(3)
        yr = a.text_input("Academic year", doc.academic_year, key=f"yr_{doc.id}")
        ses = b.text_input("Exam session", doc.exam_session, key=f"se_{doc.id}")
        sem = c.text_input("Semester", doc.semester, key=f"sm_{doc.id}")
        if st.button("Save details", key=f"det_btn_{doc.id}"):
            svc.library.update_metadata(doc.id, tags=[t for t in tags.split(",") if t.strip()], academic_year=yr, exam_session=ses, semester=sem)
            st.rerun()
        rep = st.file_uploader("Replace the file (keeps title, type and subject)", type=[e.lstrip(".") for e in config.ALLOWED_UPLOAD_EXTENSIONS], key=f"rep_{doc.id}")
        if rep is not None and st.button("Replace file", key=f"rep_btn_{doc.id}"):
            try:
                svc.library.replace(doc.id, rep.name, rep.getvalue())
                st.toast("File replaced. Saved analyses still show the content they were run on."); st.rerun()
            except SyllabusLensError as exc:
                ui.notice("error", "File was not replaced", str(exc))
    with t4:
        ui.notice("warning", "Removal is permanent", "The file, its extracted text and its sections are deleted. Saved analyses that used it keep their snapshot but will be flagged as changed.")
        ok = st.checkbox(f"Yes, delete {doc.filename}", key=f"rm_ok_{doc.id}")
        if st.button("Delete document", key=f"rm_btn_{doc.id}", disabled=not ok):
            svc.library.delete(doc.id); st.toast("Document deleted."); st.rerun()


def _bulk(svc, docs, subjects, sid) -> None:
    with st.expander("Bulk actions"):
        by_id = {d.id: d for d in docs}
        picked = st.multiselect("Select documents", list(by_id), format_func=lambda i: f"{by_id[i].title} ({by_id[i].filename})", key=f"bulk_sel_{sid}")
        action = st.selectbox("Action", ["Set document type", "Add tag", "Move to subject"], key=f"bulk_act_{sid}")
        arg = None
        if action == "Set document type":
            arg = st.selectbox("New type", list(config.DOCUMENT_KINDS), format_func=ui.kind_label, key=f"bulk_kind_{sid}")
        elif action == "Add tag":
            arg = st.text_input("Tag", key=f"bulk_tag_{sid}")
        else:
            others = [s for s in subjects if s.id != sid]
            arg = st.selectbox("Destination", [s.id for s in others], format_func={s.id: s.name for s in others}.get, key=f"bulk_dest_{sid}") if others else None
        confirm = st.checkbox("Apply to all selected documents", key=f"bulk_ok_{sid}")
        if st.button("Apply", key=f"bulk_go_{sid}", disabled=not (picked and confirm and arg)):
            kw = {"kind": arg} if action == "Set document type" else {"add_tags": [arg.strip()]} if action == "Add tag" else {"move_to": arg}
            n = svc.library.bulk_update(picked, **kw)
            st.toast(f"Updated {n} document(s)."); st.rerun()


def _library_tab(svc, active) -> None:
    sid = active.id
    subjects = svc.subjects.list()
    names = {s.id: s.name for s in subjects}
    kw, allsub = _filters(svc, sid)
    docs = svc.library.list_documents(**kw)
    stats = svc.library.stats(None if allsub else sid)
    ui.metric_grid([("Documents", stats["documents"], "In the current view"), ("Indexed sections", stats["sections"], "Searchable units of lecture material"),
                    ("Extraction problems", stats["failures"], "Failed or no readable text", "danger" if stats["failures"] else "")])
    if not docs:
        if stats["documents"] == 0:
            ui.empty_state("No documents yet", "Use the Upload tab to add lecture notes, slides or question papers to this subject.")
        else:
            ui.empty_state("No documents match these filters", "Clear the search or filters to see everything in this subject.")
        return
    pages = max(1, -(-len(docs) // PAGE_SIZE))
    page = int(st.number_input(f"Page (1-{pages})", 1, pages, 1, key=f"lib_page_{sid}")) if pages > 1 else 1
    st.caption(f"Showing {(page - 1) * PAGE_SIZE + 1}-{min(page * PAGE_SIZE, len(docs))} of {len(docs)}")
    for d in docs[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        ui.render(ui.document_card_html(d, names.get(d.subject_id, ""), show_subject=allsub))
        with st.expander("Preview and manage", expanded=False):
            _preview(svc, d)
            if allsub and d.subject_id != sid:
                st.caption("This document belongs to another subject. Switch to that subject to manage it.")
            else:
                _actions(svc, d, subjects)
    if not allsub:
        _bulk(svc, docs, subjects, sid)


def render() -> None:
    svc = state.get_services()
    ui.page_header("Document Library", "Notes, slides, question papers and syllabi for the active subject.")
    active = state.require_subject(svc)
    state.show_scope(svc, active)
    up, lib = st.tabs(["Upload", "Library"])
    with up:
        render_upload_tab(svc, active)
    with lib:
        _library_tab(svc, active)
