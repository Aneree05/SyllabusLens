"""Evaluation: score retrieval against ground-truth judgments (bundled demo set or your own file)."""
from __future__ import annotations

import streamlit as st

from components import state, ui
from services import evaluation_data as ed
from services.errors import SyllabusLensError


def _find(svc, subject_id: str, filename: str):
    return next((d for d in svc.library.list_documents(subject_id, limit=1000) if d.filename == filename), None)


def _table(results) -> None:
    rows = []
    for name, r in results["reports"].items():
        m = r["report"]
        rows.append({"Method": name, "Questions scored": m.num_queries, "P@1": round(m.precision_at_1, 3), "P@3": round(m.precision_at_3, 3),
                     "P@5": round(m.precision_at_5, 3), "R@1": round(m.recall_at_1, 3), "R@3": round(m.recall_at_3, 3), "R@5": round(m.recall_at_5, 3),
                     "MRR": round(m.mrr, 3), "MAP": round(m.map_score, 3)})
    st.dataframe(rows, width="stretch", hide_index=True)


def render() -> None:
    svc = state.get_services()
    ui.page_header("Evaluation", "Measure how often the matcher ranks the truly relevant section first, using labelled examples.")
    subject = state.require_subject(svc)
    state.show_scope(svc, subject)
    ui.render('<p class="sl-caption">Ground truth lists, for each question, which note section is relevant. P@k is the share of the top k results that are relevant; '
              'R@k is the share of relevant sections found in the top k; MRR is the average of 1/rank of the first relevant section; MAP averages precision across all relevant sections.</p>')
    src = st.radio("Ground truth", ["Bundled demo set", "Upload my own (JSON)"], horizontal=True, key=f"ev_src_{subject.id}")
    gt = None
    if src.startswith("Bundled"):
        try:
            gt = ed.load_ground_truth(ed.SAMPLE_GROUND_TRUTH)
        except SyllabusLensError as exc:
            ui.notice("error", "Bundled ground truth could not be loaded", str(exc))
    else:
        up = st.file_uploader("Ground-truth JSON (version 2 format)", type=["json"], key=f"ev_up_{subject.id}")
        if up is not None:
            try:
                gt = ed.load_ground_truth(up.getvalue())
            except SyllabusLensError as exc:
                ui.notice("error", "This file cannot be used", str(exc))
        with st.expander("Expected format"):
            st.code('{\n  "notes_document": "my_notes.pdf",\n  "paper_document": "paper_2023.pdf",\n  "judgments": {"Q1": ["Unit 2:"], "Q2.a": ["Unit 1:", "Unit 3:"], "Q7": []},\n'
                    '  "binary_coverage": {"Q1": true, "Q7": false}\n}', language="json")
            st.caption("Sections are referenced by a heading fragment that appears exactly once in the notes. Filenames must match documents in the active subject.")
    if gt is None:
        return
    notes, paper = _find(svc, subject.id, gt.notes_document), _find(svc, subject.id, gt.paper_document)
    if gt.is_demo:
        ui.notice("info", "Demo data", gt.description[:240])
    if not notes or not paper:
        missing = [n for n, d in ((gt.notes_document, notes), (gt.paper_document, paper)) if not d]
        ui.empty_state("Required documents are not in this subject", f"This ground truth needs {', '.join(missing)} in {subject.name}. "
                       "For the demo, load the demo subject from the Subjects page.")
        return
    thr = st.slider("Coverage classification threshold (best similarity >= threshold counts as 'covered')", 0.05, 0.6, 0.20, 0.01, key=f"ev_thr_{subject.id}")
    if st.button("Run evaluation", type="primary", key=f"ev_run_{subject.id}"):
        try:
            with st.status("Running TF-IDF and Bag-of-Words on identical inputs...", expanded=False) as s:
                st.session_state[state.sk(subject.id, "eval")] = ed.evaluate_against_truth(svc, subject.id, paper.id, [notes.id], gt, thr)
                s.update(label="Evaluation complete", state="complete")
        except SyllabusLensError as exc:
            ui.notice("error", "Evaluation could not run", str(exc))
    res = st.session_state.get(state.sk(subject.id, "eval"))
    if not res:
        return
    ui.section_heading("Retrieval quality")
    _table(res)
    n = next(iter(res["reports"].values()))["report"].num_queries
    ui.notice("warning", f"Small sample: {n} questions", "With a handful of questions, differences between methods are not statistically meaningful, and results here say nothing "
              "about other courses or papers. Identical numbers for both methods are common on small, clearly worded sets.")
    ui.section_heading("Covered vs not-covered detection", f"Does 'best similarity >= {res['threshold']:.2f}' separate in-syllabus from out-of-syllabus questions?")
    for name, r in res["reports"].items():
        c = r["coverage"]
        ui.render(f'<div class="sl-card"><div class="sl-card-title">{ui.esc(name)}</div><div class="sl-card-meta"><span>True positives {c.true_positives}</span><span>False positives {c.false_positives}</span>'
                  f'<span>True negatives {c.true_negatives}</span><span>False negatives {c.false_negatives}</span><span>Accuracy {c.accuracy:.2f}</span>'
                  f'<span>Precision {c.precision:.2f}</span><span>Recall {c.recall:.2f}</span><span>F1 {c.f1_score:.2f}</span></div></div>')
    if res["problems"]:
        ui.notice("warning", "Some ground-truth references could not be matched", " ".join(res["problems"][:5]))
    with st.expander("Per-question results (TF-IDF)"):
        rows = []
        for d in res["reports"]["TF-IDF"]["report"].query_details:
            heading = lambda sid: (svc.library.get_section(sid) or {}).get("heading", "?")
            rows.append({"Question": d["question_id"].split(":")[-1], "Relevant": "; ".join(heading(s) for s in d["relevant_sections"]),
                         "Top 3 retrieved": "; ".join(heading(s) for s in d["retrieved_top3"]), "P@1": d["p@1"], "Reciprocal rank": d["rr"]})
        st.dataframe(rows, width="stretch", hide_index=True)
