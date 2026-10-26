"""Question Analysis: choose papers and notes for the active subject, run, and review results."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

import config
from components import nav, record_charts as rc, state, ui
from nlp.question_segmentation import QuestionItem
from pages_content import _shared as sh
from services.analysis_runner import AnalysisSettings
from services.errors import SyllabusLensError


# ------------------------------------------------------------------ inputs
def _paper_label(d) -> str:
    extra = " ".join(x for x in (d.exam_session, d.academic_year) if x)
    return f"{d.title}" + (f" ({extra})" if extra else "")


def _settings_ui(sid: str) -> Optional[AnalysisSettings]:
    with st.expander("Matching settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        method = c1.radio("Method", ["tfidf", "bow"], key=f"an_method_{sid}", horizontal=True,
                          format_func={"tfidf": "TF-IDF", "bow": "Bag of Words"}.get,
                          help="TF-IDF down-weights words that appear in many sections. Bag of Words uses raw counts.")
        ngram = c2.radio("Word groups", [(1, 1), (1, 2)], index=1, key=f"an_ngram_{sid}", horizontal=True,
                         format_func=lambda n: "Single words" if n == (1, 1) else "Words + pairs",
                         help="Pairs (bigrams) capture short phrases such as 'cosine similarity'.")
        stem = c3.checkbox("Stemming (Porter)", key=f"an_stem_{sid}", help="Treats 'parsing' and 'parse' as the same term.")
        t1, t2, t3, t4 = st.columns(4)
        weak = t1.number_input("Weak at", 0.01, 0.9, config.DEFAULT_WEAK_THRESHOLD, 0.01, key=f"an_weak_{sid}", format="%.2f")
        mod = t2.number_input("Moderate at", 0.02, 0.95, config.DEFAULT_MODERATE_THRESHOLD, 0.01, key=f"an_mod_{sid}", format="%.2f")
        strong = t3.number_input("Strong at", 0.03, 1.0, config.DEFAULT_STRONG_THRESHOLD, 0.01, key=f"an_strong_{sid}", format="%.2f")
        top_k = int(t4.number_input("Sections per question", 1, 5, 3, 1, key=f"an_topk_{sid}"))
        st.caption("Thresholds are adjustable heuristics, not calibrated confidence levels. Changing them changes the tier labels, not the similarity scores.")
    s = AnalysisSettings(vectorizer=method, ngram_range=ngram, use_stemming=stem, strong_threshold=strong,
                         moderate_threshold=mod, weak_threshold=weak, top_k=top_k)
    try:
        s.validate()
    except ValueError as exc:
        ui.notice("error", "Settings are inconsistent", str(exc))
        return None
    return s


def _question_editor(svc, paper) -> None:
    """Inspect and correct the questions extracted from one paper."""
    qs = svc.library.questions_of(paper.id)
    df = pd.DataFrame([{"ID": q.question_id, "Question": q.clean_text, "Marks": q.marks or ""} for q in qs])
    st.caption(f"{len(qs)} question(s) " + ("(your corrections)" if paper.has_corrections else "extracted automatically. Fix splits, typos or merged questions here."))
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", hide_index=True, key=f"qed_{paper.id}",
                            column_config={"ID": st.column_config.TextColumn(width="small", required=True),
                                           "Question": st.column_config.TextColumn(width="large", required=True),
                                           "Marks": st.column_config.TextColumn(width="small")})
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Save corrections", key=f"qsave_{paper.id}"):
        by_id = {q.question_id: q for q in qs}
        items: List[QuestionItem] = []
        for _, r in edited.iterrows():
            text, qid = str(r["Question"] or "").strip(), str(r["ID"] or "").strip()
            if not text or not qid:
                continue
            old = by_id.get(qid)
            items.append(QuestionItem(question_id=qid, raw_text=text, clean_text=text, marks=(str(r["Marks"]).strip() or None),
                                      parent_id=old.parent_id if old else None, sub_label=old.sub_label if old else None,
                                      metadata=dict(old.metadata) if old else {}))
        try:
            svc.library.save_question_corrections(paper.id, items)
        except SyllabusLensError as exc:
            ui.notice("error", "Corrections were not saved", str(exc))
        else:
            st.toast("Corrections saved. They apply to new analyses; saved ones are unchanged.")
            st.rerun()
    if paper.has_corrections and c2.button("Reset to automatic", key=f"qreset_{paper.id}"):
        svc.library.clear_question_corrections(paper.id)
        st.rerun()


# ------------------------------------------------------------------ results
def render_results(svc, subject, meta: Dict[str, Any], record: Dict[str, Any], key: str = "res") -> None:
    sh.scope_statement(meta, record)
    sh.staleness_notice(svc, meta)
    m = record["metrics"]
    ui.metric_grid([
        ("Questions analysed", m["total_questions"], "Across the selected paper(s)"),
        ("Sections considered", m["total_sections"], "Indexed units of lecture material searched"),
        ("Strong", m["strong_count"], "Similarity at or above the strong threshold (heuristic)"),
        ("Moderate", m["moderate_count"], "Between the moderate and strong thresholds"),
        ("Weak", m["weak_count"], "Leads only: little shared vocabulary", "warning" if m["weak_count"] else ""),
        ("Unmatched", m["unmatched_count"], "Below the weak threshold: no notes section shares enough vocabulary", "danger" if m["unmatched_count"] else ""),
        ("Question match rate", f"{m['overall_coverage_rate']:.0f}%", "Share of questions whose best match is Strong or Moderate. Not syllabus coverage."),
    ])
    sh.similarity_caption()

    tab_q, tab_charts, tab_gaps, tab_export = st.tabs(["Questions", "Charts", "Unexamined sections", "Export"])
    with tab_q:
        c1, c2 = st.columns([2, 3])
        tiers = c1.multiselect("Show tiers", rc.TIERS, default=rc.TIERS, key=f"{key}_tiers_{meta['id']}")
        multi = len(record["papers"]) > 1
        papers = c2.multiselect("Papers", [p["document_id"] for p in record["papers"]], default=[p["document_id"] for p in record["papers"]],
                                format_func=lambda i: next(p["title"] for p in record["papers"] if p["document_id"] == i),
                                key=f"{key}_papers_{meta['id']}") if multi else [p["document_id"] for p in record["papers"]]
        shown = [q for q in record["questions"] if q["best_tier"] in tiers and q["paper_doc_id"] in papers]
        if not shown:
            ui.empty_state("No questions match these filters", "Select more tiers or papers.")
        for q in shown[:60]:
            ui.render(ui.question_card_html(q, show_paper=multi))
        if len(shown) > 60:
            st.caption(f"Showing 60 of {len(shown)}. Use Question Explorer for the complete list.")
        if st.button("Inspect questions in detail", key=f"{key}_to_explorer_{meta['id']}"):
            nav.goto("explorer")
    with tab_charts:
        with ui.chart_card(f"{key}-tiers", "Questions by match tier", "Each bar counts a paper's questions by the strength of their best-matching section. Tiers are heuristic labels."):
            fig = rc.tier_distribution(record)
            st.plotly_chart(fig, width="stretch", key=f"{key}_tier_{meta['id']}") if fig else ui.empty_state("No questions", "")
        with ui.chart_card(f"{key}-hist", "Distribution of best similarity scores", "How strongly each question's best section matches. Dotted lines mark the weak / moderate / strong thresholds."):
            fig = rc.score_histogram(record)
            st.plotly_chart(fig, width="stretch", key=f"{key}_hist_{meta['id']}") if fig else None
    with tab_gaps:
        unex = [s for s in record["sections"] if s["coverage_status"] == "Uncovered in Exam"]
        st.caption("Sections of the selected lecture material that no question in these paper(s) resembles. "
                   "That can mean they were not examined, or that the paper words the topic differently.")
        if not unex:
            ui.notice("success", "Every section is similar to at least one question", "At the weak threshold or above.")
        else:
            st.dataframe(pd.DataFrame([{"Section": s["heading"], "Document": s["doc_name"], "Location": s["locator"],
                                        "Highest similarity": s["max_similarity"]} for s in unex]), width="stretch", hide_index=True)
    with tab_export:
        st.caption("Files include the method, settings, timestamp, corpus version and source references.")
        sh.export_buttons(svc, meta, record, f"{key}_{meta['id']}", reviewed=svc.history.reviewed_ids(subject.id))


# ------------------------------------------------------------------ page
def render() -> None:
    svc = state.get_services()
    ui.page_header("Question Analysis", "Match a question paper against the lecture material of one subject.")
    subject = state.require_subject(svc)
    scope_ph = st.empty()
    lib = svc.library
    docs = lib.list_documents(subject.id, status="processed", limit=1000)
    papers = [d for d in docs if d.is_question_paper]
    material = [d for d in docs if not d.is_question_paper]
    eligible = [d for d in material if config.DOCUMENT_KINDS[d.doc_kind][2]]
    with scope_ph.container():
        state.show_scope(svc, subject)

    if not eligible or not papers:
        missing = []
        if not eligible: missing.append("lecture notes or slides")
        if not papers: missing.append("a question paper")
        ui.empty_state("This subject isn't ready for analysis", f"Upload {' and '.join(missing)} to {subject.name} first. Unreadable files are listed in the library with the reason.")
        if st.button("Open Document Library", type="primary", key="an_to_lib"):
            nav.goto("library")
        _show_latest(svc, subject)
        return

    ui.section_heading("1. Choose what to analyse")
    sid = subject.id
    chosen_papers = st.multiselect("Question paper(s)", [d.id for d in papers], default=[papers[0].id], key=f"an_papers_{sid}",
                                   format_func={d.id: _paper_label(d) for d in papers}.get)
    cross = st.toggle("Also search other subjects", key=f"an_cross_{sid}", value=False,
                      help="Off by default. When on, you must name the extra subjects; results are labelled cross-subject.")
    extra_ids: List[str] = []
    if cross:
        others = [s for s in svc.subjects.list() if s.id != sid]
        if not others:
            ui.notice("info", "There are no other subjects to search", "Create another subject first, or turn this off.")
        extra_ids = st.multiselect("Additional subjects to include", [s.id for s in others], key=f"an_extra_{sid}",
                                   format_func={s.id: s.name for s in others}.get)
        ui.notice("warning", "Cross-subject mode", "Lecture material from the subjects you select will be searched together with this subject's, "
                  "and the saved analysis will be labelled cross-subject. Every eligible document in those subjects is included.")
        with scope_ph.container():
            state.show_scope(svc, subject, [s.name for s in others if s.id in extra_ids])
        chosen_notes: Optional[List[str]] = None
    else:
        chosen_notes = st.multiselect("Lecture material", [d.id for d in material], default=[d.id for d in eligible], key=f"an_notes_{sid}",
                                      format_func={d.id: f"{d.title} ({ui.kind_label(d.doc_kind)})" for d in material}.get,
                                      help="Notes, slides, tutorials and reference material are included by default. Add assignments or syllabus documents if useful.")

    for p in papers:
        if p.id in chosen_papers:
            with st.expander(f"Review extracted questions: {p.title}"):
                _question_editor(svc, p)

    settings = _settings_ui(sid)
    ready = bool(chosen_papers) and settings is not None and (cross is False and bool(chosen_notes) or cross and bool(extra_ids))
    if not chosen_papers:
        st.caption("Select at least one question paper.")
    if st.button("Run analysis", type="primary", key=f"an_run_{sid}", disabled=not ready):
        try:
            with st.status("Matching questions to lecture material...", expanded=True) as status:
                st.write("Applying subject filter and building the corpus")
                outcome = svc.runner.run(sid, chosen_papers, chosen_notes, settings=settings, allow_cross_subject=cross, extra_subject_ids=extra_ids)
                st.write(f"Ranked {outcome.record['metrics']['total_questions']} question(s) against {outcome.record['metrics']['total_sections']} section(s)")
                status.update(label="Analysis complete", state="complete", expanded=False)
        except SyllabusLensError as exc:
            ui.notice("error", "The analysis could not run", str(exc))
        else:
            state.set_last_analysis_id(sid, outcome.analysis_id)
            if outcome.reused_saved:
                ui.notice("info", "Identical analysis already saved", "Same documents, settings and corpus version: showing the existing result instead of creating a duplicate.")
    _show_latest(svc, subject)


def _show_latest(svc, subject) -> None:
    loaded = sh.load_for_subject(svc, state.get_last_analysis_id(subject.id), subject.id)
    ui.section_heading("2. Results")
    if not loaded:
        ui.empty_state("No results yet", "Run an analysis above, or open an earlier one from Analysis History.")
        return
    render_results(svc, subject, *loaded)
