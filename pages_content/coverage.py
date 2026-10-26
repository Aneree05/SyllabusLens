"""Coverage Lab: where the selected papers do and do not touch the lecture material and syllabus."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import record_charts as rc, state, ui
from pages_content import _shared as sh

TOPIC_HELP = {
    "detected": "A question reaches the weak-match threshold for this topic.",
    "not_detected": "Topic words occur in the notes, but no question in the selected papers is similar enough.",
    "not_assessable": "Half or fewer of the topic's words occur in your notes, so a lexical comparison would rest on a few generic words. This is not evidence the topic is absent from exams.",
}


def _plot(fig, key, empty="Nothing to chart for this analysis."):
    if fig is None:
        ui.empty_state("No data", empty)
    else:
        st.plotly_chart(fig, width="stretch", key=key)


def render() -> None:
    svc = state.get_services()
    ui.page_header("Coverage Lab", "Which sections, topics and questions are covered, weakly covered or missing.")
    subject = state.require_subject(svc)
    state.show_scope(svc, subject)
    loaded = sh.analysis_picker(svc, subject, key="cov_pick")
    if not loaded:
        return
    meta, record = loaded
    sh.staleness_notice(svc, meta)
    m, secs = record["metrics"], record["sections"]
    counts = {k: sum(1 for s in secs if s["coverage_status"] == k) for k in rc.STATUS_COLORS}
    ui.metric_grid([
        ("Sections similar to a question", counts["Covered"] + counts["Partially Covered"], f"Of {len(secs)} sections, at or above the weak threshold"),
        ("Sections not examined", counts["Uncovered in Exam"], "No question in these papers resembles them", "warning" if counts["Uncovered in Exam"] else ""),
        ("Unmatched questions", m["unmatched_count"], "No section shares enough vocabulary", "danger" if m["unmatched_count"] else ""),
        ("Median best similarity", f"{m['median_similarity']:.2f}", "Across questions"),
    ])
    sh.similarity_caption()
    k = meta["id"]
    t_heat, t_sec, t_topic, t_rel, t_un = st.tabs(["Similarity heatmap", "Sections", "Syllabus topics", "Relationships", "Unmatched questions"])
    with t_heat:
        if len(record["papers"]) > 1:
            st.caption("Paper key: " + "; ".join(f"{c} = {next(p['title'] for p in record['papers'] if p['document_id'] == d)}" for d, c in rc.paper_codes(record).items()))
        with ui.chart_card(f"cov-heat-{k}", "Question vs section similarity",
                           "Each cell is the cosine similarity (0-1) between one question (row) and one section (column). Darker means more shared vocabulary. Hover for the question and section text."):
            _plot(rc.similarity_heatmap(record), f"heat_{k}")
        with st.expander("Section key (S1, S2, ...)"):
            st.dataframe(pd.DataFrame([{"Label": f"S{i + 1}", "Section": s["heading"], "Document": s["doc_name"], "Location": s["locator"]} for i, s in enumerate(secs)]),
                         width="stretch", hide_index=True)
    with t_sec:
        with ui.chart_card(f"cov-sec-{k}", "Section coverage", "Bar length is the highest similarity between the section and any question. Colour shows whether it reaches the moderate / weak thresholds."):
            _plot(rc.section_coverage_bars(record), f"sec_{k}")
        st.dataframe(pd.DataFrame([{"Section": s["heading"], "Document": s["doc_name"], "Location": s["locator"], "Status": s["coverage_status"],
                                    "Highest similarity": s["max_similarity"], "Top-ranked for": s["top1_match_count"]} for s in secs]), width="stretch", hide_index=True)
    with t_topic:
        topics = record.get("topic_coverage") or []
        if not topics:
            ui.empty_state("No syllabus topics were configured", f"Add topics to {subject.name} on the Subjects page, then run a new analysis to see which topics the papers touch.")
        else:
            with ui.chart_card(f"cov-topic-{k}", "Syllabus topics vs question papers", "Bar length is the highest similarity between the topic text and any question. Grey means the topic could not be assessed lexically."):
                _plot(rc.topic_coverage_bars(record), f"topic_{k}")
            for status in ("not_detected", "not_assessable", "detected"):
                rows = [t for t in topics if t["status"] == status]
                if rows:
                    with st.expander(f"{rc.TOPIC_LABELS[status]} ({len(rows)})", expanded=status != "detected"):
                        st.caption(TOPIC_HELP[status])
                        st.dataframe(pd.DataFrame([{"Topic": t["topic"], "Best question similarity": t["best_question_score"],
                                                    "Closest notes section": t.get("best_section_heading") or "-"} for t in rows]), width="stretch", hide_index=True)
    with t_rel:
        thr = st.slider("Show links at or above similarity", 0.05, 0.6, max(0.05, record["thresholds"]["weak"]), 0.01, key=f"cov_sank_{k}")
        with ui.chart_card(f"cov-sank-{k}", "Question to section relationships", "A link means the section is among the question's top-ranked matches with at least this similarity. Link thickness follows the score. No link means no recorded match."):
            _plot(rc.question_section_sankey(record, thr), f"sank_{k}", "No question-section pair reaches this similarity. Lower the threshold.")
    with t_un:
        un = [q for q in record["questions"] if q["best_tier"] == "Unmatched"]
        if not un:
            ui.notice("success", "Every question has at least a weak match", "Review weak matches in Question Explorer; weak means leads, not coverage.")
        else:
            st.caption("These questions share too little vocabulary with any selected lecture section. Check whether the material is missing or the wording differs.")
            for q in un:
                ui.render(ui.question_card_html(q, show_paper=len(record["papers"]) > 1))
