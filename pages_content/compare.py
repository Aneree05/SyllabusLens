"""Compare Papers: observed overlap between question papers of ONE subject (not a forecast)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import record_charts as rc, state, ui
from pages_content import _shared as sh
from services.errors import SyllabusLensError


def _run_form(svc, subject) -> None:
    papers = [d for d in svc.library.list_documents(subject.id, kind=config.QUESTION_PAPER_KIND, status="processed", limit=200)]
    eligible = [d for d in svc.library.list_documents(subject.id, status="processed", limit=1000)
                if not d.is_question_paper and config.DOCUMENT_KINDS[d.doc_kind][2]]
    if len(papers) < 2:
        ui.empty_state("Two question papers are needed", f"{subject.name} has {len(papers)}. Upload another paper to compare, for example a different year or session.")
        return
    with st.expander("Compare papers from this subject", expanded=True):
        picked = st.multiselect("Papers to compare (this subject only)", [d.id for d in papers], key=f"cmp_sel_{subject.id}",
                                format_func={d.id: f"{d.title} {d.exam_session} {d.academic_year}".strip() for d in papers}.get)
        st.caption(f"Compared against all {len(eligible)} lecture document(s) of {subject.name}. Use Question Analysis for custom note selection and settings.")
        if st.button("Compare selected papers", type="primary", key=f"cmp_run_{subject.id}", disabled=len(picked) < 2 or not eligible):
            try:
                with st.status("Comparing papers...", expanded=False) as status:
                    out = svc.runner.run(subject.id, picked, None, name=f"{subject.name}: comparison of {len(picked)} papers")
                    status.update(label="Comparison complete", state="complete")
            except SyllabusLensError as exc:
                ui.notice("error", "The comparison could not run", str(exc))
            else:
                state.set_last_analysis_id(subject.id, out.analysis_id)
                st.rerun()


def render() -> None:
    svc = state.get_services()
    ui.page_header("Compare Papers", "How question papers of one subject overlap in the material they touch.")
    subject = state.require_subject(svc)
    state.show_scope(svc, subject)
    _run_form(svc, subject)
    loaded = sh.analysis_picker(svc, subject, key="cmp_pick", need_comparison=True)
    if not loaded:
        return
    meta, record = loaded
    comp = record["comparison"]
    sh.scope_statement(meta, record)
    sh.staleness_notice(svc, meta)
    ui.notice("info", "What this shows", comp["basis"])
    names = {p["document_id"]: p["title"] for p in comp["papers"]}
    k = meta["id"]
    ui.metric_grid([("Papers compared", len(comp["papers"]), "All from " + record["subject"]["name"]),
                    ("Sections hit by 2+ papers", len(comp["repeated_sections"]), "Observed repetition in these papers only"),
                    ("Sections hit by any paper", sum(1 for s in comp["section_frequency"] if s["papers_hit"] > 0), f"Of {record['metrics']['total_sections']} sections")])
    t_over, t_rep, t_pairs = st.tabs(["Overview", "Repeated sections", "Paper pairs"])
    with t_over:
        with ui.chart_card(f"cmp-tier-{k}", "Match tiers per paper", "How each paper's questions distribute across match tiers against the same lecture material."):
            fig = rc.paper_tier_comparison(record)
            st.plotly_chart(fig, width="stretch", key=f"cmp_tier_{k}") if fig else None
        with ui.chart_card(f"cmp-freq-{k}", "Sections by number of papers touching them", "Dark bars are sections that questions from two or more papers resemble. This is observed frequency, not a prediction."):
            fig = rc.section_frequency_bars(record)
            st.plotly_chart(fig, width="stretch", key=f"cmp_freq_{k}") if fig else ui.empty_state("No overlap", "No section reaches the moderate threshold for any paper.")
        st.dataframe(pd.DataFrame([{"Paper": p["title"], "Questions": p["questions"], "Sections hit": p["sections_hit"], **p["tiers"]} for p in comp["papers"]]), width="stretch", hide_index=True)
    with t_rep:
        rep = comp["repeated_sections"]
        if not rep:
            ui.empty_state("No repeated sections", "No section is matched at the moderate threshold by questions from two or more of these papers.")
        for s in sorted(rep, key=lambda s: -s["papers_hit"]):
            hits = "%d of %d papers" % (s["papers_hit"], s["papers_total"])
            best = ", ".join("%s %.2f" % (names[p][:22], v) for p, v in s["best_by_paper"].items())
            ui.render(f'<div class="sl-card"><div class="sl-card-title">{ui.esc(s["heading"])}</div><div>{ui.badge(hits, "primary")}</div>'
                      f'<div class="sl-card-meta"><span>{ui.esc(s["doc_name"])}</span><span>{ui.esc(s["locator"])}</span>'
                      f'<span>Best per paper: {ui.esc(best)}</span></div></div>')
    with t_pairs:
        st.caption("Jaccard overlap = shared sections / sections hit by either paper. Higher means the two papers touched more of the same material.")
        st.dataframe(pd.DataFrame([{"Paper A": names[p["paper_a"]], "Paper B": names[p["paper_b"]], "Shared sections": p["shared_sections"],
                                    "Sections in either": p["union_sections"], "Overlap (Jaccard)": None if p["jaccard"] is None else round(p["jaccard"], 3)}
                                   for p in comp["pairwise"]]), width="stretch", hide_index=True)
    st.markdown("&nbsp;")
    sh.export_buttons(svc, meta, record, f"cmp_{k}")
