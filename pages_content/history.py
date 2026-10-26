"""Analysis History: reopen frozen snapshots, see what changed since, compare two runs."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import state, ui
from pages_content import _shared as sh
from pages_content.analysis import render_results


def _delta(a, b):
    return b - a


def _compare_two(items, svc, subject) -> None:
    ids = [a["id"] for a in items]
    c1, c2 = st.columns(2)
    a_id = c1.selectbox("Earlier run", ids, index=min(1, len(ids) - 1), key=f"hcmp_a_{subject.id}", format_func=lambda i: sh.analysis_label(next(x for x in items if x["id"] == i)))
    b_id = c2.selectbox("Later run", ids, index=0, key=f"hcmp_b_{subject.id}", format_func=lambda i: sh.analysis_label(next(x for x in items if x["id"] == i)))
    if a_id == b_id:
        st.caption("Choose two different runs.")
        return
    ra, rb = (sh.load_for_subject(svc, i, subject.id) for i in (a_id, b_id))
    if not ra or not rb:
        return
    ma, mb = ra[1]["metrics"], rb[1]["metrics"]
    rows = [{"Measure": label, "Earlier": ma[k], "Later": mb[k], "Change": round(_delta(ma[k], mb[k]), 2)} for label, k in
            [("Questions", "total_questions"), ("Sections", "total_sections"), ("Strong", "strong_count"), ("Moderate", "moderate_count"),
             ("Weak", "weak_count"), ("Unmatched", "unmatched_count"), ("Question match rate (%)", "overall_coverage_rate")]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    sa, sb = ra[1]["settings"], rb[1]["settings"]
    diffs = [f"{k}: {sa.get(k)} -> {sb.get(k)}" for k in ("vectorizer", "ngram_range", "use_stemming", "cross_subject") if sa.get(k) != sb.get(k)]
    da, db = {d["filename"] for d in ra[1]["documents"]}, {d["filename"] for d in rb[1]["documents"]}
    pa, pb = {p["title"] for p in ra[1]["papers"]}, {p["title"] for p in rb[1]["papers"]}
    notes = diffs + ([f"Lecture documents added: {', '.join(sorted(db - da))}"] if db - da else []) + ([f"Lecture documents removed: {', '.join(sorted(da - db))}"] if da - db else []) \
        + ([f"Papers differ: {', '.join(sorted(pa ^ pb))}"] if pa != pb else [])
    st.caption("Differences in inputs: " + ("; ".join(notes) if notes else "none (same settings, papers and documents)."))
    st.caption("Changes in these numbers can come from different inputs or settings; they are not a measure of improvement.")


def render() -> None:
    svc = state.get_services()
    ui.page_header("Analysis History", "Saved snapshots of earlier analyses for this subject. They never change after they are saved.")
    subject = state.require_subject(svc)
    state.show_scope(svc, subject)
    items = svc.history.list(subject.id, limit=200)
    if not items:
        ui.empty_state("No saved analyses", "Run an analysis on the Question Analysis page. Every run is saved here with the exact documents and settings it used.")
        return
    ui.metric_grid([("Saved analyses", len(items), "For " + subject.name)])
    for a in items[:25]:
        status = svc.history.input_status(a["id"])
        papers = ", ".join(p["title"] for p in a["inputs"].get("papers", []))
        badges = (ui.badge("Cross-subject", "warning") if a["cross_subject"] else "") + " " + (ui.badge("Inputs changed since", "warning") if status["status"] != "current" else ui.badge("Inputs unchanged", "success"))
        ui.render(f'<div class="sl-card"><div class="sl-card-title">{ui.esc(a["name"])}</div><div>{badges}</div><div class="sl-card-meta">'
                  f'<span>{ui.esc(ui.fmt_dt(a["created_at"]))}</span><span>{ui.esc(a["method"])}</span><span>Papers: {ui.esc(papers)}</span>'
                  f'<span>{len(a["inputs"].get("documents", []))} lecture document(s)</span></div></div>')
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("Open", key=f"hist_open_{a['id']}"):
            state.set_last_analysis_id(subject.id, a["id"])
            st.session_state[state.sk(subject.id, "history_open")] = a["id"]
        with c2.popover("Delete"):
            st.caption("Removes this saved snapshot permanently. Your documents are not affected.")
            if st.button("Delete snapshot", key=f"hist_del_{a['id']}"):
                svc.history.delete(a["id"])
                if state.get_last_analysis_id(subject.id) == a["id"]:
                    st.session_state.pop(state.sk(subject.id, "last_analysis"), None)
                st.session_state.pop(state.sk(subject.id, "history_open"), None)
                st.rerun()
    if len(items) > 25:
        st.caption(f"Showing the newest 25 of {len(items)}.")
    if len(items) >= 2:
        with st.expander("Compare two runs"):
            _compare_two(items, svc, subject)
    opened = sh.load_for_subject(svc, st.session_state.get(state.sk(subject.id, "history_open")), subject.id)
    if opened:
        st.divider()
        ui.section_heading(f"Snapshot: {opened[0]['name']}")
        render_results(svc, subject, *opened, key="hist")
