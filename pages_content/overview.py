"""Overview: where the active subject stands and what to do next."""
from __future__ import annotations

import streamlit as st

from components import nav, state, ui
from services import overview_service as ov
from services.sample_data_service import install_sample_subject
from services.study_plan_service import build_plan


def _subject_picker_cards(svc, active_id) -> None:
    rows = ov.subjects_by_recent_activity(svc)
    ui.section_heading("Subjects" if not active_id else "Other subjects", "Most recently used first.")
    for s, last in [r for r in rows if r[0].id != active_id][:6]:
        with ui.card(f"ov-subj-{s.id}"):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{s.name}**".replace("$", "\\$"))
            c1.caption(f"Last activity: {ui.fmt_dt(last)}" if last else "No activity yet")
            if c2.button("Open", key=f"ov_open_{s.id}"):
                state.set_active_subject(s.id); st.rerun()


def _onboarding(svc) -> None:
    ui.page_header("Welcome to SyllabusLens", "Organise your course material by subject, then see how your question papers relate to your notes.")
    ui.empty_state("Start with a subject", "A subject holds one course's notes, question papers and analyses, kept separate from every other course.")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Create a subject", type="primary", key="ov_create"):
        nav.goto("subjects")
    if c2.button("Try with demo data", key="ov_demo", help="Adds a clearly labelled synthetic NLP subject."):
        state.set_active_subject(install_sample_subject(svc).id); st.rerun()
    ui.render('<p class="sl-caption">How it works: 1. Create a subject. 2. Upload its lecture notes and past question papers. '
              '3. Run an analysis to see which sections each question relates to, what the papers never touch, and what to revise.</p>')


def render() -> None:
    svc = state.get_services()
    if not svc.subjects.list():
        return _onboarding(svc)
    subject = state.get_active_subject(svc)
    if subject is None:
        ui.page_header("Overview", "Choose a subject to see its status.")
        ui.empty_state("Select a subject", "Pick one in the sidebar or open one below. SyllabusLens never chooses for you.")
        _subject_picker_cards(svc, None)
        return
    ui.page_header(subject.name, subject.description or "Overview of this subject workspace.")
    state.show_scope(svc, subject)
    r = ov.readiness(svc, subject)
    action = ov.next_action(r, subject.name)
    with ui.card("ov-next"):
        ui.render(f'<div class="sl-card-title">Next: {ui.esc(action.title)}</div><p class="sl-caption">{ui.esc(action.body)}</p>')
        if st.button(action.title, type="primary", key="ov_next_btn"):
            nav.goto(action.target)
    ui.metric_grid([("Lecture documents", r.notes, "Readable notes, slides, tutorials and references"), ("Question papers", r.papers, "Previous-year papers in this subject"),
                    ("Indexed sections", r.sections, "Units of lecture material that can be matched"), ("Saved analyses", r.analyses, "In Analysis History"),
                    ("Syllabus topics", r.topics, "Configured on the Subjects page"),
                    ("Unreadable files", r.failures, "Failed or no extractable text", "danger" if r.failures else "")])
    if r.latest is None:
        ui.empty_state("No analysis yet", "Findings from your latest analysis will appear here.")
    else:
        loaded = svc.history.get(r.latest["id"])
        rec = loaded["record"]
        m = rec["metrics"]
        ui.section_heading("Latest analysis", f"{loaded['name']} - {ui.fmt_dt(loaded['created_at'])}")
        if r.latest_inputs_changed:
            ui.notice("warning", "Documents changed since this analysis", "Results below are the original snapshot.")
        ui.metric_grid([("Question match rate", f"{m['overall_coverage_rate']:.0f}%", "Questions with a Strong or Moderate best match"),
                        ("Unmatched questions", m["unmatched_count"], "No section shares enough vocabulary", "danger" if m["unmatched_count"] else ""),
                        ("Sections not examined", m["sections_uncovered_count"], "No question resembles them", "warning" if m["sections_uncovered_count"] else "")])
        un = [q for q in rec["questions"] if q["best_tier"] == "Unmatched"]
        if un:
            st.markdown("**Questions needing attention**")
            for q in un[:4]:
                ui.render(ui.question_card_html(q, show_paper=len(rec["papers"]) > 1))
            if len(un) > 4:
                st.caption(f"{len(un) - 4} more in Coverage Lab.")
        plan = [x for x in build_plan(rec, svc.history.reviewed_ids(subject.id)) if x["kind"] == "section" and not x["reviewed"]][:3]
        if plan:
            st.markdown("**Revision priorities from this analysis**")
            for x in plan:
                ui.render(f'<div class="sl-card"><div class="sl-card-title">{ui.esc(x["title"])}</div><div class="sl-card-meta">{ui.badge(x["rule_id"], "primary", x["rule"])}'
                          f'<span>{ui.esc("; ".join(x["evidence"][:1]))}</span></div></div>')
        recent = svc.history.list(subject.id, limit=3)
        if len(recent) > 1:
            st.markdown("**Recent analyses**")
            for a in recent:
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"{a['name']}  \n{ui.fmt_dt(a['created_at'])}".replace("$", "\\$"))
                if c2.button("Open", key=f"ov_hist_{a['id']}"):
                    state.set_last_analysis_id(subject.id, a["id"]); nav.goto("analysis")
    if len(svc.subjects.list()) > 1:
        _subject_picker_cards(svc, subject.id)
