"""Subjects: create, select, edit and delete academic subject workspaces."""
from __future__ import annotations

import streamlit as st

from components import nav, state, ui
from services.errors import ConfirmationError, SyllabusLensError
from services.sample_data_service import SAMPLE_SUBJECT_NAME, install_sample_subject


def _topics(text: str):
    return [t for t in (text or "").splitlines() if t.strip()]


def _create_form(svc, first: bool) -> None:
    with st.form("create_subject", clear_on_submit=True):
        name = st.text_input("Subject name", placeholder="e.g. Natural Language Processing", max_chars=120)
        c1, c2 = st.columns(2)
        code = c1.text_input("Course code (optional)", placeholder="e.g. CS801")
        sem = c2.text_input("Semester (optional)", placeholder="e.g. 8")
        desc = st.text_area("Description (optional)", height=80,
                            help="Used, together with topics, to suggest which subject an uploaded file belongs to.")
        topics = st.text_area("Syllabus topics (optional, one per line)", height=120,
                              help="Topics enable syllabus-gap analysis: which topics no question in your selected papers touches.")
        if st.form_submit_button("Create subject", type="primary"):
            try:
                s = svc.subjects.create(name, desc, code, sem, _topics(topics))
            except SyllabusLensError as exc:
                ui.notice("error", "Subject was not created", str(exc))
                return
            state.set_active_subject(s.id)
            st.toast(f"Created '{s.name}'. Next: upload its notes in the Document Library.")
            st.rerun()


def _subject_card(svc, s, active_id: str) -> None:
    ov = state.subject_overview(svc, s)
    meta = " ".join(x for x in (s.course_code, f"Semester {s.semester}" if s.semester else "") if x)
    with ui.card(f"subject-{s.id}"):
        head = f'<div class="sl-card-title">{ui.esc(s.name)} {ui.badge("Active", "primary") if s.id == active_id else ""}</div>'
        body = f'<div class="sl-caption">{ui.esc(s.description) if s.description else "No description."}</div>' if True else ""
        ui.render(head + (f'<div class="sl-card-meta"><span>{ui.esc(meta)}</span></div>' if meta else "") + body +
                  f'<div class="sl-card-meta"><span>{ov["notes"]} notes/slides</span><span>{ov["papers"]} question papers</span>'
                  f'<span>{len(s.topics)} syllabus topics</span><span>{ov["analyses"]} saved analyses</span></div>')
        if s.id != active_id and st.button("Open this subject", key=f"open_{s.id}"):
            state.set_active_subject(s.id)
            st.rerun()


def _edit_panel(svc, s) -> None:
    with st.expander("Edit details and syllabus topics"):
        with st.form(f"edit_{s.id}"):
            name = st.text_input("Subject name", value=s.name, max_chars=120)
            c1, c2 = st.columns(2)
            code = c1.text_input("Course code", value=s.course_code)
            sem = c2.text_input("Semester", value=s.semester)
            desc = st.text_area("Description", value=s.description, height=80)
            topics = st.text_area("Syllabus topics (one per line)", value="\n".join(s.topics), height=150)
            if st.form_submit_button("Save changes", type="primary"):
                try:
                    svc.subjects.update(s.id, name=name, description=desc, course_code=code, semester=sem, topics=_topics(topics))
                except SyllabusLensError as exc:
                    ui.notice("error", "Changes were not saved", str(exc))
                else:
                    st.toast("Subject updated. Saved analyses keep the topics they were run with.")
                    st.rerun()


def _danger_zone(svc, s) -> None:
    counts = svc.subjects.counts(s.id)
    with st.expander("Delete this subject"):
        ui.notice("warning", "This cannot be undone",
                  f"Permanently deletes {counts['documents']} document(s), their stored files and extracted text, and "
                  f"{counts['analyses']} saved analysis snapshot(s). Other subjects are not affected.")
        typed = st.text_input(f"Type the subject name ({s.name}) to confirm", key=f"del_confirm_{s.id}")
        if st.button("Delete subject permanently", key=f"del_btn_{s.id}", disabled=typed.strip() != s.name):
            try:
                removed = svc.subjects.delete(s.id, typed)
            except ConfirmationError as exc:
                ui.notice("error", "Nothing was deleted", str(exc))
                return
            state.set_active_subject(None)
            st.toast(f"Deleted '{s.name}' ({removed['documents']} documents, {removed['analyses']} analyses).")
            st.rerun()


def render() -> None:
    svc = state.get_services()
    ui.page_header("Subjects", "Each subject is a separate workspace with its own documents, analyses and history.")
    subjects = svc.subjects.list()
    active = state.get_active_subject(svc)

    if not subjects:
        ui.empty_state("Create your first subject",
                       "Start with one course, for example Natural Language Processing. You can add more at any time; "
                       "their documents will never mix unless you explicitly ask.")
        left, right = st.columns([3, 2])
        with left:
            ui.section_heading("New subject")
            _create_form(svc, first=True)
        with right:
            ui.section_heading("Or explore with demo data")
            ui.render('<p class="sl-caption">Adds synthetic NLP lecture notes and an end-semester paper, clearly labelled as sample data, '
                      'so you can try the full workflow before uploading your own material.</p>')
            if st.button("Load demo subject", key="subjects_demo"):
                state.set_active_subject(install_sample_subject(svc).id)
                st.rerun()
        return

    ui.section_heading("Your subjects")
    for s in subjects:
        _subject_card(svc, s, active.id if active else "")
        if active and s.id == active.id:
            _edit_panel(svc, s)
            _danger_zone(svc, s)
    with st.expander("Add another subject", expanded=False):
        _create_form(svc, first=False)
    if not any(s.name == SAMPLE_SUBJECT_NAME for s in subjects):
        with st.expander("Demo data"):
            ui.render('<p class="sl-caption">Adds a clearly labelled synthetic NLP subject for trying the workflow.</p>')
            if st.button("Load demo subject", key="subjects_demo2"):
                state.set_active_subject(install_sample_subject(svc).id)
                st.rerun()
    if active:
        if st.button("Go to Document Library", key="subjects_to_lib", type="primary"):
            nav.goto("library")
