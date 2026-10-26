"""Sidebar chrome: brand and the active-subject selector. Navigation itself is st.navigation."""
from __future__ import annotations

import streamlit as st

from components import nav, state, ui

SELECTOR_KEY = "subject_selector"


def _on_change() -> None:
    state.set_active_subject(st.session_state.get(SELECTOR_KEY))


def render_sidebar(svc, sections=None) -> None:
    with st.sidebar:
        ui.render('<div class="sl-brand">SyllabusLens</div><div class="sl-brand-sub">Academic intelligence workspace</div>')
        subjects = svc.subjects.list() if svc is not None else []
        active = state.get_active_subject(svc) if svc is not None else None
        if subjects:
            # keep the widget in step with changes made elsewhere (e.g. after creating a subject)
            st.session_state[SELECTOR_KEY] = active.id if active else None
            st.selectbox("Active subject", options=[s.id for s in subjects], index=None,
                         format_func={s.id: s.name for s in subjects}.get, key=SELECTOR_KEY,
                         placeholder="Select a subject", on_change=_on_change,
                         help="Every analysis uses only the documents of this subject.")
        elif svc is not None:
            ui.render('<p class="sl-caption">No subjects yet.</p>')
        if svc is not None and st.button("New subject", key="sidebar_new_subject", width="stretch"):
            nav.goto("subjects")
        render_nav(sections)


def render_nav(sections) -> None:
    """Grouped page links, rendered AFTER the subject selector so the key control stays at the top."""
    for title, pages in (sections or {}).items():
        ui.render(f'<div class="sl-nav-label">{ui.esc(title)}</div>')
        for page in pages:
            st.page_link(page, width="stretch")
