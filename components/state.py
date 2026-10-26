"""
Application state: service access, the active subject, and subject-scoped session values.

Isolation in the UI is *structural*: every session value that belongs to a subject (last analysis,
upload staging, selections) is stored under a key containing that subject's ID, so a value created
for subject A cannot be read while subject B is active. The backend (RetrievalService) remains the
authority for what an analysis may read; the UI only supplies the active subject ID.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import streamlit as st

from components import ui
from services.subject_service import Subject

if TYPE_CHECKING:     # imported lazily: the container pulls in scikit-learn/SciPy, which may fail to load
    from services.container import Services

ACTIVE_KEY = "active_subject_id"
OVERRIDE_KEY = "_services_override"     # test seam: AppTest injects an isolated Services here
SCOPED_PREFIX = "scoped."


@st.cache_resource(show_spinner=False)
def _shared_services() -> "Services":
    from services import container
    return container.build_services()


def get_services() -> "Services":
    override = st.session_state.get(OVERRIDE_KEY)
    return override if override is not None else _shared_services()


# ---------------------------------------------------------------- active subject
def get_active_subject(svc: "Services") -> Optional[Subject]:
    """The active subject if it still exists. Never falls back to some other subject."""
    sid = st.session_state.get(ACTIVE_KEY)
    if not sid:
        return None
    if not svc.subjects.exists(sid):
        st.session_state.pop(ACTIVE_KEY, None)       # deleted elsewhere: clear, do not substitute
        purge_scoped(sid)
        return None
    return svc.subjects.get(sid)


def set_active_subject(subject_id: Optional[str]) -> None:
    """Change the active subject. Scoped values of the previous subject are dropped."""
    previous = st.session_state.get(ACTIVE_KEY)
    if previous and previous != subject_id:
        purge_scoped(previous)
    if subject_id:
        st.session_state[ACTIVE_KEY] = subject_id
    else:
        st.session_state.pop(ACTIVE_KEY, None)


def sk(subject_id: str, name: str) -> str:
    """Session-state key scoped to one subject."""
    return f"{SCOPED_PREFIX}{subject_id}.{name}"


def purge_scoped(subject_id: str) -> None:
    prefix = f"{SCOPED_PREFIX}{subject_id}."
    for k in [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[k]


def get_last_analysis_id(subject_id: str) -> Optional[str]:
    return st.session_state.get(sk(subject_id, "last_analysis"))


def set_last_analysis_id(subject_id: str, analysis_id: str) -> None:
    st.session_state[sk(subject_id, "last_analysis")] = analysis_id


def require_subject(svc: "Services") -> Subject:
    """Render an onboarding/selection state and stop the page when no subject is active."""
    subject = get_active_subject(svc)
    if subject is not None:
        return subject
    from components import nav
    subjects = svc.subjects.list()
    if not subjects:
        ui.empty_state("Create your first subject",
                       "A subject keeps one course's notes, question papers and analyses together and separate from every other course.")
        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button("Create a subject", type="primary", key="req_create"):
            nav.goto("subjects")
        if c2.button("Load demo subject", key="req_demo", help="Adds clearly labelled synthetic NLP material."):
            from services.sample_data_service import install_sample_subject
            set_active_subject(install_sample_subject(svc).id)
            st.rerun()
    else:
        ui.empty_state("Select a subject to continue",
                       "Choose the subject you want to work on in the sidebar. SyllabusLens never picks one for you.")
    st.stop()


def subject_overview(svc: "Services", subject: Subject) -> Dict[str, Any]:
    """Counts used by the scope bar and dashboard (single query path)."""
    stats = svc.library.stats(subject.id)
    kinds = stats["by_kind"]
    notes = sum(n for k, n in kinds.items() if k in ("lecture_notes", "lecture_slides", "tutorial", "reference"))
    return dict(stats=stats, papers=kinds.get("question_paper", 0), notes=notes,
                syllabus_docs=kinds.get("syllabus", 0), analyses=svc.subjects.counts(subject.id)["analyses"])


def show_scope(svc: "Services", subject: Subject, cross_subject_names: List[str] = ()) -> Dict[str, Any]:
    ov = subject_overview(svc, subject)
    ui.scope_bar(subject.name, ov["notes"], ov["papers"], course_code=subject.course_code,
                 cross_subject_names=list(cross_subject_names))
    return ov
