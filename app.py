"""
SyllabusLens - academic intelligence workspace.

Entry point: configures the page, injects the theme, wires services, renders the sidebar and runs the
selected page inside an error boundary. Page logic lives in pages_content/, UI building blocks in
components/, business logic in services/ and nlp/.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import streamlit as st

import config

st.set_page_config(page_title="SyllabusLens", page_icon=":material/school:", layout="wide", initial_sidebar_state="auto")

from components import nav, state, ui  # noqa: E402
from components.sidebar import render_nav, render_sidebar  # noqa: E402
from components.theme import inject_theme  # noqa: E402
from pages_content import PAGE_SPECS, renderer  # noqa: E402
from services.errors import SyllabusLensError  # noqa: E402

log = logging.getLogger("syllabuslens")


def _configure_logging() -> None:
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(config.DATA_DIR / "syllabuslens.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)


def main() -> None:
    _configure_logging()
    inject_theme()
    degraded = None
    try:
        svc = state.get_services()
    except (ImportError, OSError) as exc:      # e.g. a native library blocked by Windows Application Control
        log.exception("Core services failed to load")
        degraded, svc = exc, None
    specs = PAGE_SPECS if degraded is None else [s for s in PAGE_SPECS if s.key == "diagnostics"]
    pages = {spec.key: st.Page(renderer(spec), title=spec.title, icon=spec.icon, url_path=spec.key,
                               default=(i == 0)) for i, spec in enumerate(specs)}
    nav.PAGES.clear()
    nav.PAGES.update(pages)
    sections: dict = {}
    for spec in specs:
        sections.setdefault(spec.section, []).append(pages[spec.key])
    current = st.navigation(sections, position="hidden")      # navigation is drawn by render_sidebar, below the subject selector
    if degraded is None:
        render_sidebar(svc, sections)
    else:
        with st.sidebar:
            ui.render('<div class="sl-brand">SyllabusLens</div><div class="sl-brand-sub">Limited mode</div>')
        render_nav(sections)
        ui.notice("error", "Analysis features are unavailable",
                  "A required library failed to load, so documents, subjects and analysis are disabled. The Diagnostics page below explains what failed and the safe ways to fix it.")
    try:
        current.run()
    except SyllabusLensError as exc:                       # expected, user-actionable
        ui.notice("error", "That action could not be completed", str(exc))
    except Exception:                                       # unexpected: log details, show a safe message
        log.exception("Unhandled error while rendering %s", getattr(current, "url_path", "?"))
        ui.notice("error", "Something went wrong", "The problem was logged. Reload the page; if it persists, check Diagnostics.")


main()
