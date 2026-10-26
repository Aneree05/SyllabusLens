"""Diagnostics: dependency health, what each failure affects, storage status and app information."""
from __future__ import annotations

import config
import streamlit as st

from components import state, ui
from services import diagnostics_service as dg

AFFECTS = {
    "numpy": "All analysis (numerical operations).",
    "scipy": "Analysis, subject suggestion and evaluation (scikit-learn requires SciPy). Browsing and the NLP Lab's preprocessing steps do not.",
    "scikit-learn": "Analysis, subject suggestion and evaluation (vectorisation and cosine similarity).",
    "pandas": "Tables and CSV exports.",
    "plotly": "All charts. Tables and text results still work.",
    "pypdf": "Uploading and reading PDF files. TXT, MD and PPTX still work.",
    "python-pptx": "Uploading and reading PPTX files. PDF, TXT and MD still work.",
    "streamlit": "The whole interface.",
    "TF-IDF + cosine similarity": "Analysis and subject suggestion.",
    "PPTX extraction": "PPTX uploads.", "PDF parsing": "PDF uploads.", "Data directory writable": "Saving subjects, documents and analyses.",
}


def _mask(text: str) -> str:
    return text.replace(str(config.DATA_DIR), "<data directory>")


@st.cache_data(show_spinner=False, ttl=60)
def _run():
    rep = dg.run_diagnostics(config.DATA_DIR)
    return rep


def render() -> None:
    ui.page_header("Diagnostics", "Live checks of the libraries, file extraction and storage this app depends on.")
    if st.button("Re-run checks", key="diag_rerun"):
        _run.clear()
    rep = _run()
    failed = rep.failed()
    if rep.ok:
        ui.notice("success", "All checks passed", "Every dependency imported and the extraction/analysis probes ran. This describes this environment only.")
    else:
        ui.notice("error", f"{len(failed)} check(s) failed", "Features that depend on them are listed below.")
    if rep.policy_blocked:
        ui.notice("warning", "A security policy appears to be blocking a native library", rep.guidance)
        st.caption("SyllabusLens never bypasses or disables Windows Application Control, AppLocker or similar protections.")
    rows = [{"Check": c.name, "Status": "OK" if c.ok else "FAILED", "Version": c.version or "-",
             "Detail": _mask(c.detail), "Affects when failing": "" if c.ok else AFFECTS.get(c.name, "Related features.")} for c in rep.checks]
    st.dataframe(rows, width="stretch", hide_index=True)
    if failed:
        st.markdown("**What still works**")
        for c in failed:
            st.markdown(f"- If **{c.name}** is unavailable: {AFFECTS.get(c.name, 'related features are limited')}".replace("$", "\\$"))

    ui.section_heading("Application")
    svc = None
    try:
        svc = state.get_services()
    except (ImportError, OSError):
        pass
    facts = [("Version", config.APP_VERSION), ("Python", rep.python), ("Platform", rep.platform),
             ("Upload limit", f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB per file"), ("Accepted files", ", ".join(config.ALLOWED_UPLOAD_EXTENSIONS))]
    if svc is not None:
        facts += [("Subjects", len(svc.subjects.list())), ("Documents", svc.library.stats()["documents"])]
    st.dataframe([{"Setting": k, "Value": str(v)} for k, v in facts], width="stretch", hide_index=True)
    ui.render('<p class="sl-caption">Storage is local to this installation (SQLite plus a private upload folder, never served publicly). There is no sign-in, '
              'so anyone who can open this app can see its data. See the README for what a multi-user deployment would need.</p>')
    ui.render('<p class="sl-caption"><strong>Verified environment.</strong> The test suite has been run on Linux with Python 3.12. Windows has not been tested; '
              'use this page to check your own environment.</p>')
