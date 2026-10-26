"""Diagnostics page and graceful degradation when a native library cannot load."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

from tests.ui_harness import md, open_page


def _ok(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_diagnostics_page_reports_real_checks_and_limits(svc):
    at = open_page(svc, "pages_content.diagnostics"); _ok(at)
    page = md(at)
    assert "All checks passed" in page and "Windows has not been tested" in page and "no sign-in" in page
    names = {r["Check"] for r in at.dataframe[0].value.to_dict("records")}
    assert {"scipy", "scikit-learn", "TF-IDF + cosine similarity"} <= names
    assert str(svc.data_dir) not in page and not any(str(svc.data_dir) in str(v) for v in at.dataframe[0].value.values.ravel())


def test_app_degrades_to_diagnostics_when_native_library_is_blocked(monkeypatch):
    import services.container as container
    def blocked(*a, **k):
        raise OSError("[WinError 4551] An Application Control policy has blocked this file: scipy\\_lib\\x.pyd")
    monkeypatch.setattr(container, "build_services", blocked)
    at = AppTest.from_file(APP, default_timeout=60).run()
    _ok(at)
    page = md(at)
    assert "Analysis features are unavailable" in page and "Limited mode" in page
    assert "Diagnostics" in page                                   # the diagnostics page is what renders
    assert not [b for b in at.button if b.label in ("Create subject", "Run analysis")]
    assert not at.selectbox                                         # no subject selector in limited mode


def test_full_shell_starts_cleanly_with_services(svc):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["_services_override"] = svc
    at.run(); _ok(at)
    assert "Start with a subject" in md(at) and "Limited mode" not in md(at)
