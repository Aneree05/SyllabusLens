from services import diagnostics_service as dg


def test_diagnostics_reports_real_state(tmp_path):
    rep = dg.run_diagnostics(tmp_path)
    assert rep.ok, [(c.name, c.detail) for c in rep.failed()]
    names = {c.name for c in rep.checks}
    assert {"scipy", "scikit-learn", "TF-IDF + cosine similarity", "PPTX extraction"} <= names
    assert not rep.policy_blocked and rep.guidance == ""


def test_missing_module_is_reported_not_hidden():
    c = dg.check_import("definitely_not_installed_pkg", "fake")
    assert not c.ok and "ModuleNotFoundError" in c.detail and not c.blocked_by_policy


def test_application_control_block_is_recognised_and_not_bypassed():
    err = OSError("[WinError 4551] An Application Control policy has blocked this file: 'scipy\\_lib\\_x.pyd'")
    assert dg._looks_like_policy_block(err)
    assert not dg._looks_like_policy_block(ImportError("No module named foo"))
    assert "does not attempt to bypass" in dg.POLICY_GUIDANCE


def test_unwritable_data_dir_detected(tmp_path):
    f = tmp_path / "afile"; f.write_text("x")
    assert not dg.check_storage(f / "sub").ok
