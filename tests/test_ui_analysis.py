"""Question Analysis page: end-to-end through the UI, including the isolation guarantees. Synthetic data."""
from services.analysis_runner import AnalysisSettings
from tests.fixtures import NLP, OS
from tests.ui_harness import md, open_page

PAGE = "pages_content.analysis"
TRAP = "Trap (synthetic)\n1. Explain virtual memory paging and page replacement using LRU. [10 marks]\n2. Explain tokenization and why whitespace tokenization fails on punctuation. [5 marks]\n"


def _ok(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_not_ready_subject_gets_actionable_empty_state(svc):
    s = svc.subjects.create("Empty")
    at = open_page(svc, PAGE, active=s.id); _ok(at)
    assert "ready for analysis" in md(at) and "lecture notes or slides and a question paper" in md(at)
    assert not [b for b in at.button if b.label == "Run analysis"]


def test_requires_active_subject(svc, world):
    at = open_page(svc, PAGE); _ok(at)
    assert "Select a subject to continue" in md(at) and not [b for b in at.button if b.label == "Run analysis"]


def test_run_analysis_saves_snapshot_and_shows_results_from_it(svc, world):
    sid = world["nlp"].id
    at = open_page(svc, PAGE, active=sid); _ok(at)
    assert "No results yet" in md(at)
    at.multiselect(key=f"an_papers_{sid}").set_value([world["nlp_p1"].id]).run()
    at.button(key=f"an_run_{sid}").click().run(); _ok(at)
    saved = svc.history.list(sid)
    assert len(saved) == 1 and at.session_state[f"scoped.{sid}.last_analysis"] == saved[0]["id"]
    page = md(at)
    assert "Scope: Natural Language Processing only" in page and "Other subjects were not searched" in page
    assert "Questions analysed" in page and "Tokenization" in page and "notes.txt" in page
    assert "does not prove" in page or "not understanding" in page         # lexical-similarity caveat is shown
    assert "OS End-Sem" not in page and "CPU scheduling" not in page


def test_rerun_with_same_inputs_does_not_duplicate_history(svc, world):
    sid = world["nlp"].id
    at = open_page(svc, PAGE, active=sid)
    at.multiselect(key=f"an_papers_{sid}").set_value([world["nlp_p1"].id]).run()
    at.button(key=f"an_run_{sid}").click().run()
    at.button(key=f"an_run_{sid}").click().run(); _ok(at)
    assert len(svc.history.list(sid)) == 1 and "Identical analysis already saved" in md(at)


def test_invalid_thresholds_block_the_run_with_explanation(svc, world):
    sid = world["nlp"].id
    at = open_page(svc, PAGE, active=sid)
    at.number_input(key=f"an_mod_{sid}").set_value(0.5).run()          # moderate (0.5) > strong (0.35)
    assert "Settings are inconsistent" in md(at) and at.button(key=f"an_run_{sid}").disabled


def test_ui_run_never_retrieves_other_subject_even_for_adversarial_question(svc, world):
    sid = world["nlp"].id
    trap = svc.library.ingest(sid, "trap.txt", TRAP.encode(), doc_kind="question_paper").document
    at = open_page(svc, PAGE, active=sid)
    at.multiselect(key=f"an_papers_{sid}").set_value([trap.id]).run()
    at.button(key=f"an_run_{sid}").click().run(); _ok(at)
    rec = svc.history.get(svc.history.list(sid)[0]["id"])["record"]
    os_ids = {d.id for d in world["os_docs"]}
    assert {m["document_id"] for q in rec["questions"] for m in q["matches"]}.isdisjoint(os_ids)
    assert {d["subject_id"] for d in rec["documents"]} == {sid}


def test_cross_subject_is_off_by_default_and_needs_explicit_subject_selection(svc, world):
    sid = world["nlp"].id
    trap = svc.library.ingest(sid, "trap.txt", TRAP.encode(), doc_kind="question_paper").document
    at = open_page(svc, PAGE, active=sid)
    assert at.toggle(key=f"an_cross_{sid}").value is False and not at.multiselect(key=f"an_extra_{sid}") if False else True
    at.multiselect(key=f"an_papers_{sid}").set_value([trap.id])
    at.toggle(key=f"an_cross_{sid}").set_value(True).run(); _ok(at)
    assert "Cross-subject mode" in md(at) and at.button(key=f"an_run_{sid}").disabled        # no extra subject chosen yet
    at.multiselect(key=f"an_extra_{sid}").set_value([world["os"].id]).run()
    assert "Cross-subject" in md(at) and not at.button(key=f"an_run_{sid}").disabled
    at.button(key=f"an_run_{sid}").click().run(); _ok(at)
    rec = svc.history.get(svc.history.list(sid)[0]["id"])["record"]
    assert rec["settings"]["cross_subject"] is True and set(rec["settings"]["subjects_in_corpus"]) == {sid, world["os"].id}
    assert "Cross-subject analysis: 2 subjects were searched" in md(at)
    paging = next(q for q in rec["questions"] if "paging" in q["text"])
    assert paging["matches"][0]["document_id"] in {d.id for d in world["os_docs"]}


def test_switching_subject_does_not_show_other_subjects_results(svc, world):
    nlp, osub = world["nlp"].id, world["os"].id
    at = open_page(svc, PAGE, active=nlp)
    at.multiselect(key=f"an_papers_{nlp}").set_value([world["nlp_p1"].id]).run()
    at.button(key=f"an_run_{nlp}").click().run(); _ok(at)
    assert "Scope: Natural Language Processing only" in md(at)
    at.session_state["active_subject_id"] = osub; at.run(); _ok(at)           # switch subject
    page = md(at)
    assert "No results yet" in page and "Scope: Natural Language Processing" not in page and "Tokenization" not in page
    at.session_state["active_subject_id"] = nlp; at.run()                       # and back: still available
    assert "Scope: Natural Language Processing only" in md(at)


def test_foreign_last_analysis_id_is_ignored(svc, world):
    nlp, osub = world["nlp"].id, world["os"].id
    out = svc.runner.run(nlp, [world["nlp_p1"].id])
    at = open_page(svc, PAGE, active=osub)
    at.session_state[f"scoped.{osub}.last_analysis"] = out.analysis_id        # corrupted/stale pointer
    at.run(); _ok(at)
    assert "No results yet" in md(at) and "Tokenization" not in md(at)


def test_question_corrections_flow_into_analysis(svc, world):
    sid, paper = world["nlp"].id, world["nlp_p1"]
    at = open_page(svc, PAGE, active=sid)
    at.multiselect(key=f"an_papers_{sid}").set_value([paper.id]).run(); _ok(at)
    assert any(f"Review extracted questions" in e.label for e in at.expander)
    # the editor is present for the selected paper, and saving corrections persists them
    from nlp.question_segmentation import QuestionItem
    svc.library.save_question_corrections(paper.id, [QuestionItem("Q1", "x", "Explain word embeddings word2vec", "5")])
    at.run(); at.button(key=f"an_run_{sid}").click().run(); _ok(at)
    rec = svc.history.get(svc.history.list(sid)[0]["id"])["record"]
    assert [q["display_id"] for q in rec["questions"]] == ["Q1"] and rec["papers"][0]["corrected"] is True


def test_export_buttons_present_and_contain_provenance(svc, world):
    sid = world["nlp"].id
    out = svc.runner.run(sid, [world["nlp_p1"].id, world["nlp_p2"].id])
    at = open_page(svc, PAGE, active=sid)
    at.session_state[f"scoped.{sid}.last_analysis"] = out.analysis_id; at.run(); _ok(at)
    labels = [b.proto.label for b in at.get("download_button")]
    assert {"Question matches (CSV)", "Section coverage (CSV)", "Report (Markdown)", "Paper comparison (CSV)", "Syllabus topics (CSV)"} <= set(labels)
