import pytest

from services import overview_service as ov
from services.sample_data_service import install_sample_subject
from tests.ui_harness import md, open_page

PAGE = "pages_content.overview"


def _ok(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_empty_database_gets_onboarding_not_zero_cards(svc):
    at = open_page(svc, PAGE); _ok(at)
    page = md(at)
    assert "Start with a subject" in page and "sl-metric" not in page
    assert {b.label for b in at.button} == {"Create a subject", "Try with demo data"}


def test_demo_button_creates_labelled_subject(svc):
    at = open_page(svc, PAGE)
    next(b for b in at.button if b.label == "Try with demo data").click().run(); _ok(at)
    assert svc.subjects.list()[0].name.endswith("(sample data)") and "Next:" in md(at)


def test_subjects_but_none_active_prompts_and_never_picks(svc, world):
    at = open_page(svc, PAGE); _ok(at)
    assert "Select a subject" in md(at) and "active_subject_id" not in at.session_state


@pytest.mark.parametrize("setup,expected", [
    ("empty", "Upload lecture notes or slides"), ("notes", "Add a question paper"),
    ("ready", "Run your first analysis"), ("ran", "Add syllabus topics"),
])
def test_next_action_follows_real_state(svc, setup, expected):
    s = svc.subjects.create("S")
    if setup != "empty":
        svc.library.ingest(s.id, "n.txt", b"# Tokenization\ntokenization splits text into tokens", doc_kind="lecture_notes")
    if setup in ("ready", "ran"):
        p = svc.library.ingest(s.id, "p.txt", b"1. Explain tokenization of text. [5 marks]\n2. Describe tokens. [5 marks]", doc_kind="question_paper").document
    if setup == "ran":
        svc.runner.run(s.id, [p.id])
    r = ov.readiness(svc, svc.subjects.get(s.id))
    assert ov.next_action(r, "S").title == expected
    at = open_page(svc, PAGE, active=s.id); _ok(at)
    assert expected in md(at)


def test_changed_inputs_trigger_rerun_suggestion(svc, world):
    svc.runner.run(world["nlp"].id, [world["nlp_p1"].id])
    svc.library.replace(world["nlp_docs"][0].id, "notes.txt", b"# Changed\nsomething else entirely")
    r = ov.readiness(svc, svc.subjects.get(world["nlp"].id))
    assert ov.next_action(r, "NLP").key == "rerun"


def test_dashboard_numbers_come_from_the_active_subject_only(svc, world):
    svc.runner.run(world["nlp"].id, [world["nlp_p1"].id, world["nlp_p2"].id])
    at = open_page(svc, PAGE, active=world["os"].id); _ok(at)
    page = md(at)
    assert "Latest analysis" not in page and "No analysis yet" in page            # NLP's run is not shown under OS
    at = open_page(svc, PAGE, active=world["nlp"].id); _ok(at)
    assert "Latest analysis" in md(at) and "Revision priorities" in md(at)


def test_other_subjects_listed_by_recent_activity(svc, world):
    svc.runner.run(world["os"].id, [world["os_p1"].id])
    rows = ov.subjects_by_recent_activity(svc)
    assert rows[0][0].id == world["os"].id


def test_scope_bar_pluralises_counts(svc):
    s = svc.subjects.create("S")
    svc.library.ingest(s.id, "n.txt", b"# T\nsome tokenization text here", doc_kind="lecture_notes")
    svc.library.ingest(s.id, "p.txt", b"1. Explain tokenization. [5 marks]\n2. Describe text. [5 marks]\n3. Define tokens. [5 marks]", doc_kind="question_paper")
    page = md(open_page(svc, "pages_content.library", active=s.id))
    assert "1 lecture/reference doc<" in page and "1 question paper<" in page
    svc.library.ingest(s.id, "n2.txt", b"# U\nmore parsing content here", doc_kind="lecture_notes")
    assert "2 lecture/reference docs<" in md(open_page(svc, "pages_content.library", active=s.id))
