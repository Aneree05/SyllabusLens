import json

from services.sample_data_service import install_sample_subject
from tests.ui_harness import FakeUpload, md, open_page

LAB, EVAL = "pages_content.nlp_lab", "pages_content.evaluation"


def _ok(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def test_lab_renders_without_subject_and_labels_unimplemented_features(svc):
    at = open_page(svc, LAB); _ok(at)
    page = md(at)
    assert "Select a subject" in page and "Illustrative heuristics, not trained models" in page
    assert "Not implemented" in [t.label for t in at.tabs]
    assert "Word embeddings" in page and "purely lexical" in page          # honest about retrieval being lexical


def test_lab_ranking_uses_only_active_subject_corpus(svc, world):
    at = open_page(svc, LAB, active=world["nlp"].id)
    at.text_area(key=f"lab_q_{world['nlp'].id}").input("Explain virtual memory paging and page replacement using LRU").run()
    at.button(key=f"lab_rank_{world['nlp'].id}").click().run(); _ok(at)
    page = md(at)
    assert "Searched" in page and "Natural Language Processing only" in page
    assert "Virtual memory and paging" not in page and "CPU scheduling" not in page      # OS sections cannot appear


def test_lab_ranking_handles_text_with_no_vocabulary_overlap(svc, world):
    at = open_page(svc, LAB, active=world["nlp"].id)
    at.text_area(key=f"lab_q_{world['nlp'].id}").input("zzz qqq xxx").run()
    at.button(key=f"lab_rank_{world['nlp'].id}").click().run(); _ok(at)
    assert "No section shares enough vocabulary" in md(at)


def test_lab_empty_text_does_not_crash(svc):
    at = open_page(svc, LAB)
    at.text_area(key="lab_text").input("").run(); _ok(at)


def test_evaluation_requires_demo_documents_in_subject(svc, world):
    at = open_page(svc, EVAL, active=world["nlp"].id); _ok(at)
    assert "Required documents are not in this subject" in md(at)


def test_evaluation_runs_on_demo_subject_with_caveats(svc):
    sub = install_sample_subject(svc)
    at = open_page(svc, EVAL, active=sub.id); _ok(at)
    at.button(key=f"ev_run_{sub.id}").click().run(); _ok(at)
    page = md(at)
    assert "Retrieval quality" in page and "Small sample: 9 questions" in page and "not statistically meaningful" in page
    rows = at.dataframe[0].value.to_dict("records")
    assert {r["Method"] for r in rows} == {"TF-IDF", "Bag-of-Words"} and all(r["Questions scored"] == 9 for r in rows)


def test_evaluation_rejects_obsolete_ground_truth_upload(svc):
    sub = install_sample_subject(svc)
    old = json.dumps({"notes_document": "n", "paper_document": "p", "judgments": {"Q1.a": ["sample_lecture_notes_nlp.txt_Sec_2"]}}).encode()
    at = open_page(svc, EVAL, active=sub.id, files=[FakeUpload("gt.json", old)])
    at.radio(key=f"ev_src_{sub.id}").set_value("Upload my own (JSON)").run(); _ok(at)
    assert "obsolete section IDs" in md(at)
