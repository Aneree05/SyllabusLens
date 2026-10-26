"""Backend additions made for the UI migration (preview, dedup guard, evaluation mapping, demo data, linguistics)."""
import json

import pytest

from nlp.optional_linguistics import RuleBasedLinguisticsEngine
from services import evaluation_data as ed
from services.analysis_runner import AnalysisSettings
from services.errors import DocumentValidationError
from services.sample_data_service import SAMPLE_SUBJECT_NAME, install_sample_subject


# ---- upload preview: nothing is persisted ----
def test_preview_upload_stores_nothing_and_reports_everything(svc):
    s = svc.subjects.create("S")
    data = b"NLP Exam - Time allowed: 3 hours\n1. Explain tokenization. [5 marks]\n2. Describe TF-IDF. [5 marks]\n3. Define parsing. [5 marks]\n"
    p = svc.library.preview_upload("../p.txt", data)
    assert p.error is None and p.filename == "p.txt" and p.status == "processed"
    assert p.kind_suggestion[0] == "question_paper" and p.question_count == 3 and p.file_hash
    assert svc.library.list_documents(None) == [] and not any(svc.files.root.rglob("*.txt"))


@pytest.mark.parametrize("name,data", [("x.exe", b"MZ"), ("e.txt", b""), ("f.pdf", b"nope")])
def test_preview_upload_returns_error_instead_of_raising(svc, name, data):
    p = svc.library.preview_upload(name, data)
    assert p.error and not p.text


def test_duplicates_of_and_get_section(svc):
    a, b = svc.subjects.create("A"), svc.subjects.create("B")
    d1 = svc.library.ingest(a.id, "n.txt", b"# H\nsome words here", doc_kind="lecture_notes").document
    d2 = svc.library.ingest(b.id, "n.txt", b"# H\nsome words here", doc_kind="lecture_notes").document
    dups = svc.library.duplicates_of(d1.file_hash)
    assert {d["subject_name"] for d in dups} == {"A", "B"}
    sec = svc.library.get_section(f"{d1.id}:1")
    assert sec["filename"] == "n.txt" and sec["subject_id"] == a.id and svc.library.get_section("nope") is None


# ---- accidental re-run must not duplicate history ----
def test_identical_rerun_reuses_saved_analysis_unless_forced(svc, world):
    r1 = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id])
    r2 = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id])
    assert r2.reused_saved and r2.analysis_id == r1.analysis_id
    assert len(svc.history.list(world["nlp"].id)) == 1
    r3 = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], force_new=True)
    assert r3.analysis_id != r1.analysis_id and len(svc.history.list(world["nlp"].id)) == 2
    svc.history.delete(r1.analysis_id); svc.history.delete(r3.analysis_id)
    assert svc.runner.run(world["nlp"].id, [world["nlp_p1"].id]).analysis_id not in (r1.analysis_id, r3.analysis_id)
    r4 = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], settings=AnalysisSettings(vectorizer="bow"))
    assert not r4.reused_saved                      # different settings are a different analysis


# ---- demo data + corrected ground truth ----
def test_sample_install_is_idempotent_and_labelled(svc):
    s1, s2 = install_sample_subject(svc), install_sample_subject(svc)
    assert s1.id == s2.id and s1.name == SAMPLE_SUBJECT_NAME and "Not real course content" in s1.description
    docs = svc.library.list_documents(s1.id)
    assert len(docs) == 2 and all("sample-data" in d.tags for d in docs)


def test_ground_truth_v1_ids_are_rejected_loudly():
    with pytest.raises(DocumentValidationError, match="obsolete"):
        ed.load_ground_truth({"notes_document": "n", "paper_document": "p", "judgments": {"Q1": ["n.txt_Sec_2"]}})
    with pytest.raises(DocumentValidationError):
        ed.load_ground_truth(b"{not json")
    with pytest.raises(DocumentValidationError):
        ed.load_ground_truth({"judgments": {}})


def test_every_sample_ground_truth_reference_resolves_to_exactly_one_section(svc):
    sub = install_sample_subject(svc)
    paper = next(d for d in svc.library.list_documents(sub.id) if d.is_question_paper)
    out = svc.runner.run(sub.id, [paper.id], save=False)
    r = ed.resolve(ed.load_ground_truth(ed.SAMPLE_GROUND_TRUTH), out.record)
    assert r.problems == []
    assert len(r.relevant) == 10 and r.relevant[next(k for k in r.relevant if k.endswith(":Q7"))] == []


def test_sample_evaluation_runs_and_reports_real_metrics(svc):
    sub = install_sample_subject(svc)
    docs = svc.library.list_documents(sub.id)
    paper = next(d for d in docs if d.is_question_paper); notes = [d.id for d in docs if not d.is_question_paper]
    res = ed.evaluate_against_truth(svc, sub.id, paper.id, notes, ed.load_ground_truth(ed.SAMPLE_GROUND_TRUTH))
    tf, bow = res["reports"]["TF-IDF"], res["reports"]["Bag-of-Words"]
    assert tf["report"].num_queries == 9 and bow["report"].num_queries == 9      # Q7 has no relevant section
    for rep in (tf["report"], bow["report"]):
        assert 0.0 <= rep.precision_at_1 <= 1.0 and 0.0 <= rep.mrr <= 1.0
    assert tf["coverage"].true_positives + tf["coverage"].false_negatives == 9
    assert tf["coverage"].true_negatives + tf["coverage"].false_positives == 1   # only Q7 is out of syllabus


def test_evaluation_with_unmatched_truth_raises_not_silently_zero(svc):
    sub = install_sample_subject(svc)
    docs = svc.library.list_documents(sub.id)
    paper = next(d for d in docs if d.is_question_paper); notes = [d.id for d in docs if not d.is_question_paper]
    bad = ed.load_ground_truth({"notes_document": "sample_lecture_notes_nlp.txt", "paper_document": "sample_question_paper.txt",
                                "judgments": {"Q99": ["Unit 2:"]}})
    with pytest.raises(DocumentValidationError, match="No ground-truth question"):
        ed.evaluate_against_truth(svc, sub.id, paper.id, notes, bad)


# ---- linguistics engine: pin REAL behaviour (a heuristic, not a trained model) ----
def test_pos_tagger_is_a_suffix_heuristic_with_known_errors():
    e = RuleBasedLinguisticsEngine()
    tags = dict(e.tag_pos(["The", "quickly", "running", "tokenization", "BERT", "42", "is"]))
    assert tags == {"The": "DT", "quickly": "RB", "running": "VBG", "tokenization": "NN", "BERT": "NNP", "42": "CD", "is": "VBZ"}
    assert dict(e.tag_pos(["builds"]))["builds"] == "NNS"            # documented limitation: this is a verb


def test_entity_extractor_is_a_dictionary_lookup_not_ner():
    e = RuleBasedLinguisticsEngine()
    found = {x["normalized"] for x in e.extract_academic_entities("Compare LSTM with BERT and TF-IDF at Stanford University.")}
    assert found == {"lstm", "bert", "tf-idf"}                       # 'Stanford University' is NOT detected
    assert e.extract_academic_entities("") == []
