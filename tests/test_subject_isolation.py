"""Automated proof that subject isolation holds. All data is synthetic (tests/fixtures.py)."""
import pytest

from services.analysis_runner import AnalysisSettings
from services.errors import (
    CrossSubjectError, DocumentValidationError, InvalidSubjectError,
)
from tests.fixtures import DBMS

TRAP_PAPER = """Trap Paper (synthetic)
1. Explain virtual memory paging and page replacement using LRU. [10 marks]
2. Explain tokenization and why whitespace tokenization fails on punctuation. [5 marks]
"""


def _subjects_in(record):
    return set(record["settings"]["subjects_in_corpus"])


def test_nlp_only_cannot_retrieve_os_section(svc, world):
    """The OS notes are a PERFECT lexical match for Q1. NLP-only mode must still never return them."""
    trap = svc.library.ingest(world["nlp"].id, "trap.txt", TRAP_PAPER.encode(), doc_kind="question_paper").document
    out = svc.runner.run(world["nlp"].id, [trap.id], save=False)
    os_section_ids = {s["id"] for d in world["os_docs"] for s in svc.library.sections_of(d.id)}
    assert os_section_ids, "fixture sanity: OS has indexed sections"
    returned = {m["section_id"] for q in out.record["questions"] for m in q["matches"]}
    assert returned and not (returned & os_section_ids)
    assert {s.metadata["subject_id"] for s in out.corpus.sections} == {world["nlp"].id}
    # The paging question has no NLP match -> must NOT be reported as a strong match
    paging_q = next(q for q in out.record["questions"] if "paging" in q["text"])
    assert paging_q["best_tier"] in ("Weak", "Unmatched")


def test_cross_subject_mode_would_have_found_it(svc, world):
    """Control experiment: the leakage the filter prevents really exists when explicitly enabled."""
    trap = svc.library.ingest(world["nlp"].id, "trap.txt", TRAP_PAPER.encode(), doc_kind="question_paper").document
    out = svc.runner.run(world["nlp"].id, [trap.id], allow_cross_subject=True,
                         extra_subject_ids=[world["os"].id], save=False)
    paging_q = next(q for q in out.record["questions"] if "paging" in q["text"])
    top_doc = paging_q["matches"][0]["document_id"]
    assert top_doc in {d.id for d in world["os_docs"]}
    assert paging_q["best_tier"] == "Strong"
    assert out.record["settings"]["cross_subject"] is True
    assert _subjects_in(out.record) == {world["nlp"].id, world["os"].id}


def test_uploading_dbms_document_does_not_alter_nlp_results(svc, world):
    before = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], save=False)
    dbms = svc.subjects.create("Database Management Systems", topics=DBMS["topics"])
    svc.library.ingest(dbms.id, "notes.txt", DBMS["files"]["notes.txt"].encode(), doc_kind="lecture_notes")
    after = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], save=False)
    assert before.corpus.fingerprint == after.corpus.fingerprint
    assert before.record["matrix"] == after.record["matrix"]
    assert [q["matches"] for q in before.record["questions"]] == [q["matches"] for q in after.record["questions"]]
    assert after.cache_hit is True  # unrelated upload must not invalidate NLP's cache


def test_reclassifying_document_invalidates_analysis(svc, world):
    first = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], save=False)
    assert svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], save=False).cache_hit is True
    # Move one NLP notes file into OS: NLP corpus must change and results must be recomputed.
    extra = svc.library.ingest(world["nlp"].id, "extra.txt", b"# Extra\nstemming lemmatization morphology",
                               doc_kind="lecture_notes").document
    mid = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], save=False)
    assert mid.cache_hit is False and mid.corpus.fingerprint != first.corpus.fingerprint
    svc.library.move_to_subject(extra.id, world["os"].id)
    after = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], save=False)
    assert extra.id not in after.corpus.document_ids
    assert after.corpus.fingerprint == first.corpus.fingerprint  # back to the original NLP corpus
    # OS corpus now includes it
    os_corpus = svc.retrieval.build_note_corpus(world["os"].id)
    assert extra.id in os_corpus.document_ids
    assert all(s.metadata["subject_id"] == world["os"].id for s in os_corpus.sections)


def test_same_filename_and_content_across_subjects_no_id_collision(svc):
    a = svc.subjects.create("Subject A"); b = svc.subjects.create("Subject B")
    text = b"# Intro\nidentical content in both subjects for collision testing"
    da = svc.library.ingest(a.id, "notes.txt", text, doc_kind="lecture_notes")
    db_ = svc.library.ingest(b.id, "notes.txt", text, doc_kind="lecture_notes")   # same bytes, other subject: allowed
    assert da.document.id != db_.document.id
    assert db_.other_subject_duplicates, "cross-subject duplicate should be reported, not blocked"
    ids_a = {s["id"] for s in svc.library.sections_of(da.document.id)}
    ids_b = {s["id"] for s in svc.library.sections_of(db_.document.id)}
    assert ids_a and ids_b and not (ids_a & ids_b)
    assert {s.metadata["subject_id"] for s in svc.retrieval.build_note_corpus(a.id).sections} == {a.id}


def test_cache_keys_scoped_by_subject_and_corpus_version(svc, world):
    runner = svc.runner
    c_nlp = svc.retrieval.build_note_corpus(world["nlp"].id)
    c_os = svc.retrieval.build_note_corpus(world["os"].id)
    p_nlp = svc.retrieval.build_paper_set(world["nlp"].id, [world["nlp_p1"].id])
    s = AnalysisSettings()
    k1 = runner.cache.key(c_nlp, p_nlp, s)
    assert k1 != runner.cache.key(c_os, p_nlp, s)                       # different subject
    assert k1 != runner.cache.key(c_nlp, p_nlp, AnalysisSettings(vectorizer="bow"))  # different settings
    svc.library.ingest(world["nlp"].id, "more.txt", b"# More\nlemmatization", doc_kind="lecture_notes")
    c_nlp2 = svc.retrieval.build_note_corpus(world["nlp"].id)
    assert k1 != runner.cache.key(c_nlp2, p_nlp, s)                     # new corpus version


@pytest.mark.parametrize("bad", [None, "", "sub_doesnotexist"])
def test_missing_or_invalid_subject_raises_not_fallback(svc, world, bad):
    with pytest.raises(InvalidSubjectError):
        svc.retrieval.build_note_corpus(bad)
    with pytest.raises(InvalidSubjectError):
        svc.runner.run(bad, [world["nlp_p1"].id], save=False)


def test_deleted_subject_does_not_fall_back_to_all_documents(svc, world):
    svc.subjects.delete(world["os"].id, "Operating Systems")
    with pytest.raises(InvalidSubjectError):
        svc.retrieval.build_note_corpus(world["os"].id)


def test_explicit_other_subject_document_is_rejected(svc, world):
    with pytest.raises(CrossSubjectError):
        svc.retrieval.build_note_corpus(world["nlp"].id, [world["os_docs"][0].id])
    with pytest.raises(CrossSubjectError):
        svc.runner.run(world["nlp"].id, [world["os_p1"].id], save=False)   # paper from another subject


def test_cross_subject_requires_explicit_subject_choice(svc, world):
    with pytest.raises(InvalidSubjectError):
        svc.retrieval.build_note_corpus(world["nlp"].id, allow_cross_subject=True)   # no extras named
    c = svc.retrieval.build_note_corpus(world["nlp"].id, allow_cross_subject=True, extra_subject_ids=[world["os"].id])
    assert c.cross_subject and set(c.subject_ids) == {world["nlp"].id, world["os"].id}


def test_question_paper_cannot_be_used_as_lecture_material(svc, world):
    with pytest.raises(DocumentValidationError):
        svc.retrieval.build_note_corpus(world["nlp"].id, [world["nlp_p1"].id])


def test_default_corpus_excludes_question_papers(svc, world):
    c = svc.retrieval.build_note_corpus(world["nlp"].id)
    assert not ({world["nlp_p1"].id, world["nlp_p2"].id} & set(c.document_ids))
