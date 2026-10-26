"""Subject suggestion behaviour (synthetic fixtures)."""
from tests.fixtures import DBMS, NLP, OS


def _setup(svc):
    n = svc.subjects.create("Natural Language Processing", description=NLP["description"], topics=NLP["topics"])
    o = svc.subjects.create("Operating Systems", description=OS["description"], topics=OS["topics"])
    d = svc.subjects.create("Database Management Systems", description=DBMS["description"], topics=DBMS["topics"])
    svc.library.ingest(n.id, "notes.txt", NLP["files"]["notes.txt"].encode(), doc_kind="lecture_notes")
    svc.library.ingest(o.id, "notes.txt", OS["files"]["notes.txt"].encode(), doc_kind="lecture_notes")
    return n, o, d


def test_suggests_correct_subject_with_evidence(svc):
    n, o, d = _setup(svc)
    r = svc.classifier.suggest_for_upload(b"# Lecture 5\nDeadlock avoidance, the banker algorithm and circular wait between processes.", "lecture5.txt")
    assert r.status == "suggested" and r.top.subject_id == o.id and r.top.evidence_terms
    r2 = svc.classifier.suggest_for_upload(b"# Week 2\nSubword tokenization, byte pair encoding and TF-IDF weighting of terms.", "week2.txt")
    assert r2.top.subject_id == n.id


def test_content_beats_misleading_filename_and_conflict_is_flagged(svc):
    n, o, d = _setup(svc)
    r = svc.classifier.suggest_for_upload(b"# Notes\nDeadlock avoidance, banker algorithm, circular wait and mutual exclusion.", "natural_language_processing_notes.txt")
    assert r.top.subject_id == o.id          # content wins
    assert r.filename_hint == n.id and r.filename_conflict


def test_insufficient_evidence_is_reported_not_guessed(svc):
    _setup(svc)
    r = svc.classifier.suggest_for_upload(b"# Recipes\nflour sugar butter oven baking cake frosting", "cake.txt")
    assert r.status == "insufficient"


def test_no_subjects_and_unextractable_text(svc):
    assert svc.classifier.suggest_for_text("anything at all", "x.txt").status == "no_subjects"
    _setup(svc)
    assert svc.classifier.suggest_for_text("", "x.txt").status == "insufficient"


def test_ambiguous_when_two_subjects_tie(svc):
    a = svc.subjects.create("Alpha", description="graph algorithms shortest path trees")
    b = svc.subjects.create("Beta", description="graph algorithms shortest path trees")
    r = svc.classifier.suggest_for_text("graph algorithms shortest path trees", "g.txt")
    assert r.status == "ambiguous" and len(r.suggestions) >= 2


def test_suggestion_never_changes_stored_assignment(svc):
    n, o, d = _setup(svc)
    doc = svc.library.ingest(n.id, "ds.txt", b"# Deadlocks\nbanker algorithm circular wait", doc_kind="lecture_notes").document
    r = svc.classifier.suggest_for_document(doc.id)
    assert r.top.subject_id == o.id
    assert svc.library.get(doc.id).subject_id == n.id     # advisory only; user must confirm a move
