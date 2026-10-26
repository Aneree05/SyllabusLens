"""
Regression tests for Topic-Wise Revision Notes and Multi-Document Content Synthesis.
Verifies source traceability, structured extraction (definitions, formulas, pros/cons),
missing material detection, Markdown export, and strict subject isolation.
"""
from __future__ import annotations

import pytest
from services.topic_notes_service import TopicNotesService, TopicRevisionNote
from services import record_export as rx


def _revision_subject(svc):
    subject = svc.subjects.create("Vector Space Models", topics=["Vector Space Models"])
    text = (
        "# Vector Space Models\n"
        "A vector space model is defined as a representation of documents using weighted term vectors.\n"
        "IDF(t) = ln((1 + |D|) / (1 + DF(t))) + 1.\n"
        "A limitation of the basic Bag of Words model is that it loses word order and produces sparse vectors.\n"
        "For example, cosine similarity compares two document vectors by their angle.\n"
    )
    svc.library.ingest(subject.id, "vector_space_notes.txt", text.encode(), doc_kind="lecture_notes")
    return subject


def test_topic_notes_extracts_definitions_and_formulas_with_sources(svc):
    notes_svc = svc.topic_notes
    nlp_sub = _revision_subject(svc)
    notes = notes_svc.build_topic_notes(nlp_sub.id)

    assert len(notes) > 0
    # Check that topics exist (e.g., Vector Space Models, Text Normalization)
    topic_names = [n.topic for n in notes]
    assert any("Vector Space" in t or "Text Normalization" in t for t in topic_names)

    # Check a covered topic
    vsm_note = next(n for n in notes if "Vector Space" in n.topic)
    assert vsm_note.status in ("covered", "partially_covered")

    # Verify source traceability: all extracted items carry doc_name, locator, heading
    for d, cit in vsm_note.definitions:
        assert cit.doc_name
        assert cit.locator
        assert cit.heading
        assert cit.section_id

    # Formulas must be detected from the notes (e.g., IDF or Sim formula)
    assert len(vsm_note.formulas) > 0
    assert any("=" in f[0] or "IDF" in f[0] or "Sim" in f[0] for f in vsm_note.formulas)


def test_topic_notes_detects_advantages_and_limitations(svc):
    notes_svc = svc.topic_notes
    nlp_sub = _revision_subject(svc)
    notes = notes_svc.build_topic_notes(nlp_sub.id)
    vsm_note = next(n for n in notes if "Vector Space" in n.topic)

    # In sample_lecture_notes_nlp, BoW has limitations: "sparsity, high dimensionality, failure to model synonymy"
    assert len(vsm_note.pros_cons_limitations) > 0
    assert any("limitation" in pcl[0].lower() or "sparsity" in pcl[0].lower() or "advantage" in pcl[0].lower()
               for pcl in vsm_note.pros_cons_limitations)


def test_topic_notes_missing_material_alert_not_hallucinated(svc):
    notes_svc = svc.topic_notes
    # Create a subject with an unmentioned topic
    s = svc.subjects.create("Robotics", topics=["Inverse Kinematics", "PID Controller"])
    # Ingest notes that only talk about general sensors
    svc.library.ingest(s.id, "sensors.txt", b"# Sensors\nUltrasonic distance sensors measure sound waves.", doc_kind="lecture_notes")

    notes = notes_svc.build_topic_notes(s.id)
    assert len(notes) == 2
    for n in notes:
        assert n.status in ("uncovered", "partially_covered")
        assert len(n.missing_material) > 0
        # Must clearly state what is missing rather than inventing facts
        assert any("missing" in m.lower() for m in n.missing_material)


def test_topic_notes_strict_subject_isolation(svc, world):
    notes_svc = svc.topic_notes
    nlp_sub = world["nlp"]
    os_sub = world["os"]

    nlp_notes = notes_svc.build_topic_notes(nlp_sub.id)
    os_notes = notes_svc.build_topic_notes(os_sub.id)

    # Invariant: No OS section citations inside NLP topic notes
    nlp_sources = [cit.doc_name for n in nlp_notes for cit in n.sources]
    for src in nlp_sources:
        assert "os_" not in src.lower()

    # Invariant: No NLP section citations inside OS topic notes
    os_sources = [cit.doc_name for n in os_notes for cit in n.sources]
    for src in os_sources:
        assert "nlp_" not in src.lower()


def test_topic_notes_markdown_export(svc):
    notes_svc = svc.topic_notes
    nlp_sub = _revision_subject(svc)
    notes = notes_svc.build_topic_notes(nlp_sub.id)

    md = notes_svc.export_notes_markdown(notes, nlp_sub.name)
    assert "# Vector Space Models - Topic-Wise Revision Notes" in md
    assert "Key Definitions & Foundations" in md
    assert "Formulas & Governing Equations" in md
    assert "Source: [" in md

    # Export via analysis record
    paper = svc.library.ingest(
        nlp_sub.id, "vector_space_paper.txt",
        b"Exam\n1. Explain vector space models and IDF. [5 marks]\n",
        doc_kind="question_paper",
    ).document
    out = svc.runner.run(nlp_sub.id, [paper.id], save=False)
    rec_md = rx.topic_notes_markdown(out.record)
    assert "Vector Space Models" in rec_md
    assert "##" in rec_md
