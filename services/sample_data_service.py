"""Installs the bundled DEMO material as an ordinary, clearly labelled subject."""
from __future__ import annotations

from pathlib import Path

from services.subject_service import Subject

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"
SAMPLE_SUBJECT_NAME = "Natural Language Processing (sample data)"
SAMPLE_TAG = "sample-data"
SAMPLE_TOPICS = [
    "Text normalization and tokenization", "Stemming and lemmatization", "Bag of Words", "TF-IDF weighting",
    "Cosine similarity", "Part-of-speech tagging", "Named entity recognition", "Word embeddings",
    "Recurrent neural networks and LSTM", "Transformers and self-attention",
    # Not asked in the sample paper (or not in the notes) - they exercise the gap statuses:
    "Precision and recall metrics", "Language models and n-grams", "Machine translation",
]


def install_sample_subject(svc) -> Subject:
    """Idempotent. Q7 of the sample paper (quantum computing) is deliberately OUT of this syllabus."""
    for s in svc.subjects.list():
        if s.name == SAMPLE_SUBJECT_NAME:
            return s
    subject = svc.subjects.create(
        SAMPLE_SUBJECT_NAME, course_code="CS801 (demo)",
        description="Bundled demonstration material: synthetic NLP lecture notes and a synthetic end-semester "
                    "paper. Not real course content. Delete this subject at any time.",
        topics=SAMPLE_TOPICS)
    notes = (SAMPLE_DIR / "sample_lecture_notes_nlp.txt").read_bytes()
    paper = (SAMPLE_DIR / "sample_question_paper.txt").read_bytes()
    svc.library.ingest(subject.id, "sample_lecture_notes_nlp.txt", notes, doc_kind="lecture_notes",
                       title="Sample NLP lecture notes (demo)", tags=[SAMPLE_TAG], course_code="CS801")
    svc.library.ingest(subject.id, "sample_question_paper.txt", paper, doc_kind="question_paper",
                       title="Sample NLP end-semester paper (demo)", tags=[SAMPLE_TAG], exam_session="Demo")
    return subject
