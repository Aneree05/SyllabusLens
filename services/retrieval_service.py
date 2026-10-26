"""
The retrieval boundary.

    Selected subject -> eligible documents -> eligible sections -> (vectorize, rank elsewhere)

Everything downstream (vectorizer, similarity) receives ONLY the sections returned here.
The subject filter is applied BEFORE ranking, in SQL and again as an in-memory invariant
check. There is deliberately no code path that defaults to "all documents": a missing or
unknown subject raises InvalidSubjectError, and documents from other subjects raise
CrossSubjectError unless cross-subject mode is enabled explicitly with explicit subjects.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import config
from nlp.document_segmentation import NoteSection
from nlp.question_segmentation import QuestionItem
from services.errors import (
    CrossSubjectError, DocumentValidationError, InvalidSubjectError, IsolationViolation, NotFoundError,
)
from services.library_service import LibraryService
from services.subject_service import SubjectService
from storage import Database


@dataclass(frozen=True)
class Corpus:
    subject_ids: Tuple[str, ...]
    cross_subject: bool
    document_ids: Tuple[str, ...]
    sections: Tuple[NoteSection, ...]
    fingerprint: str
    documents: Tuple[Dict[str, Any], ...]   # frozen per-document facts used for history


@dataclass(frozen=True)
class PaperSet:
    subject_id: str
    papers: Tuple[Dict[str, Any], ...]      # frozen facts per paper
    questions: Tuple[QuestionItem, ...]     # IDs are scoped "<doc_id>:<qid>"
    fingerprint: str


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:24]


class RetrievalService:
    def __init__(self, db: Database, subjects: SubjectService, library: LibraryService):
        self.db, self.subjects, self.library = db, subjects, library

    # ---------------------------------------------------------------- subjects
    def _require_subject(self, subject_id: Optional[str]) -> None:
        if not subject_id or not isinstance(subject_id, str):
            raise InvalidSubjectError("No subject is selected. Select a subject before running an analysis.")
        if not self.subjects.exists(subject_id):
            raise InvalidSubjectError(
                f"Subject '{subject_id}' no longer exists. Select an existing subject; "
                "no documents were searched.")

    def _allowed_subjects(self, subject_id: str, allow_cross_subject: bool,
                          extra_subject_ids: Sequence[str]) -> Tuple[str, ...]:
        self._require_subject(subject_id)
        if not allow_cross_subject:
            return (subject_id,)
        extras = [s for s in dict.fromkeys(extra_subject_ids) if s != subject_id]
        if not extras:
            raise InvalidSubjectError(
                "Cross-subject mode needs you to choose which additional subjects to include.")
        for s in extras:
            self._require_subject(s)
        return (subject_id, *extras)

    # ------------------------------------------------------------------- notes
    def build_note_corpus(self, subject_id: Optional[str], document_ids: Optional[Iterable[str]] = None, *,
                          allow_cross_subject: bool = False,
                          extra_subject_ids: Sequence[str] = ()) -> Corpus:
        allowed = self._allowed_subjects(subject_id, allow_cross_subject, extra_subject_ids)  # type: ignore[arg-type]
        allowed_set = set(allowed)

        if document_ids is None:
            docs = []
            for sid in allowed:
                docs += [d for d in self.library.list_documents(sid, status="processed", limit=1000)
                         if config.DOCUMENT_KINDS[d.doc_kind][2]]
        else:
            docs = []
            for did in dict.fromkeys(document_ids):
                d = self.library.get(did)
                if d.subject_id not in allowed_set:
                    raise CrossSubjectError(
                        f"'{d.filename}' belongs to a different subject than the one selected. "
                        "Remove it, or enable cross-subject comparison explicitly.")
                if d.is_question_paper:
                    raise DocumentValidationError(f"'{d.filename}' is a question paper, not lecture material.")
                if d.status != "processed":
                    raise DocumentValidationError(
                        f"'{d.filename}' could not be processed ({d.status}); fix or remove it first.")
                docs.append(d)
        if not docs:
            raise DocumentValidationError(
                "The selected subject has no processed lecture material. Upload lecture notes or slides first.")

        ids = tuple(d.id for d in docs)
        qmarks = ",".join("?" * len(ids))
        smarks = ",".join("?" * len(allowed))
        with self.db.connect() as conn:
            rows = conn.execute(
                f"SELECT s.*, d.filename, d.title, sub.name AS subject_name FROM sections s "
                f"JOIN documents d ON d.id=s.document_id JOIN subjects sub ON sub.id=s.subject_id "
                f"WHERE s.subject_id IN ({smarks}) AND s.document_id IN ({qmarks}) "
                f"ORDER BY s.document_id, s.ordinal", (*allowed, *ids)).fetchall()

        sections: List[NoteSection] = []
        for r in rows:
            if r["subject_id"] not in allowed_set:      # invariant: should be impossible
                raise IsolationViolation(f"Section {r['id']} from a disallowed subject reached the corpus.")
            sections.append(NoteSection(
                section_id=r["id"], doc_name=r["filename"], page_or_slide=r["page_or_slide"],
                heading=r["heading"], text=r["text"], word_count=r["word_count"],
                metadata={"document_id": r["document_id"], "subject_id": r["subject_id"],
                          "subject_name": r["subject_name"], "document_title": r["title"]}))
        if not sections:
            raise DocumentValidationError("The selected documents contain no indexed sections.")

        facts = tuple(dict(document_id=d.id, subject_id=d.subject_id, filename=d.filename, title=d.title,
                           kind=d.doc_kind, file_hash=d.file_hash, sections=sum(
                               1 for s in sections if s.metadata["document_id"] == d.id)) for d in docs)
        fp = _digest({"cross": bool(allow_cross_subject), "subjects": sorted(allowed),
                      "docs": sorted((f["document_id"], f["subject_id"], f["file_hash"], f["kind"]) for f in facts)})
        return Corpus(allowed, bool(allow_cross_subject), ids, tuple(sections), fp, facts)

    # ------------------------------------------------------------------ papers
    def build_paper_set(self, subject_id: Optional[str], paper_ids: Iterable[str]) -> PaperSet:
        self._require_subject(subject_id)
        ids = list(dict.fromkeys(paper_ids or []))
        if not ids:
            raise DocumentValidationError("Select at least one question paper.")
        papers, questions = [], []
        for did in ids:
            d = self.library.get(did)
            if d.subject_id != subject_id:
                raise CrossSubjectError(
                    f"Question paper '{d.filename}' belongs to a different subject than the one selected.")
            if not d.is_question_paper:
                raise DocumentValidationError(f"'{d.filename}' is not classified as a question paper.")
            if d.status != "processed":
                raise DocumentValidationError(f"'{d.filename}' could not be processed ({d.status}).")
            qs = self.library.questions_of(did)
            if not qs:
                raise DocumentValidationError(f"No questions could be extracted from '{d.filename}'. "
                                              "Add or correct questions manually.")
            label = d.title if len(ids) > 1 else d.title
            for q in qs:
                meta = dict(q.metadata or {})
                meta.update(paper_doc_id=d.id, paper_label=label, display_id=q.question_id)
                questions.append(QuestionItem(
                    question_id=f"{d.id}:{q.question_id}", raw_text=q.raw_text, clean_text=q.clean_text,
                    marks=q.marks, parent_id=q.parent_id, sub_label=q.sub_label, metadata=meta))
            papers.append(dict(document_id=d.id, filename=d.filename, title=d.title, file_hash=d.file_hash,
                               exam_session=d.exam_session, academic_year=d.academic_year,
                               corrected=d.has_corrections, question_count=len(qs)))
        fp = _digest([(p["document_id"], p["file_hash"], [q.clean_text for q in questions
                      if q.metadata["paper_doc_id"] == p["document_id"]]) for p in papers])
        return PaperSet(subject_id, tuple(papers), tuple(questions), fp)  # type: ignore[arg-type]
