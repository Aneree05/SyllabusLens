"""Builds subject profiles from stored data and produces subject suggestions."""
from __future__ import annotations

from typing import List, Optional

import config
from nlp.subject_classifier import SubjectProfile, SuggestionResult, suggest_subject
from services.library_service import LibraryService
from services.subject_service import SubjectService

_PER_DOC_CHARS = 3000
_PER_SUBJECT_CHARS = 20000


class ClassificationService:
    def __init__(self, subjects: SubjectService, library: LibraryService):
        self.subjects, self.library = subjects, library

    def build_profiles(self, exclude_doc_id: Optional[str] = None) -> List[SubjectProfile]:
        profiles = []
        for s in self.subjects.list():
            parts = [s.name, s.name, s.description, s.course_code, " ".join(s.topics), " ".join(s.topics)]
            budget = _PER_SUBJECT_CHARS
            for d in self.library.list_documents(s.id, status="processed", limit=200):
                if d.id == exclude_doc_id or budget <= 0:
                    continue
                excerpt = self.library.get_text(d.id)[:_PER_DOC_CHARS]
                parts.append(excerpt)
                budget -= len(excerpt)
            profiles.append(SubjectProfile(s.id, s.name, " ".join(p for p in parts if p)))
        return profiles

    def suggest_for_text(self, text: str, filename: str, exclude_doc_id: Optional[str] = None) -> SuggestionResult:
        return suggest_subject(text, filename, self.build_profiles(exclude_doc_id),
                               config.SUGGEST_MIN_SCORE, config.SUGGEST_AMBIGUITY_MARGIN)

    def suggest_for_upload(self, data: bytes, filename: str) -> SuggestionResult:
        """Extract text from a not-yet-stored upload and suggest a subject (nothing is saved)."""
        display, _ = self.library._validate_upload(filename, data)
        extracted, status, _ = self.library._extract(data, display)
        if status != "processed":
            from nlp.subject_classifier import SuggestionResult as SR
            return SR("insufficient", "Text could not be extracted, so no subject can be suggested. Choose manually.")
        return self.suggest_for_text(extracted.full_text, display)

    def suggest_for_document(self, doc_id: str) -> SuggestionResult:
        doc = self.library.get(doc_id)
        return self.suggest_for_text(self.library.get_text(doc_id), doc.filename, exclude_doc_id=doc_id)
