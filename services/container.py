"""Composition root: builds all services over one data directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from services.analysis_runner import AnalysisCache, AnalysisRunner
from services.classification_service import ClassificationService
from services.diagram_service import DiagramService
from services.history_service import HistoryService
from services.library_service import LibraryService
from services.numerical_solver import NumericalSolver
from services.question_practice_service import QuestionPracticeService
from services.retrieval_service import RetrievalService
from services.subject_service import SubjectService
from services.topic_notes_service import TopicNotesService
from nlp.question_classifier import QuestionClassifier
from storage import Database, FileStore


@dataclass
class Services:
    data_dir: Path
    db: Database
    files: FileStore
    subjects: SubjectService
    library: LibraryService
    retrieval: RetrievalService
    classifier: ClassificationService
    history: HistoryService
    runner: AnalysisRunner
    topic_notes: TopicNotesService
    question_practice: QuestionPracticeService
    question_classifier: QuestionClassifier
    numerical_solver: NumericalSolver
    diagrams: DiagramService


def build_services(data_dir: Optional[Path] = None) -> Services:
    data_dir = Path(data_dir or config.DATA_DIR)
    db = Database(data_dir / config.DB_FILENAME)
    files = FileStore(data_dir / config.UPLOAD_DIRNAME)
    subjects = SubjectService(db, files)
    library = LibraryService(db, files, subjects)
    retrieval = RetrievalService(db, subjects, library)
    history = HistoryService(db, library)
    diagrams = DiagramService()
    solver = NumericalSolver()
    q_classifier = QuestionClassifier()
    topic_notes = TopicNotesService(subjects, retrieval, diagrams)
    q_practice = QuestionPracticeService(subjects, retrieval, q_classifier, solver, diagrams)
    runner = AnalysisRunner(subjects, retrieval, history, AnalysisCache(),
                            topic_notes_svc=topic_notes, question_practice_svc=q_practice)
    return Services(data_dir, db, files, subjects, library, retrieval,
                    ClassificationService(subjects, library), history, runner,
                    topic_notes, q_practice, q_classifier, solver, diagrams)
