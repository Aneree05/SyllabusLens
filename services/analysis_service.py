"""
Core Analysis Service for SyllabusLens.
Orchestrates preprocessing, vectorization, similarity computation,
keyword explanation, and structured retrieval outputs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from nlp.preprocessing import TextPreprocessor, default_preprocessor
from nlp.question_segmentation import QuestionItem
from nlp.document_segmentation import NoteSection
from nlp.vectorization import VectorizerEngine
from nlp.similarity import SimilarityEngine, QuestionMatchResult, MatchItem
from nlp.keyword_explanations import KeywordExplainer
from config import (
    DEFAULT_STRONG_THRESHOLD,
    DEFAULT_MODERATE_THRESHOLD,
    DEFAULT_WEAK_THRESHOLD,
    DEFAULT_MAX_FEATURES,
    DEFAULT_NGRAM_RANGE
)


@dataclass
class CompleteAnalysisResult:
    """Holds complete end-to-end analysis data for UI rendering and exports."""
    questions: List[QuestionItem]
    note_sections: List[NoteSection]
    question_results: List[QuestionMatchResult]
    similarity_matrix: np.ndarray
    vectorizer: VectorizerEngine
    strong_threshold: float
    moderate_threshold: float
    weak_threshold: float
    method_name: str = "TF-IDF (Cosine Similarity)"
    settings: Dict[str, Any] = field(default_factory=dict)

    def get_question_result(self, qid: str) -> Optional[QuestionMatchResult]:
        for qr in self.question_results:
            if qr.question_id == qid:
                return qr
        return None

    def get_section(self, section_id: str) -> Optional[NoteSection]:
        for s in self.note_sections:
            if s.section_id == section_id:
                return s
        return None


class AnalysisService:
    """
    Coordinates end-to-end question paper coverage analysis against lecture notes.
    """

    def __init__(
        self,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
        moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
        weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
        ngram_range: tuple = DEFAULT_NGRAM_RANGE,
        use_stemming: bool = False
    ):
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.weak_threshold = weak_threshold
        self.ngram_range = ngram_range
        self.use_stemming = use_stemming

        self.preprocessor = TextPreprocessor(use_stemming=use_stemming)
        self.similarity_engine = SimilarityEngine(
            strong_threshold=strong_threshold,
            moderate_threshold=moderate_threshold,
            weak_threshold=weak_threshold
        )

    def run_analysis(
        self,
        questions: List[QuestionItem],
        note_sections: List[NoteSection],
        vectorizer_type: str = "tfidf",  # "tfidf" or "bow"
        top_k: int = 3
    ) -> CompleteAnalysisResult:
        """
        Executes complete coverage analysis pipeline:
        1. Preprocess texts.
        2. Fit vectorizer on joint corpus.
        3. Transform questions and sections.
        4. Compute pairwise cosine similarity.
        5. Rank matches and extract keyword explanations.
        """
        if not questions:
            raise ValueError("No questions provided for analysis.")
        if not note_sections:
            raise ValueError("No lecture note sections provided for analysis.")

        # 1. Clean texts
        cleaned_questions = [
            self.preprocessor.process(q.clean_text).cleaned_text for q in questions
        ]
        cleaned_sections = [
            self.preprocessor.process(s.text).cleaned_text for s in note_sections
        ]

        # 2. The vocabulary and IDF are learned from the note corpus ONLY.
        #    Questions are queries: they must not influence section weights, so the
        #    same corpus always yields the same section vectors regardless of paper.
        joint_corpus = cleaned_sections

        # 3. Fit vectorizer
        vectorizer = VectorizerEngine(
            ngram_range=self.ngram_range,
            max_features=DEFAULT_MAX_FEATURES
        )
        vectorizer.fit(joint_corpus)

        # 4. Transform matrices
        if vectorizer_type == "bow":
            q_matrix = vectorizer.transform_bow(cleaned_questions)
            s_matrix = vectorizer.transform_bow(cleaned_sections)
            method_label = "Bag-of-Words (Count Vectors)"
        else:
            q_matrix = vectorizer.transform_tfidf(cleaned_questions)
            s_matrix = vectorizer.transform_tfidf(cleaned_sections)
            method_label = "TF-IDF (Normalized Feature Space)"

        # 5. Compute pairwise cosine similarity
        sim_matrix = self.similarity_engine.compute_similarity_matrix(q_matrix, s_matrix)

        ids = [sec.section_id for sec in note_sections]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate section IDs in corpus; section IDs must be unique.")
        section_index = {sid: i for i, sid in enumerate(ids)}

        # 6. Keyword Explainer
        explainer = KeywordExplainer(vectorizer.feature_names)

        # 7. Build question results
        question_results: List[QuestionMatchResult] = []

        for q_idx, q in enumerate(questions):
            scores_row = sim_matrix[q_idx]
            top_matches = self.similarity_engine.rank_top_matches(
                q_idx=q_idx,
                scores_row=scores_row,
                sections=note_sections,
                top_k=top_k
            )

            # Enrich top matches with keyword explanations
            q_vec = q_matrix[q_idx:q_idx+1]
            for m in top_matches:
                sec_idx = section_index.get(m.section_id)

                if sec_idx is not None and m.similarity_score > 0.001:
                    doc_vec = s_matrix[sec_idx:sec_idx+1]
                    terms, expl = explainer.explain_match(q_vec, doc_vec, top_n=5)
                    m.matching_terms = terms
                    m.explanation = expl
                else:
                    m.explanation = "No lexical overlap detected in vocabulary space."

            best_score = top_matches[0].similarity_score if top_matches else 0.0
            best_tier = self.similarity_engine.classify_tier(best_score)

            question_results.append(QuestionMatchResult(
                question_id=q.question_id,
                question_text=q.clean_text,
                clean_text=cleaned_questions[q_idx],
                marks=q.marks,
                best_score=best_score,
                best_tier=best_tier,
                top_matches=top_matches
            ))

        return CompleteAnalysisResult(
            questions=questions,
            note_sections=note_sections,
            question_results=question_results,
            similarity_matrix=sim_matrix,
            vectorizer=vectorizer,
            strong_threshold=self.strong_threshold,
            moderate_threshold=self.moderate_threshold,
            weak_threshold=self.weak_threshold,
            method_name=method_label,
            settings={
                "vectorizer": vectorizer_type,
                "ngram_range": list(self.ngram_range),
                "use_stemming": self.use_stemming,
                "strong_threshold": self.strong_threshold,
                "moderate_threshold": self.moderate_threshold,
                "weak_threshold": self.weak_threshold,
                "top_k": top_k,
                "max_features": DEFAULT_MAX_FEATURES,
                "vocabulary_fit_on": "note sections only",
            },
        )
