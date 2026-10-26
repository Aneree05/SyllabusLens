"""
Unit tests for Coverage analytics and uncovered section detection.
"""

import unittest
from services.analysis_service import CompleteAnalysisResult
from services.coverage_service import CoverageService
from nlp.question_segmentation import QuestionItem
from nlp.document_segmentation import NoteSection
from nlp.similarity import QuestionMatchResult, MatchItem
from nlp.vectorization import VectorizerEngine
import numpy as np


class TestCoverage(unittest.TestCase):

    def test_coverage_metrics_calculation(self):
        questions = [
            QuestionItem(question_id="Q1", raw_text="What is BoW?", clean_text="What is BoW?"),
            QuestionItem(question_id="Q2", raw_text="What is Quantum Physics?", clean_text="What is Quantum Physics?")
        ]
        sections = [
            NoteSection(section_id="S1", doc_name="doc.txt", page_or_slide="P1", heading="BoW Model", text="Bag of words frequency"),
            NoteSection(section_id="S2", doc_name="doc.txt", page_or_slide="P2", heading="Embeddings", text="Word2Vec GloVe")
        ]

        q1_res = QuestionMatchResult(
            question_id="Q1",
            question_text="What is BoW?",
            clean_text="bow",
            marks=None,
            best_score=0.65,
            best_tier="Strong",
            top_matches=[
                MatchItem(rank=1, section_id="S1", doc_name="doc.txt", page_or_slide="P1", heading="BoW Model", similarity_score=0.65, match_tier="Strong", section_snippet="Bag of words")
            ]
        )
        q2_res = QuestionMatchResult(
            question_id="Q2",
            question_text="What is Quantum Physics?",
            clean_text="quantum physics",
            marks=None,
            best_score=0.01,
            best_tier="Unmatched",
            top_matches=[
                MatchItem(rank=1, section_id="S2", doc_name="doc.txt", page_or_slide="P2", heading="Embeddings", similarity_score=0.01, match_tier="Unmatched", section_snippet="Word2Vec")
            ]
        )

        analysis = CompleteAnalysisResult(
            questions=questions,
            note_sections=sections,
            question_results=[q1_res, q2_res],
            similarity_matrix=np.array([[0.65, 0.05], [0.00, 0.01]]),
            vectorizer=VectorizerEngine(),
            strong_threshold=0.35,
            moderate_threshold=0.18,
            weak_threshold=0.08
        )

        cov_svc = CoverageService()
        metrics, sec_stats = cov_svc.compute_coverage(analysis)

        self.assertEqual(metrics.total_questions, 2)
        self.assertEqual(metrics.strong_count, 1)
        self.assertEqual(metrics.unmatched_count, 1)
        self.assertEqual(metrics.overall_coverage_rate, 50.0)

        # Section 2 should be uncovered in exam
        s2_stat = next(s for s in sec_stats if s.section_id == "S2")
        self.assertEqual(s2_stat.coverage_status, "Uncovered in Exam")


if __name__ == "__main__":
    unittest.main()
