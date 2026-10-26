"""
Unit tests for CSV and Markdown reporting exports.
"""

import unittest
from services.export_service import ExportService
from services.analysis_service import CompleteAnalysisResult
from services.coverage_service import SectionCoverageStat, AggregateCoverageMetrics
from nlp.question_segmentation import QuestionItem
from nlp.document_segmentation import NoteSection
from nlp.similarity import QuestionMatchResult, MatchItem
from nlp.vectorization import VectorizerEngine
import numpy as np


class TestExports(unittest.TestCase):

    def setUp(self):
        self.export_svc = ExportService()
        self.questions = [QuestionItem(question_id="Q1", raw_text="What is BoW?", clean_text="What is BoW?")]
        self.sections = [NoteSection(section_id="S1", doc_name="doc.txt", page_or_slide="P1", heading="BoW Model", text="Bag of words")]
        q1_res = QuestionMatchResult(
            question_id="Q1",
            question_text="What is BoW?",
            clean_text="bow",
            marks="5M",
            best_score=0.72,
            best_tier="Strong",
            top_matches=[
                MatchItem(rank=1, section_id="S1", doc_name="doc.txt", page_or_slide="P1", heading="BoW Model", similarity_score=0.72, match_tier="Strong", section_snippet="Bag of words", matching_terms=[{"term": "bow", "contribution": 0.5}])
            ]
        )
        self.analysis = CompleteAnalysisResult(
            questions=self.questions,
            note_sections=self.sections,
            question_results=[q1_res],
            similarity_matrix=np.array([[0.72]]),
            vectorizer=VectorizerEngine(),
            strong_threshold=0.35,
            moderate_threshold=0.18,
            weak_threshold=0.08
        )
        self.section_stats = [
            SectionCoverageStat(
                section_id="S1", doc_name="doc.txt", page_or_slide="P1", heading="BoW Model",
                max_similarity=0.72, top1_match_count=1, top3_match_count=1,
                coverage_status="Covered", matched_questions=["Q1"]
            )
        ]
        self.metrics = AggregateCoverageMetrics(
            total_questions=1, total_sections=1, total_documents=1,
            strong_count=1, strong_pct=100.0, moderate_count=0, moderate_pct=0.0,
            weak_count=0, weak_pct=0.0, unmatched_count=0, unmatched_pct=0.0,
            overall_coverage_rate=100.0, mean_similarity=0.72, median_similarity=0.72,
            max_similarity=0.72, sections_covered_count=1, sections_uncovered_count=0,
            section_coverage_pct=100.0
        )

    def test_matches_csv_export(self):
        csv_str = self.export_svc.export_matches_csv(self.analysis)
        self.assertIn("Question ID", csv_str)
        self.assertIn("Q1", csv_str)
        self.assertIn("0.72", csv_str)

    def test_coverage_csv_export(self):
        csv_str = self.export_svc.export_coverage_csv(self.section_stats)
        self.assertIn("Section ID", csv_str)
        self.assertIn("S1", csv_str)
        self.assertIn("Covered", csv_str)

    def test_markdown_report_generation(self):
        md_report = self.export_svc.generate_markdown_report(self.analysis, self.metrics, self.section_stats)
        self.assertIn("# SyllabusLens", md_report)
        self.assertIn("100.0%", md_report)


if __name__ == "__main__":
    unittest.main()
