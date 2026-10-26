"""
Unit tests for Information Retrieval evaluation metrics: P@k, R@k, MRR, MAP, and Confusion Matrix.
"""

import unittest
from services.evaluation_service import EvaluationService


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.eval_svc = EvaluationService()

    def test_precision_recall_mrr(self):
        # Retrieved rankings
        retrieved = {
            "Q1": ["S1", "S2", "S3"],
            "Q2": ["S3", "S1", "S2"]
        }
        # Ground truth relevant sections
        ground_truth = {
            "Q1": ["S1"],        # S1 is rank 1 -> P@1 = 1.0, RR = 1.0
            "Q2": ["S1", "S2"]   # S1 is rank 2, S2 is rank 3 -> P@1 = 0.0, RR = 0.5, P@3 = 2/3
        }

        report = self.eval_svc.evaluate_retrieval("Test Method", retrieved, ground_truth)
        self.assertEqual(report.num_queries, 2)
        self.assertEqual(report.precision_at_1, 0.5)
        self.assertAlmostEqual(report.mrr, 0.75, places=3)
        self.assertGreater(report.recall_at_3, 0.8)

    def test_binary_coverage_confusion_matrix(self):
        predicted_scores = {
            "Q1": 0.45,   # Predicted Covered
            "Q2": 0.30,   # Predicted Covered
            "Q3": 0.02    # Predicted Uncovered
        }
        ground_truth_covered = {
            "Q1": True,   # Actual Covered -> TP
            "Q2": False,  # Actual Uncovered -> FP
            "Q3": False   # Actual Uncovered -> TN
        }
        cm = self.eval_svc.evaluate_coverage_classification(
            predicted_scores, ground_truth_covered, threshold=0.15
        )
        self.assertEqual(cm.true_positives, 1)
        self.assertEqual(cm.false_positives, 1)
        self.assertEqual(cm.true_negatives, 1)
        self.assertEqual(cm.false_negatives, 0)
        self.assertAlmostEqual(cm.accuracy, 2/3, places=2)


if __name__ == "__main__":
    unittest.main()
