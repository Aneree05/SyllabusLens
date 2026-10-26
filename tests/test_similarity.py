"""
Unit tests for cosine similarity calculation, match ranking, and tier classification.
"""

import unittest
import numpy as np
from nlp.similarity import SimilarityEngine
from nlp.document_segmentation import NoteSection
from nlp.vectorization import VectorizerEngine


class TestSimilarity(unittest.TestCase):

    def setUp(self):
        self.sim_engine = SimilarityEngine(
            strong_threshold=0.35,
            moderate_threshold=0.18,
            weak_threshold=0.08
        )
        self.vec = VectorizerEngine()
        self.sections = [
            NoteSection(section_id="S1", doc_name="doc1.txt", page_or_slide="Page 1", heading="Topic A: TF-IDF", text="term frequency inverse document frequency retrieval"),
            NoteSection(section_id="S2", doc_name="doc1.txt", page_or_slide="Page 2", heading="Topic B: Deep Learning", text="recurrent neural networks lstm backpropagation"),
        ]
        self.vec.fit([s.text for s in self.sections] + ["tf-idf calculation"])

    def test_tier_classification(self):
        self.assertEqual(self.sim_engine.classify_tier(0.55), "Strong")
        self.assertEqual(self.sim_engine.classify_tier(0.25), "Moderate")
        self.assertEqual(self.sim_engine.classify_tier(0.12), "Weak")
        self.assertEqual(self.sim_engine.classify_tier(0.02), "Unmatched")

    def test_cosine_similarity_matrix_and_ranking(self):
        q_vec = self.vec.transform_tfidf(["explain term frequency and inverse document frequency"])
        s_vec = self.vec.transform_tfidf([s.text for s in self.sections])

        sim_mat = self.sim_engine.compute_similarity_matrix(q_vec, s_vec)
        self.assertEqual(sim_mat.shape, (1, 2))
        self.assertGreater(sim_mat[0, 0], sim_mat[0, 1])  # S1 should match much higher than S2

        top_matches = self.sim_engine.rank_top_matches(
            q_idx=0,
            scores_row=sim_mat[0],
            sections=self.sections,
            top_k=2
        )
        self.assertEqual(len(top_matches), 2)
        self.assertEqual(top_matches[0].section_id, "S1")
        self.assertEqual(top_matches[0].rank, 1)
        self.assertEqual(top_matches[0].match_tier, "Strong")


if __name__ == "__main__":
    unittest.main()
