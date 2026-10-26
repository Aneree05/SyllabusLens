"""
Unit tests for BoW and TF-IDF vectorization and matrix calculations.
"""

import unittest
from nlp.vectorization import VectorizerEngine


class TestVectorization(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "Natural language processing and text retrieval",
            "Term frequency inverse document frequency and bag of words",
            "Recurrent neural networks and attention mechanism in transformers"
        ]
        self.engine = VectorizerEngine()
        self.engine.fit(self.corpus)

    def test_vocabulary_and_feature_names(self):
        self.assertGreater(len(self.engine.feature_names), 5)
        self.assertIn("processing", self.engine.vocabulary)
        self.assertIn("frequency", self.engine.vocabulary)

    def test_bow_transform(self):
        bow_matrix = self.engine.transform_bow(["text retrieval processing"])
        self.assertEqual(bow_matrix.shape[0], 1)
        self.assertEqual(bow_matrix.shape[1], len(self.engine.feature_names))
        self.assertGreater(bow_matrix.nnz, 0)

    def test_tfidf_transform_and_l2_norm(self):
        tfidf_matrix = self.engine.transform_tfidf(["text retrieval"])
        vec = tfidf_matrix.toarray()[0]
        # Check L2 normalization (sum of squares is approx 1.0)
        norm_val = sum(x**2 for x in vec)
        self.assertAlmostEqual(norm_val, 1.0, places=4)

    def test_empty_text_safety(self):
        bow_empty = self.engine.transform_bow([""])
        self.assertEqual(bow_empty.nnz, 0)
        tfidf_empty = self.engine.transform_tfidf([""])
        self.assertEqual(tfidf_empty.nnz, 0)

    def test_sparsity_calculation(self):
        matrix = self.engine.transform_tfidf(self.corpus)
        sparsity = self.engine.get_sparsity(matrix)
        self.assertGreater(sparsity, 50.0)
        self.assertLessEqual(sparsity, 100.0)


if __name__ == "__main__":
    unittest.main()
