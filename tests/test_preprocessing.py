"""
Unit tests for text preprocessing, tokenization, stopword removal, and stemming.
"""

import unittest
from nlp.preprocessing import TextPreprocessor
from nlp.stemmer import PorterStemmer, stem_word


class TestTextPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor(use_stemming=False)
        self.stemming_preprocessor = TextPreprocessor(use_stemming=True)

    def test_unicode_and_case_normalization(self):
        raw = "What is TF-IDF?   Explain with\tExample!"
        res = self.preprocessor.process(raw)
        self.assertIn("tf-idf", res.normalized_text)
        self.assertNotIn("\t", res.normalized_text)

    def test_tokenization_and_min_length(self):
        text = "A quick brown fox jumped over 42 lazy dogs."
        res = self.preprocessor.process(text)
        self.assertIn("quick", res.tokens)
        self.assertIn("brown", res.tokens)
        # Single char 'a' should be excluded by min_token_len=2
        self.assertNotIn("a", res.tokens)

    def test_stop_word_removal(self):
        text = "Explain the fundamental principles of bag of words briefly"
        res = self.preprocessor.process(text)
        # 'the', 'of', 'explain', 'briefly' are stopwords
        self.assertNotIn("the", res.tokens_no_stops)
        self.assertNotIn("of", res.tokens_no_stops)
        self.assertNotIn("explain", res.tokens_no_stops)
        self.assertIn("fundamental", res.tokens_no_stops)
        self.assertIn("principles", res.tokens_no_stops)
        self.assertIn("words", res.tokens_no_stops)

    def test_porter_stemmer(self):
        stemmer = PorterStemmer()
        self.assertEqual(stemmer.stem("connecting"), "connect")
        self.assertEqual(stemmer.stem("connection"), "connect")
        self.assertEqual(stemmer.stem("connections"), "connect")
        self.assertEqual(stemmer.stem("relational"), "relat")

    def test_ngrams_generation(self):
        tokens = ["vector", "space", "model"]
        bigrams = self.preprocessor.generate_ngrams(tokens, 2)
        self.assertEqual(bigrams, ["vector space", "space model"])


if __name__ == "__main__":
    unittest.main()
