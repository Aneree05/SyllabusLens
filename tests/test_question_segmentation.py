"""
Unit tests for question segmentation, subquestion detection, and marks extraction.
"""

import unittest
from nlp.question_segmentation import QuestionSegmenter


class TestQuestionSegmentation(unittest.TestCase):

    def setUp(self):
        self.segmenter = QuestionSegmenter()

    def test_numbered_questions_with_marks(self):
        text = """
        1. Define Bag of Words model. [5 Marks]
        2. Explain Term Frequency-Inverse Document Frequency. [10 Marks]
        """
        questions = self.segmenter.segment(text)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].question_id, "Q1")
        self.assertEqual(questions[0].marks, "5 Marks")
        self.assertIn("Define Bag of Words model", questions[0].clean_text)

        self.assertEqual(questions[1].question_id, "Q2")
        self.assertEqual(questions[1].marks, "10 Marks")

    def test_subquestions_structure(self):
        text = """
        1. (a) What is Cosine Similarity? [5]
        (b) Differentiate between Stemming and Lemmatization. [5]
        2. Describe RNN and LSTM. [10]
        """
        questions = self.segmenter.segment(text)
        self.assertEqual(len(questions), 3)
        self.assertEqual(questions[0].question_id, "Q1.a")
        self.assertEqual(questions[0].parent_id, "Q1")
        self.assertEqual(questions[1].question_id, "Q1.b")
        self.assertEqual(questions[1].parent_id, "Q1")
        self.assertEqual(questions[2].question_id, "Q2")

    def test_fallback_unstructured_paragraphs(self):
        text = """
        What is Named Entity Recognition and what are its applications?
        
        How does the self-attention mechanism work in Transformer networks?
        """
        questions = self.segmenter.segment(text)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].question_id, "Q1")
        self.assertEqual(questions[1].question_id, "Q2")


if __name__ == "__main__":
    unittest.main()
