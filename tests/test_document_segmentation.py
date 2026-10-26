"""
Unit tests for document segmentation across PDF pages, PPTX slides, and TXT units.
"""

import unittest
from nlp.document_segmentation import DocumentSegmenter


class TestDocumentSegmentation(unittest.TestCase):

    def setUp(self):
        self.segmenter = DocumentSegmenter(target_chunk_words=100, chunk_overlap_words=20)

    def test_pptx_slide_segmentation(self):
        slides_data = [
            {"slide_num": 1, "title": "Introduction to NLP", "text": "Overview of natural language processing and applications."},
            {"slide_num": 2, "title": "Vector Space Models", "text": "Bag of words, TF-IDF term weighting, and cosine similarity metric."}
        ]
        sections = self.segmenter.segment_pptx_slides("lecture1.pptx", slides_data)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].page_or_slide, "Slide 1")
        self.assertIn("Introduction to NLP", sections[0].heading)
        self.assertEqual(sections[1].page_or_slide, "Slide 2")

    def test_pdf_page_segmentation(self):
        pages = [
            "Chapter 1: Foundations\nText tokenization and normalization techniques.",
            "Chapter 2: Sequence Modeling\nHidden Markov Models and POS tagging."
        ]
        sections = self.segmenter.segment_pdf_pages("course.pdf", pages)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].page_or_slide, "Page 1")
        self.assertEqual(sections[1].page_or_slide, "Page 2")

    def test_txt_unit_segmentation(self):
        txt_content = """
## Unit 1: Foundations
Text tokenization, lowercasing, and stop words removal.

## Unit 2: Vector Space
TF-IDF, Bag of Words, and document retrieval.
        """
        sections = self.segmenter.segment_txt_document("notes.txt", txt_content)
        self.assertEqual(len(sections), 2)
        self.assertIn("Sec 1", sections[0].heading)
        self.assertIn("Sec 2", sections[1].heading)


if __name__ == "__main__":
    unittest.main()
