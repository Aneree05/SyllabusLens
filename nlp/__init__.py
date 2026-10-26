"""
NLP package for SyllabusLens.
Exposes modular text processing, segmentation, vectorization, and similarity algorithms.
"""

from nlp.preprocessing import TextPreprocessor, PreprocessedResult, default_preprocessor, preprocess_text
from nlp.stemmer import PorterStemmer, stem_word
from nlp.question_segmentation import QuestionSegmenter, QuestionItem
from nlp.document_segmentation import DocumentSegmenter, NoteSection
from nlp.vectorization import VectorizerEngine
from nlp.similarity import SimilarityEngine, MatchItem, QuestionMatchResult
from nlp.keyword_explanations import KeywordExplainer
from nlp.optional_linguistics import RuleBasedLinguisticsEngine

__all__ = [
    "TextPreprocessor",
    "PreprocessedResult",
    "default_preprocessor",
    "preprocess_text",
    "PorterStemmer",
    "stem_word",
    "QuestionSegmenter",
    "QuestionItem",
    "DocumentSegmenter",
    "NoteSection",
    "VectorizerEngine",
    "SimilarityEngine",
    "MatchItem",
    "QuestionMatchResult",
    "KeywordExplainer",
    "RuleBasedLinguisticsEngine"
]
