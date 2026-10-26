"""
Vectorization module for SyllabusLens.
Provides Bag-of-Words and TF-IDF feature representations using scikit-learn.
Handles edge cases (empty vocabularies, all-stopword texts, zero vectors).
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix, issparse


class EmptyVocabularyError(ValueError):
    """Raised when there is no usable text to build a vocabulary from."""


class VectorizerEngine:
    """
    Manages vector representations (Bag-of-Words and TF-IDF)
    for academic texts, questions, and lecture note sections.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = 3000,
        use_idf: bool = True,
        norm: str = "l2"
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.use_idf = use_idf
        self.norm = norm

        self.bow_vectorizer: Optional[CountVectorizer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.feature_names: List[str] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf_weights: Dict[str, float] = {}

    def fit(self, corpus: List[str]) -> "VectorizerEngine":
        """
        Fits both BoW and TF-IDF vectorizers on the provided text corpus.

        Blank entries are ignored. If nothing usable remains, or every term is
        filtered out (e.g. only stop words), EmptyVocabularyError is raised so the
        caller can report it. No placeholder vocabulary is ever substituted.
        """
        valid_texts = [t.strip() for t in corpus if t and t.strip()]
        if not valid_texts:
            raise EmptyVocabularyError(
                "The selected documents contain no usable text. Check that extraction "
                "succeeded (scanned PDFs need OCR) and that the documents are not empty."
            )

        token_pattern = r"(?u)\b\w+\b"
        self.bow_vectorizer = CountVectorizer(
            ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
            max_features=self.max_features, token_pattern=token_pattern,
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range, min_df=self.min_df, max_df=self.max_df,
            max_features=self.max_features, use_idf=self.use_idf, norm=self.norm,
            token_pattern=token_pattern,
        )
        try:
            self.bow_vectorizer.fit(valid_texts)
            self.tfidf_vectorizer.fit(valid_texts)
        except ValueError as exc:
            raise EmptyVocabularyError(
                "No vocabulary could be built: after stop-word removal and frequency "
                f"filtering no terms remain ({exc})."
            ) from exc

        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        self.vocabulary = self.tfidf_vectorizer.vocabulary_
        if hasattr(self.tfidf_vectorizer, "idf_"):
            self.idf_weights = {
                feat: float(self.tfidf_vectorizer.idf_[idx])
                for feat, idx in self.vocabulary.items()
            }
        return self

    def transform_bow(self, texts: List[str]) -> csr_matrix:
        """Transforms texts to Bag-of-Words sparse count matrix."""
        if not self.bow_vectorizer:
            raise RuntimeError("VectorizerEngine must be fitted before calling transform_bow().")
        safe_texts = [t if t and t.strip() else "" for t in texts]
        return self.bow_vectorizer.transform(safe_texts)

    def transform_tfidf(self, texts: List[str]) -> csr_matrix:
        """Transforms texts to TF-IDF sparse matrix."""
        if not self.tfidf_vectorizer:
            raise RuntimeError("VectorizerEngine must be fitted before calling transform_tfidf().")
        safe_texts = [t if t and t.strip() else "" for t in texts]
        return self.tfidf_vectorizer.transform(safe_texts)

    def get_sparsity(self, matrix: csr_matrix) -> float:
        """Calculates matrix sparsity percentage (0.0 to 100.0)."""
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return 100.0
        total_elements = matrix.shape[0] * matrix.shape[1]
        non_zero = matrix.nnz
        return (1.0 - (non_zero / total_elements)) * 100.0

    def inspect_text_vector(
        self, text: str, top_n: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Returns a detailed table of terms, their counts, IDF weights,
        and final TF-IDF scores for a specific text string.
        """
        if not self.tfidf_vectorizer or not self.bow_vectorizer:
            return []

        bow_vec = self.transform_bow([text]).toarray()[0]
        tfidf_vec = self.transform_tfidf([text]).toarray()[0]

        results = []
        non_zero_indices = np.where(bow_vec > 0)[0]

        for idx in non_zero_indices:
            term = self.feature_names[idx]
            count = int(bow_vec[idx])
            idf = float(self.tfidf_vectorizer.idf_[idx]) if hasattr(self.tfidf_vectorizer, "idf_") else 1.0
            score = float(tfidf_vec[idx])
            results.append({
                "term": term,
                "count": count,
                "idf": round(idf, 4),
                "tfidf": round(score, 4)
            })

        # Sort by TF-IDF score descending
        results.sort(key=lambda x: x["tfidf"], reverse=True)
        return results[:top_n]
