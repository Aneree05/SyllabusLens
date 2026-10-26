"""
Explainable Matching and Keyword Contribution Module for SyllabusLens.
Deconstructs vector dot products to explain why a note section was retrieved.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.sparse import csr_matrix


class KeywordExplainer:
    """
    Computes term-level contributions to the cosine similarity
    between question vectors and note section vectors.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_match(
        self,
        q_vec: csr_matrix,
        doc_vec: csr_matrix,
        top_n: int = 5
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Computes element-wise product of question and document vectors (q_i * d_i)
        to identify the specific terms responsible for the similarity score.
        """
        # Element-wise product of sparse vectors
        # Note: both vectors are 1 x V
        prod = q_vec.multiply(doc_vec)

        if prod.nnz == 0:
            return [], "No overlapping vocabulary terms detected. Vector dot product is zero."

        # Extract non-zero indices and values
        coo = prod.tocoo()
        contributions = []
        for idx, val in zip(coo.col, coo.data):
            term = self.feature_names[idx]
            contributions.append({
                "term": term,
                "contribution": float(val),
                "q_weight": float(q_vec[0, idx]),
                "doc_weight": float(doc_vec[0, idx])
            })

        # Sort by contribution descending
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        top_terms = contributions[:top_n]

        # Generate readable explanation summary
        terms_str = ", ".join([f"'{t['term']}' ({t['contribution']:.3f})" for t in top_terms])
        explanation = f"Primary matching features: {terms_str}."

        return top_terms, explanation
