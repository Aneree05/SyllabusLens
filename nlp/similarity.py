"""
Cosine Similarity and Match Ranking Module for SyllabusLens.
Computes similarity between questions and note sections and ranks matches.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from config import DEFAULT_STRONG_THRESHOLD, DEFAULT_MODERATE_THRESHOLD, DEFAULT_WEAK_THRESHOLD


@dataclass
class MatchItem:
    """Represents a single ranked match between a question and a note section."""
    rank: int
    section_id: str
    doc_name: str
    page_or_slide: str
    heading: str
    similarity_score: float
    match_tier: str   # "Strong", "Moderate", "Weak", "Unmatched"
    section_snippet: str
    matching_terms: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "section_id": self.section_id,
            "doc_name": self.doc_name,
            "page_or_slide": self.page_or_slide,
            "heading": self.heading,
            "similarity_score": round(self.similarity_score, 4),
            "match_tier": self.match_tier,
            "section_snippet": self.section_snippet,
            "explanation": self.explanation,
            "key_terms": ", ".join([t.get("term", "") for t in self.matching_terms[:5]])
        }


@dataclass
class QuestionMatchResult:
    """Represents complete analysis for an individual question."""
    question_id: str
    question_text: str
    clean_text: str
    marks: Optional[str]
    best_score: float
    best_tier: str
    top_matches: List[MatchItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        top1 = self.top_matches[0] if self.top_matches else None
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "marks": self.marks or "",
            "best_score": round(self.best_score, 4),
            "best_tier": self.best_tier,
            "top_section_id": top1.section_id if top1 else "None",
            "top_doc_name": top1.doc_name if top1 else "None",
            "top_page_or_slide": top1.page_or_slide if top1 else "None",
            "top_heading": top1.heading if top1 else "None",
            "top_key_terms": ", ".join([t.get("term", "") for t in (top1.matching_terms[:5] if top1 else [])]),
            "explanation": top1.explanation if top1 else "No match detected"
        }


class SimilarityEngine:
    """
    Computes pairwise cosine similarity between questions and note sections.
    """

    def __init__(
        self,
        strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
        moderate_threshold: float = DEFAULT_MODERATE_THRESHOLD,
        weak_threshold: float = DEFAULT_WEAK_THRESHOLD
    ):
        self.strong_threshold = strong_threshold
        self.moderate_threshold = moderate_threshold
        self.weak_threshold = weak_threshold

    def classify_tier(self, score: float) -> str:
        """Classifies a similarity score into a human-readable match tier."""
        if score >= self.strong_threshold:
            return "Strong"
        elif score >= self.moderate_threshold:
            return "Moderate"
        elif score >= self.weak_threshold:
            return "Weak"
        else:
            return "Unmatched"

    def compute_similarity_matrix(
        self,
        q_matrix: csr_matrix,
        section_matrix: csr_matrix
    ) -> np.ndarray:
        """
        Calculates pairwise cosine similarity between question vectors and section vectors.
        Output shape: (num_questions, num_sections).
        Handles zero vectors safely (outputs 0.0 without NaN/inf).
        """
        if q_matrix.shape[0] == 0 or section_matrix.shape[0] == 0:
            return np.zeros((q_matrix.shape[0], section_matrix.shape[0]), dtype=np.float32)

        sim_matrix = cosine_similarity(q_matrix, section_matrix)
        # Clean any potential NaN values from zero vectors
        sim_matrix = np.nan_to_num(sim_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return sim_matrix

    def rank_top_matches(
        self,
        q_idx: int,
        scores_row: np.ndarray,
        sections: List[Any],
        top_k: int = 3
    ) -> List[MatchItem]:
        """
        Ranks top-k note sections for a single question given similarity scores.
        """
        num_sections = len(scores_row)
        if num_sections == 0:
            return []

        # Argsort descending
        sorted_indices = np.argsort(-scores_row)
        matches: List[MatchItem] = []

        k = min(top_k, num_sections)
        for rank in range(1, k + 1):
            s_idx = int(sorted_indices[rank - 1])
            score = float(scores_row[s_idx])
            sec = sections[s_idx]

            # Snippet of section text (first 200 characters)
            snippet = (sec.text or "").strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."

            tier = self.classify_tier(score)

            matches.append(MatchItem(
                rank=rank,
                section_id=sec.section_id,
                doc_name=sec.doc_name,
                page_or_slide=sec.page_or_slide,
                heading=sec.heading,
                similarity_score=score,
                match_tier=tier,
                section_snippet=snippet
            ))

        return matches
