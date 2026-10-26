"""
Question Classifier Module for SyllabusLens.
Deterministically categorises exam questions into distinct pedagogical categories
using local regex rules, grammatical patterns, and marks weighting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


# Canonical category identifiers
CAT_SHORT_ANSWER = "short_answer"
CAT_LONG_ANSWER = "long_answer"
CAT_COMPARISON = "comparison"
CAT_DERIVATION = "derivation"
CAT_NUMERICAL = "numerical"
CAT_DIAGRAM = "diagram"

CATEGORY_NAMES = {
    CAT_SHORT_ANSWER: "Short-Answer & Definition",
    CAT_LONG_ANSWER: "Long-Answer & Theory",
    CAT_COMPARISON: "Comparison & Distinction",
    CAT_DERIVATION: "Derivation & Proof",
    CAT_NUMERICAL: "Numerical Problem",
    CAT_DIAGRAM: "Diagram-Based",
}

CATEGORY_BADGE_TONES = {
    CAT_SHORT_ANSWER: "info",
    CAT_LONG_ANSWER: "primary",
    CAT_COMPARISON: "success",
    CAT_DERIVATION: "warning",
    CAT_NUMERICAL: "primary",
    CAT_DIAGRAM: "info",
}


# Heuristic Regex Patterns
_RE_NUMERICAL = re.compile(
    r"\b(calculate|compute|evaluate the value|find the value|determine the numerical|"
    r"given that|frequency of|sample rate|probability of|value of)\b|"
    r"\b\d+\.?\d*\s*(?:khz|mhz|hz|ms|sec|seconds|kb|mb|bytes|bits|km|meters|volts|watts|joules|%)\b|"
    r"(?:\|D\|\s*=|TF\s*=|IDF\s*=|Sim\(|P\([A-Z]\)\s*=|\bN\s*=\s*\d+)",
    re.IGNORECASE,
)

_RE_COMPARISON = re.compile(
    r"\b(differentiate\s+(?:between|from)|distinguish\s+(?:between|from)|"
    r"difference\s+between|differences\s+between|compare\s+(?:and|with|to)?|"
    r"comparison\s+between|contrast\s+between|versus|\bvs\.?\b)\b",
    re.IGNORECASE,
)

_RE_DERIVATION = re.compile(
    r"\b(derive|derivation\s+of|prove\s+that|proof\s+of|show\s+that|"
    r"mathematical\s+proof|formulate\s+the\s+mathematical\s+equation|"
    r"deduce\s+the\s+formula|derive\s+an\s+expression)\b",
    re.IGNORECASE,
)

_RE_DIAGRAM = re.compile(
    r"\b(draw|sketch|schematic|diagram\s+of|flowchart|block\s+diagram|"
    r"circuit\s+diagram|labelled\s+diagram|architectural\s+diagram|"
    r"illustrate\s+with\s+(?:a\s+)?diagram|draw\s+and\s+explain)\b",
    re.IGNORECASE,
)

_RE_SHORT_ANSWER = re.compile(
    r"^(define|what\s+is|what\s+are|state\s+the|list\s+(?:down|the)?|"
    r"name\s+the|mention\s+the|give\s+(?:two|three|any)?\s+examples?|"
    r"write\s+short\s+notes?\s+on|define\s+the\s+term)\b",
    re.IGNORECASE,
)

_RE_LONG_ANSWER = re.compile(
    r"\b(explain\s+in\s+detail|discuss\s+in\s+detail|describe\s+the\s+architecture|"
    r"explain\s+the\s+architecture|elaborate\s+on|comprehensively\s+discuss|"
    r"critically\s+examine|explain\s+how|discuss\s+the\s+role)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    category_name: str
    rationale: str
    marks_weight: Optional[float] = None


class QuestionClassifier:
    """
    Classifies questions into 6 distinct categories with documented rationale.
    Order of evaluation balances specific cue triggers (comparison, derivation, numerical, diagram)
    before general short vs long theory categorisation.
    """

    @staticmethod
    def parse_marks(marks_str: Optional[str]) -> Optional[float]:
        if not marks_str:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", str(marks_str))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def classify(self, text: str, marks: Optional[str] = None) -> ClassificationResult:
        cleaned = text.strip()
        parsed_marks = self.parse_marks(marks)

        # 1. Comparison & Distinction
        m_comp = _RE_COMPARISON.search(cleaned)
        if m_comp:
            return ClassificationResult(
                category=CAT_COMPARISON,
                category_name=CATEGORY_NAMES[CAT_COMPARISON],
                rationale=f"Detected comparative phrase '{m_comp.group(0)}'.",
                marks_weight=parsed_marks,
            )

        # 2. Derivations & Proofs
        m_deriv = _RE_DERIVATION.search(cleaned)
        if m_deriv:
            return ClassificationResult(
                category=CAT_DERIVATION,
                category_name=CATEGORY_NAMES[CAT_DERIVATION],
                rationale=f"Detected mathematical proof / derivation trigger '{m_deriv.group(0)}'.",
                marks_weight=parsed_marks,
            )

        # 3. Numerical Problems
        m_num = _RE_NUMERICAL.search(cleaned)
        if m_num:
            return ClassificationResult(
                category=CAT_NUMERICAL,
                category_name=CATEGORY_NAMES[CAT_NUMERICAL],
                rationale=f"Detected calculation / numerical indicator '{m_num.group(0)}'.",
                marks_weight=parsed_marks,
            )

        # 4. Diagram-Based Questions
        m_diag = _RE_DIAGRAM.search(cleaned)
        if m_diag:
            return ClassificationResult(
                category=CAT_DIAGRAM,
                category_name=CATEGORY_NAMES[CAT_DIAGRAM],
                rationale=f"Detected graphical illustration / diagram request '{m_diag.group(0)}'.",
                marks_weight=parsed_marks,
            )

        # 5. Short-Answer & Definition
        m_short = _RE_SHORT_ANSWER.search(cleaned)
        word_count = len(cleaned.split())
        if m_short or (parsed_marks is not None and parsed_marks <= 3.0) or (word_count <= 14 and not _RE_LONG_ANSWER.search(cleaned)):
            cue = f"'{m_short.group(0)}'" if m_short else f"short prompt ({word_count} words)"
            if parsed_marks is not None and parsed_marks <= 3.0:
                cue += f", {parsed_marks:g} marks"
            return ClassificationResult(
                category=CAT_SHORT_ANSWER,
                category_name=CATEGORY_NAMES[CAT_SHORT_ANSWER],
                rationale=f"Triggered by definition / brief prompt cue: {cue}.",
                marks_weight=parsed_marks,
            )

        # 6. Long-Answer & Theory (Default for detailed conceptual questions)
        m_long = _RE_LONG_ANSWER.search(cleaned)
        detail = f"'{m_long.group(0)}'" if m_long else "comprehensive theory structure"
        if parsed_marks is not None:
            detail += f", {parsed_marks:g} marks"
        return ClassificationResult(
            category=CAT_LONG_ANSWER,
            category_name=CATEGORY_NAMES[CAT_LONG_ANSWER],
            rationale=f"Explanatory / theoretical inquiry: {detail}.",
            marks_weight=parsed_marks,
        )
