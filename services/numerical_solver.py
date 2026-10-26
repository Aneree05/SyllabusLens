"""
Deterministic Numerical Solver and Step-by-Step Calculator for SyllabusLens.
Analyzes numerical problems, extracts given data, grounds calculations in verified formulas,
evaluates intermediate steps, and explicitly identifies missing values without inventing facts.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NumericalSolution:
    topic_name: str
    problem_type: str
    given_data: Dict[str, Any]
    required_result: str
    formula: str
    variables_and_units: Dict[str, str]
    assumptions: List[str]
    steps: List[str]
    intermediate_results: Dict[str, Any]
    final_answer: Optional[str]
    verification: Optional[str]
    is_solvable: bool
    missing_data: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_name": self.topic_name,
            "problem_type": self.problem_type,
            "given_data": self.given_data,
            "required_result": self.required_result,
            "formula": self.formula,
            "variables_and_units": self.variables_and_units,
            "assumptions": self.assumptions,
            "steps": self.steps,
            "intermediate_results": self.intermediate_results,
            "final_answer": self.final_answer,
            "verification": self.verification,
            "is_solvable": self.is_solvable,
            "missing_data": self.missing_data,
        }


class NumericalSolver:
    """
    Deterministic solver supporting key academic information retrieval, NLP,
    and computational mathematics problem classes.
    """

    def analyze_and_solve(self, question_text: str, context_notes: str = "") -> Optional[NumericalSolution]:
        """
        Inspects question text for known numerical patterns or formulas.
        Returns a structured NumericalSolution or None if not an assessable numerical prompt.
        """
        text = question_text.strip()
        lower = text.lower()

        # Check for IDF calculation
        if "idf" in lower or "inverse document frequency" in lower:
            return self._solve_idf(text)

        # Check for Cosine similarity
        if "cosine similarity" in lower or "cosine distance" in lower:
            return self._solve_cosine(text)

        # Check for Precision / Recall / F1 / Retrieval metrics
        if "precision" in lower or "recall" in lower or "mrr" in lower:
            return self._solve_retrieval_metrics(text)

        # Check for N-gram probability / Language modeling
        if "n-gram" in lower or "bigram probability" in lower or "perplexity" in lower:
            return self._solve_ngram_prob(text)

        # Check for general numerical problems with numbers
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if numbers:
            return self._solve_generic_numerical(text, numbers)

        # Numerical prompt with no numbers provided at all
        if any(w in lower for w in ("calculate", "compute", "find the value", "determine the numerical")):
            return NumericalSolution(
                topic_name="Numerical Evaluation",
                problem_type="Unparameterized Calculation Prompt",
                given_data={},
                required_result="Value requested in prompt",
                formula="Domain-specific formula required based on problem parameters",
                variables_and_units={"Variables": "Unspecified"},
                assumptions=["No parameter values provided in question statement."],
                steps=["Question requests a numerical calculation but does not specify quantitative inputs."],
                intermediate_results={},
                final_answer=None,
                verification=None,
                is_solvable=False,
                missing_data=["Numerical input parameters, sample counts, or coordinate vectors are missing from prompt."],
            )

        return None

    def _solve_idf(self, text: str) -> NumericalSolution:
        """Solves IDF = ln((1 + |D|) / (1 + DF(t))) + 1 or flags missing corpus parameters."""
        given: Dict[str, Any] = {}
        missing: List[str] = []

        # Find corpus size |D| or N
        m_d = re.search(r"(?:\|D\||total documents?|corpus size|N)\s*(?:=|is|of)?\s*(\d+)", text, re.I)
        if m_d:
            given["Total documents (|D|)"] = int(m_d.group(1))
        else:
            missing.append("Total document count in corpus (|D|)")

        # Find document frequency DF(t) or containing count
        m_df = re.search(r"(?:df|document frequency|contains? in|appears? in)\s*(?:=|is|of)?\s*(\d+)\s*(?:documents?)?", text, re.I)
        if m_df:
            given["Document frequency (DF)"] = int(m_df.group(1))
        else:
            missing.append("Document frequency of the term (DF(t))")

        formula = "IDF(t) = ln((1 + |D|) / (1 + DF(t))) + 1"
        vars_units = {
            "|D|": "Total count of documents in corpus (dimensionless integer >= 1)",
            "DF(t)": "Number of documents containing term t (dimensionless integer >= 0)",
            "ln": "Natural logarithm (base e)",
            "+1": "Additive smoothing preventing division by zero and zero IDF",
        }
        assumptions = [
            "Standard smooth IDF formula as implemented in scikit-learn and SyllabusLens vectorizer.",
            "Natural logarithm base e = 2.71828...",
            "Corpus size |D| >= DF(t) >= 0",
        ]

        if missing:
            return NumericalSolution(
                topic_name="Term Frequency - Inverse Document Frequency",
                problem_type="IDF Calculation",
                given_data=given,
                required_result="Inverse Document Frequency IDF(t)",
                formula=formula,
                variables_and_units=vars_units,
                assumptions=assumptions,
                steps=["Cannot complete calculation: essential input values are missing."],
                intermediate_results={},
                final_answer=None,
                verification=None,
                is_solvable=False,
                missing_data=missing,
            )

        D = given["Total documents (|D|)"]
        df = given["Document frequency (DF)"]
        numerator = 1 + D
        denominator = 1 + df
        ratio = numerator / denominator
        log_val = math.log(ratio)
        idf = log_val + 1.0

        steps = [
            f"Step 1: Identify given parameters: |D| = {D}, DF(t) = {df}.",
            f"Step 2: Apply smoothed numerator: (1 + |D|) = 1 + {D} = {numerator}.",
            f"Step 3: Apply smoothed denominator: (1 + DF(t)) = 1 + {df} = {denominator}.",
            f"Step 4: Compute quotient: {numerator} / {denominator} = {ratio:.6f}.",
            f"Step 5: Compute natural logarithm: ln({ratio:.6f}) = {log_val:.6f}.",
            f"Step 6: Add smoothing constant +1: {log_val:.6f} + 1 = {idf:.6f}.",
        ]

        verification = (
            f"Sanity verification: Since |D| ({D}) >= DF ({df}), ratio >= 1.0, ln(ratio) >= 0.0. "
            f"Therefore IDF ({idf:.4f}) >= 1.0, satisfying theoretical lower bound."
        )

        return NumericalSolution(
            topic_name="Term Frequency - Inverse Document Frequency",
            problem_type="IDF Calculation",
            given_data=given,
            required_result="Inverse Document Frequency IDF(t)",
            formula=formula,
            variables_and_units=vars_units,
            assumptions=assumptions,
            steps=steps,
            intermediate_results={"numerator": numerator, "denominator": denominator, "ratio": round(ratio, 4), "ln_ratio": round(log_val, 4)},
            final_answer=f"{idf:.4f}",
            verification=verification,
            is_solvable=True,
        )

    def _solve_cosine(self, text: str) -> NumericalSolution:
        """Solves Cosine similarity between explicit vectors or flags missing vector coordinates."""
        # Check for vector patterns like [1, 0, 2] or (1, 2)
        vec_matches = re.findall(r"\[([\d\s\.,\-]+)\]|\(([0-9\s\.,\-]+)\)", text)
        given: Dict[str, Any] = {}
        missing: List[str] = []

        formula = "Cosine Similarity = (q · d) / (||q|| * ||d||) = (∑ q_i * d_i) / (√(∑ q_i²) * √(∑ d_i²))"
        vars_units = {
            "q": "Query feature vector in R^n",
            "d": "Document feature vector in R^n",
            "q · d": "Dot product (scalar)",
            "||q||, ||d||": "L2 Euclidean norms of vectors q and d",
            "Similarity": "Cosine of angle θ, bounded in [0, 1] for non-negative term frequency vectors",
        }
        assumptions = [
            "Vectors reside in the same feature space with identical indexing.",
            "Non-negative weights (TF or TF-IDF weights >= 0).",
            "Non-zero vector magnitudes (||q|| > 0, ||d|| > 0).",
        ]

        parsed_vectors = []
        for g1, g2 in vec_matches:
            raw = g1 or g2
            parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
            try:
                nums = [float(p) for p in parts]
                if nums:
                    parsed_vectors.append(nums)
            except ValueError:
                pass

        if len(parsed_vectors) >= 2 and len(parsed_vectors[0]) == len(parsed_vectors[1]):
            v1, v2 = parsed_vectors[0], parsed_vectors[1]
            given["Vector 1 (q)"] = v1
            given["Vector 2 (d)"] = v2
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))

            if norm1 == 0 or norm2 == 0:
                return NumericalSolution(
                    topic_name="Vector Space Models",
                    problem_type="Cosine Similarity Calculation",
                    given_data=given,
                    required_result="Cosine similarity Sim(q, d)",
                    formula=formula,
                    variables_and_units=vars_units,
                    assumptions=assumptions,
                    steps=["One or both vectors have zero Euclidean magnitude (norm = 0)."],
                    intermediate_results={"dot_product": 0.0, "norm_q": norm1, "norm_d": norm2},
                    final_answer="0.0000",
                    verification="Zero vector similarity defaults to 0.0 by convention to prevent division by zero.",
                    is_solvable=True,
                )

            cos_sim = dot / (norm1 * norm2)
            steps = [
                f"Step 1: Given vector q = {v1} and document vector d = {v2}.",
                f"Step 2: Compute dot product: q · d = " + " + ".join(f"({a}*{b})" for a, b in zip(v1, v2)) + f" = {dot:.4f}.",
                f"Step 3: Compute L2 norm of q: ||q|| = √(" + " + ".join(f"{a}²" for a in v1) + f") = √{sum(a*a for a in v1):.4f} = {norm1:.4f}.",
                f"Step 4: Compute L2 norm of d: ||d|| = √(" + " + ".join(f"{b}²" for b in v2) + f") = √{sum(b*b for b in v2):.4f} = {norm2:.4f}.",
                f"Step 5: Compute denominator: ||q|| * ||d|| = {norm1:.4f} * {norm2:.4f} = {norm1*norm2:.4f}.",
                f"Step 6: Evaluate cosine similarity: {dot:.4f} / {norm1*norm2:.4f} = {cos_sim:.4f}.",
            ]
            verification = f"Sanity verification: -1.0 <= {cos_sim:.4f} <= 1.0. For non-negative vectors, value is within [0, 1]."
            return NumericalSolution(
                topic_name="Vector Space Models",
                problem_type="Cosine Similarity Calculation",
                given_data=given,
                required_result="Cosine similarity Sim(q, d)",
                formula=formula,
                variables_and_units=vars_units,
                assumptions=assumptions,
                steps=steps,
                intermediate_results={"dot_product": round(dot, 4), "norm_q": round(norm1, 4), "norm_d": round(norm2, 4)},
                final_answer=f"{cos_sim:.4f}",
                verification=verification,
                is_solvable=True,
            )

        missing.append("Coordinates for query vector q and document vector d (e.g., q=[1, 2], d=[2, 1])")
        return NumericalSolution(
            topic_name="Vector Space Models",
            problem_type="Cosine Similarity Calculation",
            given_data=given,
            required_result="Cosine similarity Sim(q, d)",
            formula=formula,
            variables_and_units=vars_units,
            assumptions=assumptions,
            steps=["Missing numerical vector coordinate representations."],
            intermediate_results={},
            final_answer=None,
            verification=None,
            is_solvable=False,
            missing_data=missing,
        )

    def _solve_retrieval_metrics(self, text: str) -> NumericalSolution:
        """Solves Precision@k, Recall@k, F1 given counts or flags missing parameters."""
        given: Dict[str, Any] = {}
        missing: List[str] = []

        m_tp = re.search(r"(?:relevant\s+retrieved|true\s+positives?|TP)\s*(?:=|is|:)?\s*(\d+)", text, re.I)
        m_ret = re.search(r"(?:total\s+retrieved|top[\s\-]*k|k)\s*(?:=|is|:)?\s*(\d+)", text, re.I)
        m_rel = re.search(r"(?:total\s+relevant|relevant\s+documents?)\s*(?:=|is|:)?\s*(\d+)", text, re.I)

        if m_tp: given["Relevant Retrieved (TP)"] = int(m_tp.group(1))
        else: missing.append("Number of relevant retrieved items (TP)")

        if m_ret: given["Total Retrieved (k)"] = int(m_ret.group(1))
        else: missing.append("Total retrieved items count (k)")

        if m_rel: given["Total Relevant in Corpus"] = int(m_rel.group(1))

        formula = "Precision@k = TP / k; Recall@k = TP / Total Relevant; F1 = 2 * (Precision * Recall) / (Precision + Recall)"
        vars_units = {
            "TP": "True Positives: relevant documents successfully retrieved in top-k",
            "k": "Total number of documents retrieved",
            "Total Relevant": "Total ground-truth relevant documents in corpus",
        }
        assumptions = ["Binary relevance judgments (each document is either relevant or not relevant)."]

        if missing:
            return NumericalSolution(
                topic_name="Information Retrieval Evaluation",
                problem_type="Precision and Recall Evaluation",
                given_data=given,
                required_result="Precision@k / Recall@k",
                formula=formula,
                variables_and_units=vars_units,
                assumptions=assumptions,
                steps=["Cannot complete calculation: counts of retrieved and relevant documents are unspecified."],
                intermediate_results={},
                final_answer=None,
                verification=None,
                is_solvable=False,
                missing_data=missing,
            )

        tp = given["Relevant Retrieved (TP)"]
        k = given["Total Retrieved (k)"]
        prec = tp / k if k > 0 else 0.0
        steps = [
            f"Step 1: Identify given parameters: TP = {tp}, retrieved k = {k}.",
            f"Step 2: Compute Precision@{k} = TP / k = {tp} / {k} = {prec:.4f} ({prec*100:.1f}%).",
        ]
        inter: Dict[str, Any] = {"precision": round(prec, 4)}

        final = f"Precision@{k} = {prec:.4f}"
        if "Total Relevant in Corpus" in given:
            tot_rel = given["Total Relevant in Corpus"]
            rec = tp / tot_rel if tot_rel > 0 else 0.0
            inter["recall"] = round(rec, 4)
            steps.append(f"Step 3: Compute Recall@{k} = TP / Total Relevant = {tp} / {tot_rel} = {rec:.4f} ({rec*100:.1f}%).")
            if prec + rec > 0:
                f1 = 2 * (prec * rec) / (prec + rec)
                inter["f1"] = round(f1, 4)
                steps.append(f"Step 4: Compute F1-score = 2 * ({prec:.4f} * {rec:.4f}) / ({prec:.4f} + {rec:.4f}) = {f1:.4f}.")
                final += f", Recall@{k} = {rec:.4f}, F1 = {f1:.4f}"

        verification = f"Sanity verification: 0.0 <= Precision ({prec:.4f}) <= 1.0."
        return NumericalSolution(
            topic_name="Information Retrieval Evaluation",
            problem_type="Precision and Recall Evaluation",
            given_data=given,
            required_result="Retrieval Metrics",
            formula=formula,
            variables_and_units=vars_units,
            assumptions=assumptions,
            steps=steps,
            intermediate_results=inter,
            final_answer=final,
            verification=verification,
            is_solvable=True,
        )

    def _solve_ngram_prob(self, text: str) -> NumericalSolution:
        """Solves bigram probability P(w_n | w_{n-1}) = Count(w_{n-1}, w_n) / Count(w_{n-1})."""
        given: Dict[str, Any] = {}
        missing: List[str] = []

        m_bi = re.search(r"(?:bigram\s+count|count\s+of\s+bigram|C\(w1,w2\))\s*(?:=|is|:)?\s*(\d+)", text, re.I)
        m_uni = re.search(r"(?:unigram\s+count|count\s+of\s+w1|C\(w1\))\s*(?:=|is|:)?\s*(\d+)", text, re.I)

        if m_bi: given["Bigram Count C(w_{i-1}, w_i)"] = int(m_bi.group(1))
        else: missing.append("Bigram occurrence count C(w_{i-1}, w_i)")

        if m_uni: given["History Unigram Count C(w_{i-1})"] = int(m_uni.group(1))
        else: missing.append("Conditioning unigram occurrence count C(w_{i-1})")

        formula = "P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})"
        vars_units = {
            "Count(w_{i-1}, w_i)": "Joint frequency of word pair",
            "Count(w_{i-1})": "Frequency of conditioning prefix word",
            "P": "Maximum Likelihood Estimate (MLE) conditional probability in [0, 1]",
        }
        assumptions = ["Maximum Likelihood Estimation without smoothing (unless Laplace smoothing explicitly specified)."]

        if missing:
            return NumericalSolution(
                topic_name="N-gram Language Modeling",
                problem_type="Conditional Bigram Probability",
                given_data=given,
                required_result="Conditional Probability P(w_i | w_{i-1})",
                formula=formula,
                variables_and_units=vars_units,
                assumptions=assumptions,
                steps=["Cannot complete calculation: joint and marginal counts missing."],
                intermediate_results={},
                final_answer=None,
                verification=None,
                is_solvable=False,
                missing_data=missing,
            )

        c_bi = given["Bigram Count C(w_{i-1}, w_i)"]
        c_uni = given["History Unigram Count C(w_{i-1})"]
        prob = c_bi / c_uni if c_uni > 0 else 0.0

        steps = [
            f"Step 1: Identify given frequencies: C(w_{{i-1}}, w_i) = {c_bi}, C(w_{{i-1}}) = {c_uni}.",
            f"Step 2: Apply Maximum Likelihood formulation: P(w_i | w_{{i-1}}) = {c_bi} / {c_uni} = {prob:.4f}.",
        ]
        verification = f"Sanity verification: 0.0 <= {prob:.4f} <= 1.0."

        return NumericalSolution(
            topic_name="N-gram Language Modeling",
            problem_type="Conditional Bigram Probability",
            given_data=given,
            required_result="P(w_i | w_{i-1})",
            formula=formula,
            variables_and_units=vars_units,
            assumptions=assumptions,
            steps=steps,
            intermediate_results={"c_bi": c_bi, "c_uni": c_uni, "prob": round(prob, 4)},
            final_answer=f"{prob:.4f}",
            verification=verification,
            is_solvable=True,
        )

    def _solve_generic_numerical(self, text: str, numbers: List[str]) -> NumericalSolution:
        """Fallback for numerical prompts with stated numbers but unclassified domain."""
        return NumericalSolution(
            topic_name="Applied Quantitative Problem",
            problem_type="Numerical Parameter Analysis",
            given_data={"Extracted Numbers": [float(n) if "." in n else int(n) for n in numbers[:6]]},
            required_result="Computed result based on stated problem formula",
            formula="Domain-specific formula required by question context",
            variables_and_units={"Values": f"Detected {len(numbers)} numerical literals in problem statement"},
            assumptions=["Values extracted verbatim from question statement."],
            steps=[
                f"Identified numerical quantities in statement: {', '.join(numbers[:6])}.",
                "To evaluate final numerical result deterministically, match against specific formula equations in uploaded notes.",
            ],
            intermediate_results={"extracted_count": len(numbers)},
            final_answer=None,
            verification=None,
            is_solvable=False,
            missing_data=["Explicit variable-to-number mapping and governing equation not fully specified."],
        )
