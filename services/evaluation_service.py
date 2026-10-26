"""
Evaluation and Experimentation Service for SyllabusLens.
Computes Precision@k, Recall@k, Mean Reciprocal Rank (MRR), Mean Average Precision (MAP),
and provides side-by-side benchmarking between Bag-of-Words and TF-IDF.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class MethodEvaluationReport:
    """Evaluation metrics for a specific retrieval model."""
    method_name: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    map_score: float
    num_queries: int
    query_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConfusionMatrixMetrics:
    """Binary coverage classification performance metrics."""
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class EvaluationService:
    """
    Rigorously assesses information retrieval and coverage classification
    using academic IR evaluation standards.
    """

    def evaluate_retrieval(
        self,
        method_name: str,
        retrieved_rankings: Dict[str, List[str]],  # {qid: [sec_id_rank1, sec_id_rank2, ...]}
        ground_truth: Dict[str, List[str]]          # {qid: [relevant_sec_id1, ...]}
    ) -> MethodEvaluationReport:
        """
        Calculates P@1, P@3, P@5, R@1, R@3, R@5, MRR, and MAP.
        """
        p1_list, p3_list, p5_list = [], [], []
        r1_list, r3_list, r5_list = [], [], []
        reciprocal_ranks = []
        average_precisions = []
        query_details = []

        eval_qids = [qid for qid in ground_truth if qid in retrieved_rankings]

        for qid in eval_qids:
            relevant_set: Set[str] = set(ground_truth[qid])
            retrieved_list: List[str] = retrieved_rankings[qid]

            if not relevant_set:
                continue

            # Precision & Recall @ K
            def calc_pk_rk(k: int) -> Tuple[float, float]:
                top_k = retrieved_list[:k]
                rel_ret = sum(1 for sec in top_k if sec in relevant_set)
                p_k = rel_ret / k
                r_k = rel_ret / len(relevant_set)
                return p_k, r_k

            p1, r1 = calc_pk_rk(1)
            p3, r3 = calc_pk_rk(3)
            p5, r5 = calc_pk_rk(5)

            p1_list.append(p1)
            p3_list.append(p3)
            p5_list.append(p5)
            r1_list.append(r1)
            r3_list.append(r3)
            r5_list.append(r5)

            # Reciprocal Rank (RR)
            rr = 0.0
            for rank_idx, sec_id in enumerate(retrieved_list, 1):
                if sec_id in relevant_set:
                    rr = 1.0 / rank_idx
                    break
            reciprocal_ranks.append(rr)

            # Average Precision (AP)
            cum_rel = 0
            prec_sum = 0.0
            for rank_idx, sec_id in enumerate(retrieved_list, 1):
                if sec_id in relevant_set:
                    cum_rel += 1
                    prec_sum += cum_rel / rank_idx

            ap = (prec_sum / len(relevant_set)) if len(relevant_set) > 0 else 0.0
            average_precisions.append(ap)

            query_details.append({
                "question_id": qid,
                "relevant_sections": list(relevant_set),
                "retrieved_top3": retrieved_list[:3],
                "p@1": round(p1, 3),
                "p@3": round(p3, 3),
                "r@3": round(r3, 3),
                "rr": round(rr, 3),
                "ap": round(ap, 3)
            })

        return MethodEvaluationReport(
            method_name=method_name,
            precision_at_1=float(np.mean(p1_list)) if p1_list else 0.0,
            precision_at_3=float(np.mean(p3_list)) if p3_list else 0.0,
            precision_at_5=float(np.mean(p5_list)) if p5_list else 0.0,
            recall_at_1=float(np.mean(r1_list)) if r1_list else 0.0,
            recall_at_3=float(np.mean(r3_list)) if r3_list else 0.0,
            recall_at_5=float(np.mean(r5_list)) if r5_list else 0.0,
            mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            map_score=float(np.mean(average_precisions)) if average_precisions else 0.0,
            num_queries=len(p1_list),   # queries actually scored (those with >=1 relevant section)
            query_details=query_details
        )

    def evaluate_coverage_classification(
        self,
        predicted_scores: Dict[str, float],   # {qid: max_similarity_score}
        ground_truth_has_match: Dict[str, bool], # {qid: True if covered in syllabus, False if out of syllabus}
        threshold: float = 0.20
    ) -> ConfusionMatrixMetrics:
        """
        Formulates coverage detection as binary classification:
        Positive = Question is present in notes
        Negative = Question is outside syllabus / uncovered
        """
        tp = fp = tn = fn = 0

        for qid, actual_covered in ground_truth_has_match.items():
            score = predicted_scores.get(qid, 0.0)
            pred_covered = (score >= threshold)

            if actual_covered and pred_covered:
                tp += 1
            elif not actual_covered and pred_covered:
                fp += 1
            elif not actual_covered and not pred_covered:
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return ConfusionMatrixMetrics(
            threshold=threshold,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1
        )
