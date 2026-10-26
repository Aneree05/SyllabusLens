"""
Coverage Analytics Service for SyllabusLens.
Computes topic, section, and document-level coverage metrics,
identifies uncovered syllabus areas, and formats data for interactive visualizations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from services.analysis_service import CompleteAnalysisResult


@dataclass
class SectionCoverageStat:
    """Coverage summary for a specific lecture note section."""
    section_id: str
    doc_name: str
    page_or_slide: str
    heading: str
    max_similarity: float
    top1_match_count: int
    top3_match_count: int
    coverage_status: str  # "Covered", "Partially Covered", "Uncovered in Exam"
    matched_questions: List[str] = field(default_factory=list)


@dataclass
class AggregateCoverageMetrics:
    """High-level summary metrics for the entire analysis."""
    total_questions: int
    total_sections: int
    total_documents: int
    strong_count: int
    strong_pct: float
    moderate_count: int
    moderate_pct: float
    weak_count: int
    weak_pct: float
    unmatched_count: int
    unmatched_pct: float
    overall_coverage_rate: float   # share of questions whose best match is Strong or Moderate (NOT syllabus coverage)
    mean_similarity: float
    median_similarity: float
    max_similarity: float
    sections_covered_count: int
    sections_uncovered_count: int
    section_coverage_pct: float


class CoverageService:
    """
    Computes rigorous coverage statistics from actual similarity matrices and rankings.
    """

    def compute_coverage(self, analysis: CompleteAnalysisResult) -> Tuple[AggregateCoverageMetrics, List[SectionCoverageStat]]:
        """
        Calculates all aggregate metrics and section-by-section coverage summaries.
        """
        questions = analysis.questions
        sections = analysis.note_sections
        results = analysis.question_results
        sim_matrix = analysis.similarity_matrix

        total_q = len(questions)
        total_s = len(sections)
        docs = list(set([s.doc_name for s in sections]))

        if total_q == 0 or total_s == 0:
            empty_agg = AggregateCoverageMetrics(
                total_questions=total_q,
                total_sections=total_s,
                total_documents=len(docs),
                strong_count=0, strong_pct=0.0,
                moderate_count=0, moderate_pct=0.0,
                weak_count=0, weak_pct=0.0,
                unmatched_count=0, unmatched_pct=0.0,
                overall_coverage_rate=0.0,
                mean_similarity=0.0, median_similarity=0.0, max_similarity=0.0,
                sections_covered_count=0, sections_uncovered_count=0, section_coverage_pct=0.0
            )
            return empty_agg, []

        # Question tier breakdown
        strong_c = sum(1 for r in results if r.best_tier == "Strong")
        mod_c = sum(1 for r in results if r.best_tier == "Moderate")
        weak_c = sum(1 for r in results if r.best_tier == "Weak")
        unmatched_c = sum(1 for r in results if r.best_tier == "Unmatched")

        strong_pct = (strong_c / total_q) * 100.0
        mod_pct = (mod_c / total_q) * 100.0
        weak_pct = (weak_c / total_q) * 100.0
        unmatched_pct = (unmatched_c / total_q) * 100.0

        # Overall coverage = (Strong + Moderate) / Total
        overall_cov = ((strong_c + mod_c) / total_q) * 100.0

        best_scores = [r.best_score for r in results]
        mean_sim = float(np.mean(best_scores))
        median_sim = float(np.median(best_scores))
        max_sim = float(np.max(best_scores))

        # Section-level aggregation uses the FULL question x section matrix, not just
        # each question's top-k list (a section can be strongly similar to a question
        # without being in that question's top-k).
        sec_index = {sec.section_id: i for i, sec in enumerate(sections)}
        col_max = sim_matrix.max(axis=0) if sim_matrix.size else np.zeros(total_s)
        sec_max_scores = {sid: float(col_max[i]) for sid, i in sec_index.items()}
        sec_top1_counts = {sid: 0 for sid in sec_index}
        sec_top3_counts = {sid: 0 for sid in sec_index}
        sec_matched_qs = {sid: [] for sid in sec_index}

        for q_i, r in enumerate(results):
            row = sim_matrix[q_i]
            for s_i, sec in enumerate(sections):
                if row[s_i] >= analysis.weak_threshold:
                    sec_matched_qs[sec.section_id].append(r.question_id)
            for m_idx, m in enumerate(r.top_matches):
                if m.section_id not in sec_index or m.similarity_score < analysis.weak_threshold:
                    continue
                sec_top3_counts[m.section_id] += 1
                if m_idx == 0:
                    sec_top1_counts[m.section_id] += 1

        section_stats: List[SectionCoverageStat] = []
        covered_sec_count = 0

        for s in sections:
            m_score = sec_max_scores[s.section_id]
            top1_c = sec_top1_counts[s.section_id]
            top3_c = sec_top3_counts[s.section_id]
            matched_q_list = list(dict.fromkeys(sec_matched_qs[s.section_id]))

            if m_score >= analysis.moderate_threshold:
                status = "Covered"
                covered_sec_count += 1
            elif m_score >= analysis.weak_threshold:
                status = "Partially Covered"
                covered_sec_count += 1
            else:
                status = "Uncovered in Exam"

            section_stats.append(SectionCoverageStat(
                section_id=s.section_id,
                doc_name=s.doc_name,
                page_or_slide=s.page_or_slide,
                heading=s.heading,
                max_similarity=m_score,
                top1_match_count=top1_c,
                top3_match_count=top3_c,
                coverage_status=status,
                matched_questions=matched_q_list
            ))

        uncovered_sec_count = total_s - covered_sec_count
        sec_cov_pct = (covered_sec_count / total_s) * 100.0 if total_s > 0 else 0.0

        agg_metrics = AggregateCoverageMetrics(
            total_questions=total_q,
            total_sections=total_s,
            total_documents=len(docs),
            strong_count=strong_c,
            strong_pct=strong_pct,
            moderate_count=mod_c,
            moderate_pct=mod_pct,
            weak_count=weak_c,
            weak_pct=weak_pct,
            unmatched_count=unmatched_c,
            unmatched_pct=unmatched_pct,
            overall_coverage_rate=overall_cov,
            mean_similarity=mean_sim,
            median_similarity=median_sim,
            max_similarity=max_sim,
            sections_covered_count=covered_sec_count,
            sections_uncovered_count=uncovered_sec_count,
            section_coverage_pct=sec_cov_pct
        )

        return agg_metrics, section_stats

    def prepare_sankey_data(self, analysis: CompleteAnalysisResult, min_threshold: float = 0.12) -> Dict[str, Any]:
        """
        Nodes/links for a Questions -> Note Sections Sankey diagram.
        Nodes are keyed by unique IDs (never by truncated headings); labels are display-only.
        """
        q_ids = [r.question_id for r in analysis.question_results]
        s_ids = [s.section_id for s in analysis.note_sections]
        node_ids = [("q", i) for i in q_ids] + [("s", i) for i in s_ids]
        node_map = {key: idx for idx, key in enumerate(node_ids)}
        labels = list(q_ids) + [s.heading[:40] for s in analysis.note_sections]

        sources, targets, values = [], [], []
        for r in analysis.question_results:
            for m in r.top_matches[:2]:
                if m.similarity_score >= min_threshold and ("s", m.section_id) in node_map:
                    sources.append(node_map[("q", r.question_id)])
                    targets.append(node_map[("s", m.section_id)])
                    values.append(max(0.01, m.similarity_score))
        return {
            "node_labels": labels,
            "sources": sources,
            "targets": targets,
            "values": values,
            "colors": ["rgba(79, 70, 229, 0.35)"] * len(values),
        }
