"""LEGACY: exports from live analysis objects. The app now exports from saved snapshots via services.record_export; kept (and tested) for programmatic use."""
"""
Export and Reporting Service for SyllabusLens.
Generates structured CSV files and Markdown / JSON analysis reports for download.
"""

import io
import json
import pandas as pd
from typing import List, Dict, Any
from services.analysis_service import CompleteAnalysisResult
from services.coverage_service import AggregateCoverageMetrics, SectionCoverageStat
from services.evaluation_service import MethodEvaluationReport


class ExportService:
    """
    Exports results, coverage analytics, and evaluation benchmarks.
    """

    def export_matches_csv(self, analysis: CompleteAnalysisResult) -> str:
        """Exports ranked question matches as CSV string."""
        rows = []
        for r in analysis.question_results:
            top_m = r.top_matches[0] if r.top_matches else None
            second_m = r.top_matches[1] if len(r.top_matches) > 1 else None

            rows.append({
                "Question ID": r.question_id,
                "Question Text": r.question_text,
                "Marks": r.marks or "",
                "Best Similarity Score": round(r.best_score, 4),
                "Match Tier": r.best_tier,
                "Top Section ID": top_m.section_id if top_m else "",
                "Top Document": top_m.doc_name if top_m else "",
                "Top Page/Slide": top_m.page_or_slide if top_m else "",
                "Top Section Heading": top_m.heading if top_m else "",
                "Top Key Terms": ", ".join([t.get("term", "") for t in (top_m.matching_terms[:5] if top_m else [])]),
                "Top Explanation": top_m.explanation if top_m else "",
                "Rank 2 Section": second_m.heading if second_m else "",
                "Rank 2 Score": round(second_m.similarity_score, 4) if second_m else 0.0
            })

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def export_coverage_csv(self, section_stats: List[SectionCoverageStat]) -> str:
        """Exports note-section coverage statistics as CSV string."""
        rows = []
        for s in section_stats:
            rows.append({
                "Section ID": s.section_id,
                "Document": s.doc_name,
                "Page/Slide": s.page_or_slide,
                "Heading": s.heading,
                "Max Similarity": round(s.max_similarity, 4),
                "Top-1 Hits": s.top1_match_count,
                "Top-3 Hits": s.top3_match_count,
                "Coverage Status": s.coverage_status,
                "Matched Questions": ", ".join(s.matched_questions)
            })

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def export_evaluation_csv(self, reports: List[MethodEvaluationReport]) -> str:
        """Exports comparative evaluation benchmark results as CSV string."""
        rows = []
        for rep in reports:
            rows.append({
                "Method": rep.method_name,
                "P@1": round(rep.precision_at_1, 4),
                "P@3": round(rep.precision_at_3, 4),
                "P@5": round(rep.precision_at_5, 4),
                "R@1": round(rep.recall_at_1, 4),
                "R@3": round(rep.recall_at_3, 4),
                "R@5": round(rep.recall_at_5, 4),
                "MRR": round(rep.mrr, 4),
                "MAP": round(rep.map_score, 4),
                "Evaluated Queries": rep.num_queries
            })

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def generate_markdown_report(
        self,
        analysis: CompleteAnalysisResult,
        metrics: AggregateCoverageMetrics,
        section_stats: List[SectionCoverageStat]
    ) -> str:
        """Generates a comprehensive executive summary in Markdown format."""
        md = []
        md.append("# SyllabusLens — Academic Intelligence Coverage Report\n")
        md.append(f"**Methodology:** {analysis.method_name}")
        md.append(f"**Overall Syllabus Coverage Rate:** {metrics.overall_coverage_rate:.1f}%\n")

        md.append("## 1. Executive Summary Metrics\n")
        md.append(f"- **Total Questions Evaluated:** {metrics.total_questions}")
        md.append(f"- **Total Note Sections Analyzed:** {metrics.total_sections}")
        md.append(f"- **Strong Match Questions:** {metrics.strong_count} ({metrics.strong_pct:.1f}%)")
        md.append(f"- **Moderate Match Questions:** {metrics.moderate_count} ({metrics.moderate_pct:.1f}%)")
        md.append(f"- **Weak Match Questions:** {metrics.weak_count} ({metrics.weak_pct:.1f}%)")
        md.append(f"- **Unmatched Questions:** {metrics.unmatched_count} ({metrics.unmatched_pct:.1f}%)")
        md.append(f"- **Mean Similarity Score:** {metrics.mean_similarity:.3f}")
        md.append(f"- **Uncovered Syllabus Sections:** {metrics.sections_uncovered_count} of {metrics.total_sections} ({100.0 - metrics.section_coverage_pct:.1f}%)\n")

        md.append("## 2. Uncovered Syllabus Areas (Not Detected in Exam Paper)\n")
        uncovered = [s for s in section_stats if s.coverage_status == "Uncovered in Exam"]
        if uncovered:
            md.append("| Section ID | Source Document | Page / Slide | Heading |")
            md.append("|---|---|---|---|")
            for u in uncovered[:15]:
                md.append(f"| {u.section_id} | {u.doc_name} | {u.page_or_slide} | {u.heading} |")
            if len(uncovered) > 15:
                md.append(f"\n*...and {len(uncovered) - 15} additional uncovered sections.*")
        else:
            md.append("All lecture note sections had at least one question match.")

        md.append("\n## 3. Question-to-Section Match Overview\n")
        md.append("| QID | Marks | Best Score | Tier | Top Retrieved Section | Source |")
        md.append("|---|---|---|---|---|---|")
        for qr in analysis.question_results:
            top_m = qr.top_matches[0] if qr.top_matches else None
            top_h = top_m.heading if top_m else "None"
            top_src = f"{top_m.doc_name} ({top_m.page_or_slide})" if top_m else "None"
            md.append(f"| {qr.question_id} | {qr.marks or '-'} | {qr.best_score:.3f} | {qr.best_tier} | {top_h} | {top_src} |")

        md.append("\n---\n*Report generated locally by SyllabusLens Academic Intelligence Engine.*")
        return "\n".join(md)
