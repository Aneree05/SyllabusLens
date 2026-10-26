"""Multi-paper comparison within one subject. Reports observed history only - never predictions."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

from services.analysis_service import CompleteAnalysisResult


def compare_papers(analysis: CompleteAnalysisResult, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A section is "hit" by a paper when ANY question of that paper has similarity >= the moderate
    threshold with it (full question x section matrix, not just top-k lists).
    """
    thr = analysis.moderate_threshold
    sims = analysis.similarity_matrix
    paper_ids = [p["document_id"] for p in papers]
    rows = {pid: [i for i, q in enumerate(analysis.questions) if q.metadata.get("paper_doc_id") == pid]
            for pid in paper_ids}
    sec_ids = [s.section_id for s in analysis.note_sections]

    hit_sets: Dict[str, set] = {}
    per_paper = []
    for p in papers:
        idx = rows[p["document_id"]]
        mx = sims[idx].max(axis=0) if idx else None
        hits = {sec_ids[j] for j in range(len(sec_ids)) if mx is not None and mx[j] >= thr}
        hit_sets[p["document_id"]] = hits
        tiers = {"Strong": 0, "Moderate": 0, "Weak": 0, "Unmatched": 0}
        for i in idx:
            tiers[analysis.question_results[i].best_tier] += 1
        per_paper.append(dict(document_id=p["document_id"], title=p["title"], filename=p["filename"],
                              exam_session=p.get("exam_session", ""), academic_year=p.get("academic_year", ""),
                              questions=len(idx), sections_hit=len(hits), tiers=tiers))

    pairwise = []
    for a, b in combinations(paper_ids, 2):
        union = hit_sets[a] | hit_sets[b]
        inter = hit_sets[a] & hit_sets[b]
        pairwise.append(dict(paper_a=a, paper_b=b, shared_sections=len(inter), union_sections=len(union),
                             jaccard=(len(inter) / len(union)) if union else None))

    section_freq = []
    for j, s in enumerate(analysis.note_sections):
        hitting = [pid for pid in paper_ids if s.section_id in hit_sets[pid]]
        best = {pid: float(sims[rows[pid], j].max()) if rows[pid] else 0.0 for pid in paper_ids}
        section_freq.append(dict(section_id=s.section_id, heading=s.heading, doc_name=s.doc_name,
                                 locator=s.page_or_slide, papers_hit=len(hitting), papers_total=len(paper_ids),
                                 paper_ids=hitting, best_by_paper=best))
    return dict(
        basis=f"Section counted as hit by a paper when any of its questions has similarity >= {thr:.2f} "
              "(moderate threshold). Counts describe the selected papers only; they are not forecasts.",
        papers=per_paper, pairwise=pairwise,
        repeated_sections=[s for s in section_freq if s["papers_hit"] >= 2],
        section_frequency=section_freq)
