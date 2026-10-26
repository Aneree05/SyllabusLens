"""
Rule-based, evidence-carrying study recommendations built from a saved analysis record.

Every recommendation names the rule that produced it and lists its evidence. Counts describe
the selected historical papers. Nothing here forecasts future exam content.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

RULES = {
    "R1": "Several weak-match questions share the same best-matching section.",
    "R2": "Section is matched by questions in two or more of the selected papers (observed repetition).",
    "R3": "Syllabus topic has no question reaching the weak-match threshold in the selected papers.",
    "R4": "Question has no usable match in the notes (unmatched): material may be missing.",
    "R5": "Section strongly matched by one or more questions (high observed relevance).",
}


def build_plan(record: Dict[str, Any], reviewed: Optional[Set[str]] = None,
               min_weak_cluster: int = 2) -> List[Dict[str, Any]]:
    reviewed = reviewed or set()
    recs: List[Dict[str, Any]] = []
    sec_by_id = {s["section_id"]: s for s in record["sections"]}

    # R1 - clusters of weak questions on one section
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for q in record["questions"]:
        if q["best_tier"] == "Weak" and q["matches"]:
            clusters.setdefault(q["matches"][0]["section_id"], []).append(q)
    for sid, qs in clusters.items():
        if len(qs) >= min_weak_cluster and sid in sec_by_id:
            recs.append(_rec("R1", f"Review: {sec_by_id[sid]['heading']}", sid, sec_by_id[sid], 2 + len(qs),
                             [f"{q['display_id']} ({q['paper_label']}) best score {q['best_score']:.2f}" for q in qs],
                             reviewed))

    # R2 - observed repetition across papers
    comp = record.get("comparison")
    if comp:
        for s in comp["repeated_sections"]:
            sec = sec_by_id.get(s["section_id"])
            if sec:
                recs.append(_rec("R2", f"Repeated in {s['papers_hit']} of {s['papers_total']} papers: {s['heading']}",
                                 s["section_id"], sec, 3 + s["papers_hit"],
                                 [f"Hit by paper {pid[-6:]} (best {s['best_by_paper'][pid]:.2f})" for pid in s["paper_ids"]],
                                 reviewed))

    # R3 - syllabus gaps
    for t in record.get("topic_coverage", []):
        if t["status"] == "not_detected":
            recs.append(dict(rule_id="R3", rule=RULES["R3"], title=f"Syllabus topic not detected: {t['topic']}",
                             section_id=t.get("best_section_id"), priority=2,
                             evidence=[f"Highest question similarity {t['best_question_score']:.2f} (below weak threshold)",
                                       f"Closest notes section: {t.get('best_section_heading') or 'none'}"],
                             reviewed=bool(t.get("best_section_id") in reviewed), kind="topic"))

    # R4 - unmatched questions
    for q in record["questions"]:
        if q["best_tier"] == "Unmatched":
            recs.append(dict(rule_id="R4", rule=RULES["R4"], title=f"No notes match for {q['display_id']} ({q['paper_label']})",
                             section_id=None, priority=2,
                             evidence=[f"Best similarity {q['best_score']:.3f}; question: {q['text'][:140]}"],
                             reviewed=False, kind="question"))

    # R5 - strongly matched sections
    for s in record["sections"]:
        if s["top1_match_count"] >= 1 and s["max_similarity"] >= record["thresholds"]["strong"]:
            recs.append(_rec("R5", f"Strongly matched by a question: {s['heading']}", s["section_id"], s, 1 + s["top1_match_count"],
                             [f"Top match for {s['top1_match_count']} question(s); max similarity {s['max_similarity']:.2f}"],
                             reviewed))
    recs.sort(key=lambda r: (-r["priority"], r["rule_id"], r["title"]))
    return recs


def _rec(rule: str, title: str, sid: str, sec: Dict[str, Any], priority: int, evidence: List[str],
         reviewed: Set[str]) -> Dict[str, Any]:
    return dict(rule_id=rule, rule=RULES[rule], title=title, section_id=sid, priority=priority,
                evidence=evidence, reviewed=sid in reviewed, kind="section",
                source=f"{sec.get('doc_name', '')} - {sec.get('locator', sec.get('page_or_slide', ''))}")
