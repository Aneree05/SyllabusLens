"""
Exports generated from a saved analysis record (works for fresh runs and for history).
Every export carries method, configuration, timestamp and source references.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd

from services.study_plan_service import build_plan

_FORMULA_PREFIX = ("=", "+", "-", "@", "\t", "\r")


def _safe(v: Any) -> Any:
    """Neutralise spreadsheet formula injection from uploaded text."""
    if isinstance(v, str) and v.startswith(_FORMULA_PREFIX):
        return "'" + v
    return v


def _csv(rows: List[Dict[str, Any]], preface: List[str]) -> str:
    df = pd.DataFrame([{k: _safe(v) for k, v in r.items()} for r in rows])
    buf = io.StringIO()
    for line in preface:
        buf.write("# " + line.replace("\n", " ") + "\n")
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _preface(record: Dict[str, Any], meta: Dict[str, Any], what: str) -> List[str]:
    s = record["settings"]
    return [
        f"SyllabusLens export: {what}",
        f"Subject: {record['subject']['name']} ({record['subject']['id']})",
        f"Analysis: {meta.get('id', 'unsaved')} created {meta.get('created_at', 'n/a')}",
        f"Method: {record['method']}; n-grams {s.get('ngram_range')}; stemming {s.get('use_stemming')}; "
        f"thresholds strong={record['thresholds']['strong']} moderate={record['thresholds']['moderate']} weak={record['thresholds']['weak']}",
        f"Corpus fingerprint: {record['corpus_fingerprint']}; cross-subject: {s.get('cross_subject', False)}",
        "Limitations: scores are lexical similarity, not proof an answer exists; thresholds are uncalibrated.",
    ]


def matches_csv(record: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    meta = meta or {}
    rows = []
    for q in record["questions"]:
        for m in q["matches"]:
            rows.append({"Paper": q["paper_label"], "Question": q["display_id"], "Marks": q["marks"],
                         "Question Text": q["text"], "Question Best Score": q["best_score"], "Question Tier": q["best_tier"],
                         "Rank": m["rank"], "Section ID": m["section_id"], "Document": m["doc_name"],
                         "Page/Slide": m["locator"], "Heading": m["heading"], "Score": m["score"], "Tier": m["tier"],
                         "Key Terms": ", ".join(m["key_terms"])})
    return _csv(rows, _preface(record, meta, "question-to-section matches"))


def coverage_csv(record: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    rows = [{"Section ID": s["section_id"], "Document": s["doc_name"], "Page/Slide": s["locator"],
             "Heading": s["heading"], "Max Similarity": s["max_similarity"], "Top-1 Questions": s["top1_match_count"],
             "Status": s["coverage_status"], "Matching Questions": ", ".join(s["matched_questions"])}
            for s in record["sections"]]
    return _csv(rows, _preface(record, meta or {}, "section coverage summary"))


def topics_csv(record: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    rows = [{"Topic": t["topic"], "Status": t["status"], "Best Question Score": t["best_question_score"],
             "Best Question": t.get("best_question_display") or "", "Closest Section": t.get("best_section_heading") or "",
             "Papers Detected In": len(t["papers_detected"])} for t in record.get("topic_coverage", [])]
    return _csv(rows, _preface(record, meta or {}, "syllabus topic coverage"))


def comparison_csv(record: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    comp = record.get("comparison")
    if not comp:
        raise ValueError("This analysis used a single paper; there is nothing to compare.")
    names = {p["document_id"]: p["title"] for p in comp["papers"]}
    rows = [{"Section ID": s["section_id"], "Document": s["doc_name"], "Heading": s["heading"],
             "Papers Hit": f"{s['papers_hit']} of {s['papers_total']}",
             "Hit By": "; ".join(names[p] for p in s["paper_ids"]),
             **{f"Best score - {names[p]}": round(v, 4) for p, v in s["best_by_paper"].items()}}
            for s in comp["section_frequency"]]
    return _csv(rows, _preface(record, meta or {}, "question-paper comparison") + [comp["basis"]])


def checklist_csv(record: Dict[str, Any], reviewed: Optional[set] = None, meta: Optional[Dict[str, Any]] = None) -> str:
    plan = build_plan(record, reviewed)
    rows = [{"Priority": r["priority"], "Rule": r["rule_id"], "Why (rule)": r["rule"], "Item": r["title"],
             "Source": r.get("source", ""), "Evidence": " | ".join(r["evidence"]),
             "Reviewed": "yes" if r["reviewed"] else "no"} for r in plan]
    return _csv(rows, _preface(record, meta or {}, "revision checklist (observed evidence, not a forecast)"))


def markdown_report(record: Dict[str, Any], meta: Optional[Dict[str, Any]] = None,
                    reviewed: Optional[set] = None) -> str:
    meta, m = meta or {}, record["metrics"]
    s = record["settings"]
    out = [f"# SyllabusLens report - {record['subject']['name']}", "",
           f"- Analysis: `{meta.get('id', 'unsaved')}` created {meta.get('created_at', 'n/a')}",
           f"- Method: {record['method']} (n-grams {s.get('ngram_range')}, stemming {s.get('use_stemming')})",
           f"- Thresholds: strong {record['thresholds']['strong']}, moderate {record['thresholds']['moderate']}, weak {record['thresholds']['weak']}",
           f"- Corpus fingerprint: `{record['corpus_fingerprint']}` | cross-subject: {s.get('cross_subject', False)}",
           f"- Question papers: {', '.join(p['title'] for p in record['papers'])}",
           f"- Lecture documents: {', '.join(d['filename'] for d in record['documents'])}", "",
           "## Summary",
           f"- Questions: {m['total_questions']}; sections indexed: {m['total_sections']}",
           f"- Strong {m['strong_count']}, moderate {m['moderate_count']}, weak {m['weak_count']}, unmatched {m['unmatched_count']}",
           f"- Question match rate (strong+moderate): {m['overall_coverage_rate']:.1f}%",
           f"- Sections with no matching question: {m['sections_uncovered_count']} of {m['total_sections']}", "",
           "## Question matches", "", "| Paper | Q | Score | Tier | Best section | Source |", "|---|---|---|---|---|---|"]
    for q in record["questions"]:
        t = q["matches"][0] if q["matches"] else None
        out.append(f"| {q['paper_label']} | {q['display_id']} | {q['best_score']:.3f} | {q['best_tier']} | "
                   f"{t['heading'] if t else '-'} | {(t['doc_name'] + ' ' + t['locator']) if t else '-'} |")
    plan = build_plan(record, reviewed)
    if plan:
        out += ["", "## Evidence-based revision suggestions", ""]
        for r in plan[:25]:
            out.append(f"- **{r['title']}** ({r['rule_id']}: {r['rule']}) - " + "; ".join(r["evidence"]))
    out += ["", "## Limitations", "", record["limitations"]]
    return "\n".join(out)


def practice_questions_csv(record: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    rows = []
    for q in record["questions"]:
        ao = q.get("answer_outline") or {}
        ns = q.get("numerical_solution") or {}
        rows.append({
            "Paper": q.get("paper_label", ""),
            "Question ID": q.get("display_id", q.get("question_id", "")),
            "Marks": q.get("marks", ""),
            "Category": q.get("category_name", q.get("category", "")),
            "Category Rationale": q.get("category_rationale", ""),
            "Question Text": q.get("text", ""),
            "Best Score": q.get("best_score", 0.0),
            "Best Tier": q.get("best_tier", ""),
            "Core Answer": ao.get("core_answer", ""),
            "Formula": ao.get("formula", "") or ns.get("formula", ""),
            "Is Solvable (Numerical)": ns.get("is_solvable", ""),
            "Final Answer (Numerical)": ns.get("final_answer", ""),
            "Missing Material": ao.get("insufficient_reason", "") or (", ".join(ns.get("missing_data", []))),
        })
    return _csv(rows, _preface(record, meta or {}, "categorized practice questions with grounded answers"))


def topic_notes_markdown(record: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> str:
    notes = record.get("topic_revision_notes") or []
    sub_name = record.get("subject", {}).get("name", "Subject")
    out = [
        f"# {sub_name} - Structured Revision Notes",
        f"Exported from SyllabusLens Analysis `{meta.get('id', 'unsaved') if meta else 'current'}`.",
        "Grounded in uploaded lecture materials and past papers.",
        "",
        "---",
        "",
    ]
    if not notes:
        out.append("No synthesized topic revision notes available in this record.")
        return "\n".join(out)

    for n in notes:
        out.append(f"## {n.get('topic', 'Topic')}")
        out.append(f"**Coverage Status**: `{n.get('status', 'covered').replace('_', ' ').title()}`")
        out.append(f"**Summary**: {n.get('summary', '')}\n")

        if n.get("definitions"):
            out.append("### Key Definitions")
            for item in n["definitions"]:
                out.append(f"- {item.get('text', '')}  \n  *Source: [{item.get('source', '')}]*")
            out.append("")

        if n.get("explanations"):
            out.append("### Explanations & Algorithms")
            for item in n["explanations"]:
                out.append(f"- {item.get('text', '')}  \n  *Source: [{item.get('source', '')}]*")
            out.append("")

        if n.get("formulas"):
            out.append("### Formulas & Governing Equations")
            for item in n["formulas"]:
                out.append(f"- `{item.get('text', '')}`  \n  *Source: [{item.get('source', '')}]*")
            out.append("")

        if n.get("pros_cons_limitations"):
            out.append("### Advantages, Limitations & Comparisons")
            for item in n["pros_cons_limitations"]:
                out.append(f"- {item.get('text', '')}  \n  *Source: [{item.get('source', '')}]*")
            out.append("")

        if n.get("missing_material"):
            out.append("### Missing Material Alerts")
            for m in n["missing_material"]:
                out.append(f"- ⚠️ {m}")
            out.append("")

        if n.get("sources"):
            out.append("### Source References")
            for s in n["sources"]:
                out.append(f"- {s}")
            out.append("")

        out.append("---\n")

    return "\n".join(out)
