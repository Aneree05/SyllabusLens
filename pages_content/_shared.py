"""Helpers shared by every page that displays a saved analysis record."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from components import state, ui
from services import record_export as rx
from services.errors import NotFoundError

Loaded = Tuple[Dict[str, Any], Dict[str, Any]]    # (meta, record)


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:40] or "subject"


def load_for_subject(svc, analysis_id: Optional[str], subject_id: str) -> Optional[Loaded]:
    """Load a saved analysis ONLY if it belongs to the given subject (stale/foreign ids yield None)."""
    if not analysis_id:
        return None
    try:
        meta = svc.history.get(analysis_id)
    except NotFoundError:
        return None
    if meta["subject_id"] != subject_id:
        return None
    return meta, meta["record"]


def analysis_label(a: Dict[str, Any]) -> str:
    return f'{a["name"]} - {ui.fmt_dt(a["created_at"])}'


def analysis_picker(svc, subject, *, key: str, need_comparison: bool = False, empty_cta=None) -> Optional[Loaded]:
    """Select one of the ACTIVE subject's saved analyses. Defaults to the latest run in this session."""
    items = svc.history.list(subject.id, limit=100)
    if need_comparison:
        items = [a for a in items if len(a["inputs"].get("papers", [])) > 1]
    if not items:
        msg = ("No saved analysis of two or more papers for this subject yet." if need_comparison
               else "No saved analysis for this subject yet. Run one on the Question Analysis page.")
        ui.empty_state("Nothing to show yet", msg)
        if empty_cta:
            empty_cta()
        return None
    ids = [a["id"] for a in items]
    preferred = state.get_last_analysis_id(subject.id)
    index = ids.index(preferred) if preferred in ids else 0
    chosen = st.selectbox("Analysis", ids, index=index, key=f"{key}_{subject.id}",
                          format_func=lambda i: analysis_label(next(a for a in items if a["id"] == i)),
                          help="Saved snapshots for the active subject only.")
    return load_for_subject(svc, chosen, subject.id)


def scope_statement(meta: Dict[str, Any], record: Dict[str, Any]) -> None:
    """State exactly which subject, documents and settings produced these results."""
    s = record["settings"]
    cross = s.get("cross_subject", False)
    docs = ", ".join(d["filename"] for d in record["documents"])
    papers = ", ".join(p["title"] for p in record["papers"])
    body = (f"Question paper(s): {papers}. Lecture material ({len(record['documents'])} document(s), "
            f"{record['metrics']['total_sections']} sections): {docs}. Method: {record['method']}, n-grams {tuple(s.get('ngram_range', ()))}, "
            f"stemming {'on' if s.get('use_stemming') else 'off'}. Run {ui.fmt_dt(meta['created_at'])}. Corpus version {record['corpus_fingerprint'][:10]}.")
    if cross:
        ui.notice("warning", f"Cross-subject analysis: {len(s.get('subjects_in_corpus', []))} subjects were searched", body)
    else:
        ui.notice("info", f"Scope: {record['subject']['name']} only. Other subjects were not searched.", body)


def staleness_notice(svc, meta: Dict[str, Any]) -> None:
    status = svc.history.input_status(meta["id"])
    if status["status"] == "inputs_changed":
        ui.notice("warning", "Source documents changed after this analysis",
                  "This is a frozen snapshot and still shows the original results. Changes since: " + " ".join(status["changes"]) +
                  " Run a new analysis to reflect the current library.")


def similarity_caption() -> None:
    ui.render('<p class="sl-caption"><strong>How to read similarity.</strong> Scores are cosine similarity between word-frequency vectors (0 = no shared '
              'vocabulary, 1 = identical wording). They measure shared terminology, not understanding: a high score does not prove a section answers the '
              'question, and a low score does not prove the topic is absent. Tier thresholds are adjustable heuristics, not validated confidence levels.</p>')


def export_buttons(svc, meta: Dict[str, Any], record: Dict[str, Any], prefix: str, *, reviewed: Optional[set] = None) -> None:
    """Generate CSV/Markdown on demand. Buttons only offer the file; they do not claim a download happened."""
    m = {"id": meta["id"], "created_at": meta["created_at"]}
    base = f"syllabuslens_{slug(record['subject']['name'])}_{meta['id']}"
    items: List[Tuple[str, str, str, str]] = [
        ("Question matches (CSV)", rx.matches_csv(record, m), f"{base}_matches.csv", "text/csv"),
        ("Section coverage (CSV)", rx.coverage_csv(record, m), f"{base}_coverage.csv", "text/csv"),
        ("Report (Markdown)", rx.markdown_report(record, m, reviewed), f"{base}_report.md", "text/markdown"),
    ]
    if record.get("topic_coverage"):
        items.append(("Syllabus topics (CSV)", rx.topics_csv(record, m), f"{base}_topics.csv", "text/csv"))
    if record.get("topic_revision_notes"):
        items.append(("Topic notes (Markdown)", rx.topic_notes_markdown(record, m), f"{base}_topic_notes.md", "text/markdown"))
    if hasattr(rx, "practice_questions_csv"):
        items.append(("Practice questions (CSV)", rx.practice_questions_csv(record, m), f"{base}_practice_questions.csv", "text/csv"))
    if record.get("comparison"):
        items.append(("Paper comparison (CSV)", rx.comparison_csv(record, m), f"{base}_comparison.csv", "text/csv"))
    cols = st.columns(min(len(items), 3))
    for i, (label, data, name, mime) in enumerate(items):
        cols[i % len(cols)].download_button(label, data, file_name=name, mime=mime, key=f"{prefix}_dl_{i}", width="stretch")
