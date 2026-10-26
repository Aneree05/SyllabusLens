"""Dashboard facts and the 'what next?' decision for a subject. Pure reads; no UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import config


@dataclass
class Readiness:
    notes: int
    papers: int
    syllabus_docs: int
    failures: int
    sections: int
    topics: int
    analyses: int
    latest: Optional[Dict[str, Any]]          # newest saved analysis (summary row), if any
    latest_inputs_changed: bool


@dataclass
class NextAction:
    key: str
    title: str
    body: str
    target: str            # page key


def readiness(svc, subject) -> Readiness:
    stats = svc.library.stats(subject.id)
    kinds = stats["by_kind"]
    eligible = [d for d in svc.library.list_documents(subject.id, status="processed", limit=1000)
                if not d.is_question_paper and config.DOCUMENT_KINDS[d.doc_kind][2]]
    items = svc.history.list(subject.id, limit=1)
    latest = items[0] if items else None
    changed = bool(latest) and svc.history.input_status(latest["id"])["status"] != "current"
    return Readiness(notes=len(eligible), papers=kinds.get(config.QUESTION_PAPER_KIND, 0), syllabus_docs=kinds.get("syllabus", 0),
                     failures=stats["failures"], sections=stats["sections"], topics=len(subject.topics),
                     analyses=svc.subjects.counts(subject.id)["analyses"], latest=latest, latest_inputs_changed=changed)


def next_action(r: Readiness, subject_name: str) -> NextAction:
    if r.notes == 0:
        extra = f" {r.failures} file(s) could not be read; check the library for the reason." if r.failures else ""
        return NextAction("upload_notes", "Upload lecture notes or slides",
                          f"{subject_name} has no readable lecture material yet, so there is nothing to match questions against.{extra}", "library")
    if r.papers == 0:
        return NextAction("add_paper", "Add a question paper", "Upload a previous-year paper so its questions can be matched to your notes.", "library")
    if r.analyses == 0:
        return NextAction("run", "Run your first analysis", "You have notes and a paper. Match the questions to the lecture sections.", "analysis")
    if r.latest_inputs_changed:
        return NextAction("rerun", "Re-run the analysis", "Documents changed since your latest analysis. The saved results still show the old inputs.", "analysis")
    if r.topics == 0:
        return NextAction("topics", "Add syllabus topics", "With topics listed, SyllabusLens can show which ones your papers never touch.", "subjects")
    return NextAction("plan", "Review your study plan", "See which sections the selected papers point to and tick them off as you revise.", "planner")


def last_activity(svc, subject_id: str) -> str:
    docs = svc.library.list_documents(subject_id, sort="uploaded_desc", limit=1)
    runs = svc.history.list(subject_id, limit=1)
    return max([d.uploaded_at for d in docs] + [a["created_at"] for a in runs] or [""])


def subjects_by_recent_activity(svc) -> List[Tuple[Any, str]]:
    rows = [(s, last_activity(svc, s.id)) for s in svc.subjects.list()]
    return sorted(rows, key=lambda r: r[1], reverse=True)
