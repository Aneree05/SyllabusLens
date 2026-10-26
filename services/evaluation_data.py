"""
Ground-truth loading, reference resolution and retrieval evaluation against saved/new analyses.

Ground truth references sections by HEADING FRAGMENT within a named notes document, never by
internal ID, so it survives ID-scheme changes. Every unresolved reference is reported; an
evaluation with zero resolvable questions raises instead of returning empty metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from services.analysis_runner import AnalysisSettings
from services.errors import DocumentValidationError
from services.evaluation_service import EvaluationService

SAMPLE_GROUND_TRUTH = Path(__file__).resolve().parent.parent / "sample_data" / "ground_truth_relevance.json"


@dataclass
class GroundTruth:
    description: str
    notes_document: str
    paper_document: str
    judgments: Dict[str, List[str]]           # display question id -> heading fragments
    binary_coverage: Dict[str, bool]
    is_demo: bool = False
    version: str = ""


@dataclass
class ResolvedTruth:
    relevant: Dict[str, List[str]] = field(default_factory=dict)   # scoped question id -> section ids
    binary: Dict[str, bool] = field(default_factory=dict)          # scoped question id -> covered?
    problems: List[str] = field(default_factory=list)


def load_ground_truth(source: Union[str, Path, bytes, Dict[str, Any]]) -> GroundTruth:
    if isinstance(source, dict):
        raw = source
    else:
        try:
            text = source.decode("utf-8") if isinstance(source, bytes) else Path(source).read_text(encoding="utf-8")
            raw = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentValidationError(f"Ground-truth file could not be read as JSON: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("judgments"), dict):
        raise DocumentValidationError("Ground truth must be a JSON object with a 'judgments' mapping.")
    for qid, refs in raw["judgments"].items():
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            raise DocumentValidationError(f"Judgments for '{qid}' must be a list of heading fragments.")
        if any(r.endswith(tuple(f"_Sec_{i}" for i in range(1, 100))) for r in refs):
            raise DocumentValidationError(
                "This ground-truth file uses obsolete section IDs (version 1). Use version 2, which "
                "references sections by heading fragment such as 'Unit 2:'.")
    for key in ("notes_document", "paper_document"):
        if not raw.get(key):
            raise DocumentValidationError(f"Ground truth must name '{key}' (a filename in the library).")
    return GroundTruth(
        description=str(raw.get("description", "")), notes_document=raw["notes_document"],
        paper_document=raw["paper_document"], judgments=raw["judgments"],
        binary_coverage={k: bool(v) for k, v in (raw.get("binary_coverage") or {}).items()},
        is_demo=bool(raw.get("is_demo_data", False)), version=str(raw.get("dataset_version", "")))


def resolve(gt: GroundTruth, record: Dict[str, Any]) -> ResolvedTruth:
    """Map the ground truth onto a record's real question and section IDs."""
    out = ResolvedTruth()
    paper_ids = {p["document_id"] for p in record["papers"] if p["filename"] == gt.paper_document}
    if not paper_ids:
        out.problems.append(f"Paper '{gt.paper_document}' is not part of this analysis.")
        return out
    q_by_display = {q["display_id"]: q for q in record["questions"] if q["paper_doc_id"] in paper_ids}
    notes_secs = [s for s in record["sections"] if s["doc_name"] == gt.notes_document]
    if not notes_secs:
        out.problems.append(f"Notes '{gt.notes_document}' are not part of this analysis.")
        return out
    for disp, fragments in gt.judgments.items():
        q = q_by_display.get(disp)
        if q is None:
            out.problems.append(f"Question '{disp}' was not found in the analyzed paper.")
            continue
        ids: List[str] = []
        for frag in fragments:
            hits = [s["section_id"] for s in notes_secs if frag in s["heading"]]
            if len(hits) != 1:
                out.problems.append(f"Reference '{frag}' for {disp} matched {len(hits)} sections (needs exactly 1).")
            else:
                ids.append(hits[0])
        out.relevant[q["question_id"]] = ids
        if disp in gt.binary_coverage:
            out.binary[q["question_id"]] = gt.binary_coverage[disp]
    return out


def evaluate_against_truth(svc, subject_id: str, paper_id: str, note_ids: Sequence[str], gt: GroundTruth,
                           threshold: float = 0.20) -> Dict[str, Any]:
    """Run TF-IDF and Bag-of-Words on identical inputs (not saved to history) and score both."""
    reports, resolved = {}, None
    for label, vec in (("TF-IDF", "tfidf"), ("Bag-of-Words", "bow")):
        out = svc.runner.run(subject_id, [paper_id], list(note_ids), save=False,
                             settings=AnalysisSettings(vectorizer=vec, top_k=5))
        resolved = resolved or resolve(gt, out.record)
        rankings = {q["question_id"]: [m["section_id"] for m in q["matches"]] for q in out.record["questions"]}
        rep = EvaluationService().evaluate_retrieval(label, rankings, resolved.relevant)
        if rep.num_queries == 0:
            raise DocumentValidationError(
                "No ground-truth question could be matched to this analysis, so no metrics were computed. "
                + " ".join(resolved.problems[:3]))
        scores = {q["question_id"]: q["best_score"] for q in out.record["questions"]}
        reports[label] = dict(report=rep, coverage=EvaluationService().evaluate_coverage_classification(
            scores, resolved.binary, threshold))
    return dict(reports=reports, problems=resolved.problems if resolved else [], threshold=threshold,
                is_demo=gt.is_demo)
