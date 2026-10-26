"""
Transparent subject suggestion by TF-IDF profile matching.

Each subject is represented by a text profile (name, description, syllabus topics and
excerpts of its existing documents). A new document is scored by cosine similarity to
each profile. Scores are similarities in [0, 1], NOT calibrated probabilities, and the
result states when evidence is insufficient or ambiguous. Filename tokens are scored
separately and can only raise a *conflict warning*; they never decide the ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.preprocessing import TextPreprocessor

METHOD = "TF-IDF cosine similarity between the document and each subject profile (not a calibrated probability)."


@dataclass
class SubjectProfile:
    subject_id: str
    name: str
    text: str


@dataclass
class SubjectSuggestion:
    subject_id: str
    name: str
    score: float
    evidence_terms: List[str] = field(default_factory=list)


@dataclass
class SuggestionResult:
    status: str                      # suggested | ambiguous | insufficient | no_subjects
    message: str
    suggestions: List[SubjectSuggestion] = field(default_factory=list)
    filename_hint: Optional[str] = None       # subject_id suggested by filename tokens alone
    filename_conflict: bool = False
    method: str = METHOD

    @property
    def top(self) -> Optional[SubjectSuggestion]:
        return self.suggestions[0] if self.suggestions else None


def _doc_text(text: str, title_lines: int = 5, body_chars: int = 20000) -> str:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    head = " ".join(lines[:title_lines])
    # headings/title are counted twice so they weigh more than a single body mention
    return f"{head} {head} {(text or '')[:body_chars]}"


def _build_matrix(profiles: List[SubjectProfile], pre: TextPreprocessor):
    texts = [pre.process(p.text).cleaned_text for p in profiles]
    if not any(t.strip() for t in texts):
        return None, None
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    try:
        return vec, vec.fit_transform(texts)
    except ValueError:
        return None, None


def suggest_subject(text: str, filename: str, profiles: List[SubjectProfile],
                    min_score: float = 0.05, ambiguity_margin: float = 0.25) -> SuggestionResult:
    if not profiles:
        return SuggestionResult("no_subjects", "No subjects exist yet. Create a subject first.")
    pre = TextPreprocessor()
    vec, pmat = _build_matrix(profiles, pre)
    cleaned = pre.process(_doc_text(text)).cleaned_text
    if vec is None or not cleaned.strip():
        return SuggestionResult(
            "insufficient",
            "Not enough text in the document or subject profiles to suggest a subject. Choose one manually.")
    dmat = vec.transform([cleaned])
    scores = (pmat @ dmat.T).toarray().ravel()  # rows are L2-normalised -> cosine
    names = vec.get_feature_names_out()
    ranked: List[SubjectSuggestion] = []
    for i in np.argsort(-scores):
        contrib = pmat[i].multiply(dmat).tocoo()
        order = np.argsort(-contrib.data)[:6]
        ranked.append(SubjectSuggestion(profiles[i].subject_id, profiles[i].name, float(scores[i]),
                                        [str(names[contrib.col[j]]) for j in order]))

    # Filename is a weak, separate signal used only to flag disagreement.
    stem = (filename or "").rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    fn_hint, fn_conflict = None, False
    fn_clean = pre.process(stem).cleaned_text
    if fn_clean.strip():
        fs = (pmat @ vec.transform([fn_clean]).T).toarray().ravel()
        if fs.max() > 0:
            fn_hint = profiles[int(np.argmax(fs))].subject_id

    best = ranked[0]
    if best.score < min_score:
        res = SuggestionResult("insufficient",
                               "No subject matches the document's content well enough to suggest one. "
                               "Choose a subject manually or create a new one.", ranked[:3])
    else:
        gap = (best.score - ranked[1].score) / best.score if len(ranked) > 1 and best.score > 0 else 1.0
        if gap < ambiguity_margin:
            res = SuggestionResult("ambiguous",
                                   f"Content is similar to several subjects (best: {best.name}). "
                                   "Please confirm the correct one.", ranked[:3])
        else:
            res = SuggestionResult("suggested", f"Content most resembles {best.name}.", ranked[:3])
    if fn_hint and res.top and fn_hint != res.top.subject_id:
        res.filename_conflict = True
        res.message += " Note: the filename points to a different subject than the content does; content was used."
    res.filename_hint = fn_hint
    return res
