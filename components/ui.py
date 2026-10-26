"""
Reusable UI primitives for SyllabusLens.

Rules enforced here:
  * every dynamic string is passed through esc(), so filenames/questions can never inject HTML;
  * HTML is emitted as single-line strings (no indentation / blank lines) so Streamlit's Markdown
    parser never turns part of a block into a code block;
  * components only render what they are given - no data access, no NLP.
"""
from __future__ import annotations

import html as _html
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import streamlit as st

import config
from components.theme import TIER_SOFT  # noqa: F401  (re-exported for charts)


# ------------------------------------------------------------------ text helpers
def esc(value: Any) -> str:
    """HTML-escape, collapse whitespace to one line, and neutralise Streamlit's $-math syntax."""
    text = " ".join(str(value if value is not None else "").split())
    return _html.escape(text, quote=True).replace("$", "&#36;")


def fmt_dt(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M UTC")
    except (ValueError, TypeError):
        return iso or ""


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def render(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ------------------------------------------------------------------ badges
_TONES = {"neutral", "primary", "success", "info", "warning", "danger"}


def badge(text: str, tone: str = "neutral", title: str = "") -> str:
    tone = tone if tone in _TONES else "neutral"
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="sl-badge sl-{tone}"{t}>{esc(text)}</span>'


_TIER_TONE = {"Strong": "success", "Moderate": "info", "Weak": "warning", "Unmatched": "danger"}
TIER_HELP = {
    "Strong": "Lexical similarity at or above the strong threshold (a configurable heuristic).",
    "Moderate": "Similarity between the moderate and strong thresholds (heuristic).",
    "Weak": "Similarity between the weak and moderate thresholds; treat as a lead, not a match.",
    "Unmatched": "Below the weak threshold: no section shares enough vocabulary with this question.",
}


def tier_badge(tier: str) -> str:
    return badge(tier, _TIER_TONE.get(tier, "neutral"), TIER_HELP.get(tier, ""))


_STATUS = {"processed": ("Processed", "success"), "failed": ("Extraction failed", "danger"),
           "no_text": ("No text found", "warning")}


def status_badge(status: str) -> str:
    label, tone = _STATUS.get(status, (status or "Unknown", "neutral"))
    return badge(label, tone)


def kind_label(kind: str) -> str:
    return config.DOCUMENT_KINDS.get(kind, (kind,))[0]


def kind_badge(kind: str) -> str:
    return badge(kind_label(kind), "primary")


# ------------------------------------------------------------------ layout blocks
def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    render(f'<header class="sl-page-head"><h1>{esc(title)}</h1>{sub}</header>')


def section_heading(title: str, caption: str = "") -> None:
    st.markdown(f"### {title.replace('$', chr(92) + '$')}")
    if caption:
        render(f'<p class="sl-caption">{esc(caption)}</p>')


def scope_bar(subject_name: str, documents: int, papers: int, *, course_code: str = "",
              cross_subject_names: Sequence[str] = ()) -> None:
    """Persistent statement of exactly which corpus the page operates on."""
    code = f' <span class="sl-scope-item">{esc(course_code)}</span>' if course_code else ""
    if cross_subject_names:
        extra = ", ".join(esc(n) for n in cross_subject_names)
        render(f'<div class="sl-scope sl-cross" role="status"><span>{badge("Cross-subject", "warning")}</span>'
               f'<strong>{esc(subject_name)}</strong><span class="sl-scope-item">+ {extra}</span></div>')
        return
    render(f'<div class="sl-scope" role="status"><span>{badge("Subject scope", "primary", "Analyses use only documents in this subject.")}</span>'
           f'<strong>{esc(subject_name)}</strong>{code}<span class="sl-scope-item">{documents} lecture/reference {"doc" if documents == 1 else "docs"}</span>'
           f'<span class="sl-scope-item">{papers} question {"paper" if papers == 1 else "papers"}</span><span class="sl-scope-item">Other subjects are excluded</span></div>')


Metric = Union[Tuple[str, Any], Tuple[str, Any, str], Tuple[str, Any, str, str]]


def metric_grid(items: Iterable[Metric]) -> None:
    cells = []
    for it in items:
        label, value = it[0], it[1]
        hint = it[2] if len(it) > 2 else ""
        tone = it[3] if len(it) > 3 else ""
        cls = f" sl-tone-{tone}" if tone else ""
        h = f'<div class="sl-metric-hint">{esc(hint)}</div>' if hint else ""
        cells.append(f'<div class="sl-metric{cls}" title="{esc(hint)}"><div class="sl-metric-label">{esc(label)}</div>'
                     f'<div class="sl-metric-value">{esc(value)}</div>{h}</div>')
    render(f'<div class="sl-metrics">{"".join(cells)}</div>')


def notice(kind: str, title: str, body: str = "") -> None:
    """kind: info | success | warning | error"""
    kind = kind if kind in {"info", "success", "warning", "error"} else "info"
    role = "alert" if kind == "error" else "status"
    b = f"<span>{esc(body)}</span>" if body else ""
    render(f'<div class="sl-notice sl-{kind}" role="{role}"><strong>{esc(title)}</strong>{b}</div>')


def empty_state(title: str, body: str) -> None:
    render(f'<div class="sl-empty"><h3>{esc(title)}</h3><p>{esc(body)}</p></div>')


@contextmanager
def card(key: str):
    """A bordered white surface. `key` must be unique on the page (it becomes a stable CSS hook)."""
    with st.container(key=f"sl-card-{key}"):
        yield


def chart_card(key: str, title: str, caption: str):
    """Container with a title and a plain-language explanation of what the values mean."""
    c = card(key)
    return _ChartCard(c, title, caption)


class _ChartCard:
    def __init__(self, ctx, title, caption):
        self.ctx, self.title, self.caption = ctx, title, caption

    def __enter__(self):
        self.ctx.__enter__()
        st.markdown(f"**{self.title}**".replace("$", "\\$"))
        render(f'<p class="sl-caption">{esc(self.caption)}</p>')
        return self

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc)


def fmt_marks(marks: Any) -> str:
    """'10' -> '10 marks'; '10 marks' / '[10m]'-style values are left alone."""
    m = str(marks or "").strip()
    return m if any(c.isalpha() for c in m) else f"{m} marks"


# ------------------------------------------------------------------ domain cards
def document_card_html(doc, subject_name: str = "", show_subject: bool = False) -> str:
    tags = "".join(badge(t, "neutral") for t in doc.tags)
    subj = badge(subject_name, "info") if show_subject and subject_name else ""
    meta = [f"{esc(doc.filename)}", f"{human_size(doc.size_bytes)}", f".{esc(doc.file_ext)}"]
    if doc.page_count:
        meta.append(f"{doc.page_count} {'slides' if doc.file_ext == 'pptx' else 'pages'}")
    meta.append(f"Uploaded {esc(fmt_dt(doc.uploaded_at))}")
    extra = []
    if doc.academic_year: extra.append(f"Year {esc(doc.academic_year)}")
    if doc.exam_session: extra.append(esc(doc.exam_session))
    if doc.semester: extra.append(f"Sem {esc(doc.semester)}")
    detail = f'<div class="sl-card-meta" style="color:var(--sl-warning)">{esc(doc.status_detail)}</div>' if doc.status_detail else ""
    return (f'<div class="sl-card"><div class="sl-card-title">{esc(doc.title)}</div>'
            f'<div>{kind_badge(doc.doc_kind)} {status_badge(doc.status)} {subj} {tags}</div>'
            f'<div class="sl-card-meta">{" ".join(f"<span>{m}</span>" for m in meta + extra)}</div>{detail}</div>')


_CAT_TONE = {
    "short_answer": "info",
    "long_answer": "primary",
    "comparison": "success",
    "derivation": "warning",
    "numerical": "primary",
    "diagram": "info",
}

_CAT_NAME = {
    "short_answer": "Short Answer",
    "long_answer": "Theory",
    "comparison": "Comparison",
    "derivation": "Derivation",
    "numerical": "Numerical",
    "diagram": "Diagram",
}


def category_badge(category: str, title: str = "") -> str:
    name = _CAT_NAME.get(category, category.replace("_", " ").title())
    tone = _CAT_TONE.get(category, "neutral")
    return badge(name, tone, title or f"Pedagogical category: {name}")


def question_card_html(q: Dict[str, Any], *, show_paper: bool = False) -> str:
    top = q["matches"][0] if q["matches"] else None
    if top and q["best_tier"] == "Unmatched":
        src = (f'No match. Closest section, shown for reference only (below the weak threshold): <strong>{esc(top["heading"])}</strong> '
               f'&mdash; {esc(top["doc_name"])}, {esc(top["locator"])}')
    elif top:
        src = f'Best section: <strong>{esc(top["heading"])}</strong> &mdash; {esc(top["doc_name"])}, {esc(top["locator"])}'
    else:
        src = "No section reached the weak-match threshold."
    paper = f'<span>{esc(q["paper_label"])}</span>' if show_paper and q.get("paper_label") else ""
    marks = f'<span>{esc(fmt_marks(q["marks"]))}</span>' if q.get("marks") else ""
    cat = category_badge(q.get("category", ""), q.get("category_rationale", "")) if q.get("category") else ""
    return (f'<div class="sl-qcard sl-q-{esc(q["best_tier"])}"><div class="sl-qhead"><strong>{esc(q["display_id"])}</strong>'
            f'{cat}{tier_badge(q["best_tier"])}<span>similarity {q["best_score"]:.3f}</span>{marks}{paper}</div>'
            f'<div class="sl-qtext">{esc(q["text"])}</div><div class="sl-qsrc">{src}</div></div>')
