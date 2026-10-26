"""
Plotly figures built ONLY from a saved analysis record (the dict produced by AnalysisRunner).
No chart reads live objects or invents values; each returns None when there is nothing to draw so
callers can show an empty state instead of an empty axis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

from components.theme import TIER_COLORS, TOKENS

TIERS = ["Strong", "Moderate", "Weak", "Unmatched"]
STATUS_COLORS = {"Covered": TIER_COLORS["Strong"], "Partially Covered": TIER_COLORS["Weak"],
                 "Uncovered in Exam": TIER_COLORS["Unmatched"]}
TOPIC_COLORS = {"detected": TIER_COLORS["Strong"], "not_detected": TIER_COLORS["Unmatched"], "not_assessable": "#9CA3AF"}
TOPIC_LABELS = {"detected": "Detected", "not_detected": "Not detected", "not_assessable": "Not assessable"}
_HEAT = [[0.0, "#F4F3FB"], [0.25, "#C7C4F2"], [0.6, "#6D63DB"], [1.0, "#2E2599"]]


def _layout(fig: go.Figure, height: int, margin: Optional[dict] = None, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=margin or dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#FFFFFF", font=dict(family=TOKENS["font_body"], size=12, color=TOKENS["ink"]),
                      hoverlabel=dict(bgcolor="#FFFFFF", font_size=12, bordercolor=TOKENS["border_strong"]),
                      legend=dict(orientation="h", y=-0.2, x=0), **kw)
    fig.update_xaxes(gridcolor="#ECEAF3", zerolinecolor="#D5D2E3")
    fig.update_yaxes(gridcolor="#ECEAF3", zerolinecolor="#D5D2E3")
    return fig


def _short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def paper_codes(record: Dict[str, Any]) -> Dict[str, str]:
    """Compact stable labels (P1, P2, ...) for papers, in the order they were analysed."""
    return {p["document_id"]: f"P{i + 1}" for i, p in enumerate(record["papers"])}


def _qlabel(q: Dict[str, Any], multi: bool, codes: Optional[Dict[str, str]] = None) -> str:
    if multi and codes:
        return f'{codes.get(q["paper_doc_id"], "P?")} {q["display_id"]}'
    return q["display_id"]


def similarity_heatmap(record: Dict[str, Any]) -> Optional[go.Figure]:
    vals = record["matrix"]["values"]
    if not vals or not vals[0]:
        return None
    multi = len(record["papers"]) > 1
    codes = paper_codes(record)
    qs, secs = record["questions"], record["sections"]
    ylab = [_qlabel(q, multi, codes) for q in qs]
    xlab = [f"S{i + 1}" for i in range(len(secs))]
    hover = [[f'<b>{_short(q["text"], 70)}</b><br>Section S{j + 1}: {_short(s["heading"], 60)}<br>{s["doc_name"]} · {s["locator"]}<br>Similarity: {vals[i][j]:.3f}'
              for j, s in enumerate(secs)] for i, q in enumerate(qs)]
    fig = go.Figure(go.Heatmap(z=vals, x=xlab, y=ylab, zmin=0, zmax=1, colorscale=_HEAT, text=hover,
                               hoverinfo="text", xgap=1, ygap=1,
                               colorbar=dict(orientation="h", title=dict(text="Cosine similarity", side="top"), thickness=10, len=0.7,
                                             x=0, xanchor="left", y=-0.02, yanchor="top", tickvals=[0, 0.25, 0.5, 0.75, 1])))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category", side="top", tickangle=0 if len(secs) <= 14 else -60)
    return _layout(fig, max(300, 26 * len(qs) + 170), margin=dict(l=8, r=8, t=28, b=64))


def tier_distribution(record: Dict[str, Any]) -> Optional[go.Figure]:
    qs = record["questions"]
    if not qs:
        return None
    papers = record["papers"]
    fig = go.Figure()
    for tier in TIERS:
        counts = [sum(1 for q in qs if q["paper_doc_id"] == p["document_id"] and q["best_tier"] == tier) for p in papers]
        fig.add_bar(name=tier, x=[_short(p["title"], 26) for p in papers], y=counts, marker_color=TIER_COLORS[tier],
                    hovertemplate="%{x}<br>" + tier + ": %{y} questions<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Questions", dtick=1 if len(qs) < 12 else None, rangemode="tozero")
    return _layout(fig, 320)


def score_histogram(record: Dict[str, Any]) -> Optional[go.Figure]:
    scores = [q["best_score"] for q in record["questions"]]
    if not scores:
        return None
    thr = record["thresholds"]
    fig = go.Figure(go.Histogram(x=scores, xbins=dict(start=0, end=1.0, size=0.05), marker_color="#6D63DB",
                                 marker_line=dict(color="#FFFFFF", width=1),
                                 hovertemplate="Best similarity %{x}<br>%{y} questions<extra></extra>"))
    for name, key in (("weak", "weak"), ("moderate", "moderate"), ("strong", "strong")):
        fig.add_vline(x=thr[key], line=dict(color="#6B7089", dash="dot", width=1),
                      annotation_text=name, annotation_position="top", annotation_font_size=11)
    fig.update_xaxes(title="Best similarity per question", range=[0, max(1.0, max(scores) + 0.05)])
    fig.update_yaxes(title="Questions", rangemode="tozero")
    return _layout(fig, 300, bargap=0.05)


def section_coverage_bars(record: Dict[str, Any], limit: int = 25) -> Optional[go.Figure]:
    secs = record["sections"]
    if not secs:
        return None
    order = sorted(range(len(secs)), key=lambda i: secs[i]["max_similarity"])[-limit:]
    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        idx = [i for i in order if secs[i]["coverage_status"] == status]
        fig.add_bar(name=status, orientation="h", y=[f"S{i + 1} {_short(secs[i]['heading'], 34)}" for i in idx],
                    x=[secs[i]["max_similarity"] for i in idx], marker_color=color,
                    customdata=[[secs[i]["doc_name"], secs[i]["locator"], secs[i]["top1_match_count"]] for i in idx],
                    hovertemplate="<b>%{y}</b><br>%{customdata[0]} · %{customdata[1]}<br>Max similarity: %{x:.3f}"
                                  "<br>Top-ranked for %{customdata[2]} question(s)<extra></extra>")
    fig.update_layout(barmode="stack", yaxis=dict(categoryorder="array", categoryarray=[
        f"S{i + 1} {_short(secs[i]['heading'], 34)}" for i in order]))
    fig.update_xaxes(title="Highest similarity to any question", range=[0, 1])
    return _layout(fig, max(260, 26 * len(order) + 110))


def topic_coverage_bars(record: Dict[str, Any]) -> Optional[go.Figure]:
    topics = record.get("topic_coverage") or []
    if not topics:
        return None
    fig = go.Figure()
    for status, label in TOPIC_LABELS.items():
        rows = [t for t in topics if t["status"] == status]
        fig.add_bar(name=label, orientation="h", y=[_short(t["topic"], 40) for t in rows],
                    x=[t["best_question_score"] for t in rows], marker_color=TOPIC_COLORS[status],
                    customdata=[[t.get("best_section_heading") or "-", len(t["papers_detected"])] for t in rows],
                    hovertemplate="<b>%{y}</b><br>Best question similarity: %{x:.3f}<br>Closest notes section: %{customdata[0]}"
                                  "<br>Papers where detected: %{customdata[1]}<extra></extra>")
    fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title="Highest similarity between topic and any question", range=[0, 1])
    return _layout(fig, max(240, 30 * len(topics) + 110))


def paper_tier_comparison(record: Dict[str, Any]) -> Optional[go.Figure]:
    comp = record.get("comparison")
    if not comp:
        return None
    fig = go.Figure()
    names = [_short(p["title"], 24) for p in comp["papers"]]
    for tier in TIERS:
        fig.add_bar(name=tier, x=names, y=[p["tiers"][tier] for p in comp["papers"]], marker_color=TIER_COLORS[tier],
                    hovertemplate="%{x}<br>" + tier + ": %{y}<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Questions", rangemode="tozero", dtick=1)
    return _layout(fig, 320)


def section_frequency_bars(record: Dict[str, Any]) -> Optional[go.Figure]:
    comp = record.get("comparison")
    if not comp:
        return None
    rows = [s for s in comp["section_frequency"] if s["papers_hit"] > 0]
    if not rows:
        return None
    rows.sort(key=lambda s: (s["papers_hit"], s["heading"]))
    total = rows[0]["papers_total"]
    fig = go.Figure(go.Bar(orientation="h", y=[_short(s["heading"], 40) for s in rows], x=[s["papers_hit"] for s in rows],
                           marker_color=["#4338CA" if s["papers_hit"] >= 2 else "#B9B5EA" for s in rows],
                           customdata=[[s["doc_name"], s["locator"], total] for s in rows],
                           hovertemplate="<b>%{y}</b><br>%{customdata[0]} · %{customdata[1]}<br>Hit by %{x} of %{customdata[2]} papers<extra></extra>"))
    fig.update_xaxes(title="Number of selected papers with a matching question", dtick=1, range=[0, total])
    return _layout(fig, max(240, 26 * len(rows) + 100), showlegend=False)


def question_section_sankey(record: Dict[str, Any], min_score: float = 0.12) -> Optional[go.Figure]:
    """Questions -> sections. Nodes keyed by unique IDs; a link exists only for a recorded top-ranked match."""
    multi = len(record["papers"]) > 1
    links = [(q, m) for q in record["questions"] for m in q["matches"][:2] if m["score"] >= min_score]
    if not links:
        return None
    codes = paper_codes(record)
    q_ids = list(dict.fromkeys(q["question_id"] for q, _ in links))
    s_ids = list(dict.fromkeys(m["section_id"] for _, m in links))
    qinfo = {q["question_id"]: q for q, _ in links}
    sinfo = {m["section_id"]: m for _, m in links}
    idx = {("q", i): n for n, i in enumerate(q_ids)}
    idx.update({("s", i): len(q_ids) + n for n, i in enumerate(s_ids)})
    labels = [_qlabel(qinfo[i], multi, codes) for i in q_ids] + [_short(sinfo[i]["heading"], 34) for i in s_ids]
    colors = [TIER_COLORS[qinfo[i]["best_tier"]] for i in q_ids] + ["#4338CA"] * len(s_ids)
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color=colors, pad=14, thickness=14, line=dict(color="#FFFFFF", width=0.5)),
        link=dict(source=[idx[("q", q["question_id"])] for q, _ in links], target=[idx[("s", m["section_id"])] for _, m in links],
                  value=[max(m["score"], 0.01) for _, m in links], color=["rgba(67,56,202,0.25)"] * len(links),
                  customdata=[f'{q["display_id"]} → {m["heading"]}: {m["score"]:.3f} (rank {m["rank"]})' for q, m in links],
                  hovertemplate="%{customdata}<extra></extra>")))
    return _layout(fig, max(320, 30 * max(len(q_ids), len(s_ids)) + 80), margin=dict(l=8, r=8, t=8, b=8))
