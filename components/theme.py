"""
Design tokens and the single global stylesheet for SyllabusLens (light theme only).

Everything visual is derived from TOKENS so there is exactly one place to change a colour,
a radius or a type size. Custom components in components/ui.py use the CSS classes defined
here (prefixed `sl-`). Streamlit's own widgets are styled through documented `data-testid`
hooks and `st-key-*` container classes, never through generated class names.
"""
from __future__ import annotations

from string import Template

import streamlit as st

TOKENS = {
    # surfaces and ink
    "bg": "#F6F5F2", "surface": "#FFFFFF", "surface_alt": "#F0EFF8", "border": "#E2E0EA", "border_strong": "#C9C6D8",
    "ink": "#14183A", "ink_muted": "#4B5068", "ink_faint": "#5C6180",
    # brand
    "primary": "#4338CA", "primary_hover": "#3730A3", "primary_soft": "#E8E7FA",
    "accent": "#2563EB", "violet": "#7C3AED",
    # semantic
    "success": "#0F766E", "success_soft": "#DDF3EF",
    "info": "#1D4ED8", "info_soft": "#E3ECFD",
    "warning": "#B45309", "warning_soft": "#FBEBD3",
    "danger": "#BE123C", "danger_soft": "#FADDE4",
    # geometry
    "radius": "10px", "radius_sm": "6px", "shadow": "0 1px 2px rgba(20,24,58,.06), 0 4px 14px rgba(20,24,58,.05)",
    # type
    "font_display": "'Newsreader', Georgia, 'Times New Roman', serif",
    "font_body": "'Hanken Grotesk', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
}

# Match-tier palette shared by badges and every chart (single source of truth).
TIER_COLORS = {"Strong": "#0F766E", "Moderate": "#2563EB", "Weak": "#D97706", "Unmatched": "#E11D48"}
TIER_SOFT = {"Strong": "#DDF3EF", "Moderate": "#E3ECFD", "Weak": "#FBEBD3", "Unmatched": "#FADDE4"}
TIER_INK = {"Strong": "#0B5F58", "Moderate": "#1D4ED8", "Weak": "#8A4208", "Unmatched": "#9F1239"}

_CSS = Template("""
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
  --sl-bg:$bg; --sl-surface:$surface; --sl-surface-alt:$surface_alt; --sl-border:$border; --sl-border-strong:$border_strong;
  --sl-ink:$ink; --sl-muted:$ink_muted; --sl-faint:$ink_faint;
  --sl-primary:$primary; --sl-primary-hover:$primary_hover; --sl-primary-soft:$primary_soft; --sl-accent:$accent; --sl-violet:$violet;
  --sl-success:$success; --sl-success-soft:$success_soft; --sl-info:$info; --sl-info-soft:$info_soft;
  --sl-warning:$warning; --sl-warning-soft:$warning_soft; --sl-danger:$danger; --sl-danger-soft:$danger_soft;
  --sl-radius:$radius; --sl-radius-sm:$radius_sm; --sl-shadow:$shadow;
  --sl-display:$font_display; --sl-body:$font_body;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] { font-family: var(--sl-body); color: var(--sl-ink); }
[data-testid="stAppViewContainer"] { background: var(--sl-bg); }
[data-testid="stHeader"] { background: var(--sl-bg); }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1180px; padding: 1.6rem 2rem 4rem; }
h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] .sl-page-head h1 { font-family: var(--sl-display); color: var(--sl-ink); letter-spacing: -0.01em; font-weight: 600; }
[data-testid="stMarkdownContainer"] h2 { font-size: 1.45rem; margin-top: 1.4rem; }
[data-testid="stMarkdownContainer"] h3 { font-size: 1.2rem; padding-top: .6rem; }
p, li, label, .stMarkdown { line-height: 1.55; }
a { color: var(--sl-primary); }

/* keyboard focus is always visible */
:focus-visible { outline: 2px solid var(--sl-accent) !important; outline-offset: 2px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--sl-surface); border-right: 1px solid var(--sl-border); }
.sl-brand { font-family: var(--sl-display); font-size: 1.35rem; font-weight: 600; color: var(--sl-ink); margin: .2rem 0 .1rem; }
.sl-brand-sub { font-size: .8rem; color: var(--sl-faint); margin-bottom: .8rem; }
.sl-nav-label { font-size: .78rem; font-weight: 650; color: var(--sl-faint); margin: .9rem 0 .15rem .2rem; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] { border-radius: var(--sl-radius-sm); padding: .3rem .55rem; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover { background: var(--sl-surface-alt); }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] { background: var(--sl-primary-soft); color: var(--sl-primary-hover); font-weight: 650; }

/* ---------- page header & scope bar ---------- */
.sl-page-head { margin: 0 0 1rem; }
.sl-page-head h1 { font-size: 2rem; margin: 0 0 .25rem; line-height: 1.15; overflow-wrap: anywhere; }
.sl-page-head p { color: var(--sl-muted); margin: 0; max-width: 70ch; }
.sl-scope { display: flex; flex-wrap: wrap; gap: .4rem .9rem; align-items: center; background: var(--sl-primary-soft);
  border: 1px solid #D5D3F5; border-radius: var(--sl-radius); padding: .55rem .85rem; margin-bottom: 1.2rem; font-size: .88rem; }
.sl-scope strong { font-weight: 700; overflow-wrap: anywhere; }
.sl-scope .sl-scope-item { color: var(--sl-muted); }
.sl-scope.sl-cross { background: var(--sl-warning-soft); border-color: #F0CFA0; }

/* ---------- cards, metrics ---------- */
.sl-card, [class*="st-key-sl-card"] { background: var(--sl-surface); border: 1px solid var(--sl-border); border-radius: var(--sl-radius);
  box-shadow: var(--sl-shadow); padding: 1rem 1.1rem; overflow-wrap: anywhere; }
[class*="st-key-sl-card"] { margin-bottom: .9rem; }
.sl-card + .sl-card { margin-top: .7rem; }
.sl-card-title { font-weight: 650; font-size: 1rem; margin: 0 0 .15rem; overflow-wrap: anywhere; }
.sl-card-meta { color: var(--sl-faint); font-size: .8rem; display: flex; flex-wrap: wrap; gap: .25rem .8rem; margin-top: .35rem; }
.sl-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .75rem; margin: .4rem 0 1.1rem; }
.sl-metric { background: var(--sl-surface); border: 1px solid var(--sl-border); border-radius: var(--sl-radius); padding: .8rem .95rem; }
.sl-metric-label { font-size: .78rem; color: var(--sl-muted); font-weight: 600; }
.sl-metric-value { font-family: var(--sl-display); font-size: 1.85rem; line-height: 1.15; font-weight: 600; margin-top: .1rem; }
.sl-metric-hint { font-size: .76rem; color: var(--sl-faint); margin-top: .15rem; }
.sl-metric.sl-tone-danger .sl-metric-value { color: var(--sl-danger); }
.sl-metric.sl-tone-success .sl-metric-value { color: var(--sl-success); }
.sl-metric.sl-tone-warning .sl-metric-value { color: var(--sl-warning); }

/* ---------- badges ---------- */
.sl-badge { display: inline-block; font-size: .74rem; font-weight: 650; padding: .12rem .5rem; border-radius: 999px;
  border: 1px solid transparent; white-space: nowrap; vertical-align: middle; }
.sl-badge.sl-neutral { background: var(--sl-surface-alt); color: var(--sl-muted); border-color: var(--sl-border); }
.sl-badge.sl-primary { background: var(--sl-primary-soft); color: var(--sl-primary-hover); }
.sl-badge.sl-success { background: var(--sl-success-soft); color: #0B5F58; }
.sl-badge.sl-info { background: var(--sl-info-soft); color: var(--sl-info); }
.sl-badge.sl-warning { background: var(--sl-warning-soft); color: #8A4208; }
.sl-badge.sl-danger { background: var(--sl-danger-soft); color: #9F1239; }

/* ---------- tiered question cards ---------- */
.sl-qcard { background: var(--sl-surface); border: 1px solid var(--sl-border); border-left: 4px solid var(--sl-border-strong);
  border-radius: var(--sl-radius); padding: .8rem 1rem; margin-bottom: .6rem; overflow-wrap: anywhere; }
.sl-qcard .sl-qhead { display: flex; flex-wrap: wrap; gap: .35rem .6rem; align-items: center; font-size: .82rem; color: var(--sl-muted); margin-bottom: .3rem; }
.sl-qcard .sl-qtext { font-size: .96rem; }
.sl-qcard .sl-qsrc { font-size: .82rem; color: var(--sl-muted); margin-top: .4rem; }
.sl-q-Strong { border-left-color: ${c_strong}; } .sl-q-Moderate { border-left-color: ${c_moderate}; }
.sl-q-Weak { border-left-color: ${c_weak}; } .sl-q-Unmatched { border-left-color: ${c_unmatched}; }

/* ---------- notices & empty states ---------- */
.sl-notice { border-radius: var(--sl-radius); padding: .7rem .95rem; border: 1px solid; margin: .5rem 0 .9rem; font-size: .92rem; overflow-wrap: anywhere; }
.sl-notice strong { display: block; margin-bottom: .1rem; }
.sl-notice.sl-info { background: var(--sl-info-soft); border-color: #BFD3F7; }
.sl-notice.sl-success { background: var(--sl-success-soft); border-color: #A9DDD5; }
.sl-notice.sl-warning { background: var(--sl-warning-soft); border-color: #F0CFA0; }
.sl-notice.sl-error { background: var(--sl-danger-soft); border-color: #F2B4C2; }
.sl-empty { text-align: center; background: var(--sl-surface); border: 1px dashed var(--sl-border-strong); border-radius: var(--sl-radius);
  padding: 2rem 1.2rem; margin: .8rem 0; }
.sl-empty h3 { margin: .2rem 0 .35rem; } .sl-empty p { color: var(--sl-muted); max-width: 52ch; margin: 0 auto; }
.sl-caption { color: var(--sl-muted); font-size: .85rem; max-width: 80ch; }
.sl-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .85em; }
.sl-snippet { background: var(--sl-surface-alt); border-radius: var(--sl-radius-sm); padding: .6rem .8rem; font-size: .88rem; overflow-wrap: anywhere; white-space: pre-wrap; }

/* ---------- Streamlit widgets ---------- */
[data-testid^="stBaseButton"] { border-radius: var(--sl-radius-sm); font-weight: 600; transition: background-color .12s ease, border-color .12s ease; }
[data-testid="stBaseButton-primary"] { background: var(--sl-primary); border-color: var(--sl-primary); color: #fff; }
[data-testid="stBaseButton-primary"]:hover { background: var(--sl-primary-hover); border-color: var(--sl-primary-hover); }
[data-testid="stBaseButton-secondary"] { background: var(--sl-surface); border: 1px solid var(--sl-border-strong); color: var(--sl-ink); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--sl-primary); color: var(--sl-primary); }
[data-baseweb="tab-list"] { gap: .25rem; flex-wrap: wrap; }
[data-baseweb="tab"] { font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { background: var(--sl-surface); border: 1.5px dashed var(--sl-border-strong); border-radius: var(--sl-radius); }
[data-testid="stExpander"] { background: var(--sl-surface); border: 1px solid var(--sl-border); border-radius: var(--sl-radius); }
[data-testid="stDataFrame"] { border: 1px solid var(--sl-border); border-radius: var(--sl-radius-sm); }

/* ---------- small screens ---------- */
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 1rem .9rem 3rem; }
  .sl-page-head h1 { font-size: 1.6rem; }
  .sl-metrics { grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); }
  .sl-metric-value { font-size: 1.5rem; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition: none !important; animation: none !important; scroll-behavior: auto !important; } }
""")


def build_css() -> str:
    return _CSS.substitute(**TOKENS, c_strong=TIER_COLORS["Strong"], c_moderate=TIER_COLORS["Moderate"],
                           c_weak=TIER_COLORS["Weak"], c_unmatched=TIER_COLORS["Unmatched"])


def inject_theme() -> None:
    """Inject the global stylesheet once per script run."""
    st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)
