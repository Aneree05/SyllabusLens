"""Design-system guarantees: accessible contrast, single source of truth for colours, no unescaped HTML in primitives."""
import re

from components import ui
from components.theme import TIER_INK, TIER_SOFT, TOKENS as T, build_css


def _lum(h):
    h = h.lstrip("#"); r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _cr(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_colour_pairs_meet_wcag_aa():
    pairs = [(T["ink"], T["bg"]), (T["ink"], T["surface"]), (T["ink_muted"], T["surface"]), (T["ink_muted"], T["bg"]),
             (T["ink_faint"], T["surface"]), (T["ink_faint"], T["bg"]), (T["ink_faint"], T["surface_alt"]), ("#FFFFFF", T["primary"]),
             (T["primary_hover"], T["primary_soft"]), (T["primary"], T["surface"]), ("#0B5F58", T["success_soft"]), (T["info"], T["info_soft"]),
             ("#8A4208", T["warning_soft"]), ("#9F1239", T["danger_soft"]), (T["danger"], T["surface"]), (T["warning"], T["surface"]), (T["success"], T["surface"])]
    pairs += [(TIER_INK[k], TIER_SOFT[k]) for k in TIER_INK]
    low = [(a, b, round(_cr(a, b), 2)) for a, b in pairs if _cr(a, b) < 4.5]
    assert not low, low


def test_stylesheet_has_no_unresolved_tokens_and_honours_reduced_motion_and_focus():
    css = build_css()
    assert not re.findall(r"\$\{?\w+", css)
    assert "prefers-reduced-motion: reduce" in css and ":focus-visible" in css and "max-width: 768px" in css


def test_primitives_escape_all_dynamic_text():
    evil = '<img src=x onerror=alert(1)> "q" $x$'
    for html in (ui.badge(evil), ui.tier_badge(evil)):
        assert "<img" not in html and "&lt;img" in html or evil not in html
    q = dict(best_tier="Strong", display_id=evil, best_score=0.5, text=evil, matches=[dict(heading=evil, doc_name=evil, locator=evil)], marks=evil, paper_label=evil)
    out = ui.question_card_html(q, show_paper=True)
    assert "<img" not in out and "$x$" not in out
