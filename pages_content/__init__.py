"""
Page registry. Each module exposes render(). Only implemented pages are listed here; nothing is
registered as a placeholder.
"""
from __future__ import annotations

from importlib import import_module
from typing import List, NamedTuple


class PageSpec(NamedTuple):
    section: str
    key: str            # stable id, also the URL path
    title: str
    icon: str           # Material Symbols name
    module: str


PAGE_SPECS: List[PageSpec] = [
    PageSpec("Workspace", "overview", "Overview", ":material/dashboard:", "pages_content.overview"),
    PageSpec("Workspace", "subjects", "Subjects", ":material/folder_special:", "pages_content.subjects"),
    PageSpec("Workspace", "library", "Document Library", ":material/library_books:", "pages_content.library"),
    PageSpec("Analysis", "analysis", "Question Analysis", ":material/query_stats:", "pages_content.analysis"),
    PageSpec("Analysis", "explorer", "Question Explorer", ":material/manage_search:", "pages_content.explorer"),
    PageSpec("Analysis", "coverage", "Coverage Lab", ":material/stacked_bar_chart:", "pages_content.coverage"),
    PageSpec("Analysis", "compare", "Compare Papers", ":material/compare_arrows:", "pages_content.compare"),
    PageSpec("Analysis", "planner", "Study Planner", ":material/checklist:", "pages_content.planner"),
    PageSpec("Analysis", "history", "Analysis History", ":material/history:", "pages_content.history"),
    PageSpec("Lab", "nlp-lab", "NLP Lab", ":material/science:", "pages_content.nlp_lab"),
    PageSpec("Lab", "evaluation", "Evaluation", ":material/fact_check:", "pages_content.evaluation"),
    PageSpec("System", "diagnostics", "Diagnostics", ":material/health_and_safety:", "pages_content.diagnostics"),
]


def renderer(spec: PageSpec):
    def _run():
        import_module(spec.module).render()
    _run.__name__ = f"page_{spec.key.replace('-', '_')}"
    return _run
