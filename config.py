"""
Configuration settings and constants for SyllabusLens.
"""

from dataclasses import dataclass
from typing import Dict, Any

APP_NAME = "SyllabusLens"
APP_TAGLINE = "Academic Intelligence & Question Paper Coverage Analyzer"
APP_VERSION = "1.0.0"

# Similarity Thresholds
DEFAULT_STRONG_THRESHOLD = 0.35
DEFAULT_MODERATE_THRESHOLD = 0.18
DEFAULT_WEAK_THRESHOLD = 0.08

# Supported file formats
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".pptx"]

# UI Theme Color Palette (Dark-first, Deep Navy/Charcoal with Violet/Cyan accents)
THEME_COLORS = {
    "background": "#0b0f19",
    "surface": "#111827",
    "surface_card": "#1e293b",
    "surface_card_border": "#334155",
    "primary": "#8b5cf6",       # Violet accent
    "primary_light": "#a78bfa",
    "secondary": "#06b6d4",     # Cyan accent
    "secondary_light": "#67e8f9",
    "success": "#10b981",       # Emerald green
    "warning": "#f59e0b",       # Amber
    "danger": "#ef4444",        # Red
    "text_main": "#f8fafc",     # Crisp slate white
    "text_muted": "#94a3b8",    # Slate gray
    "chart_palette": [
        "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b",
        "#ec4899", "#3b82f6", "#14b8a6", "#f97316"
    ]
}

# Vectorization defaults
DEFAULT_MAX_FEATURES = 2500
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 0.95
DEFAULT_USE_IDF = True
DEFAULT_NORM = "l2"

# Document Segmentation defaults
DEFAULT_PAGE_CHUNK_SIZE = 400  # words
DEFAULT_CHUNK_OVERLAP = 50     # words


# ---------------------------------------------------------------------------
# Product / storage configuration (v2)
# ---------------------------------------------------------------------------
import os
from pathlib import Path

APP_VERSION = "2.0.0-dev"

# All persistent state lives under one directory, outside the source tree by default
# is NOT assumed: it defaults to ./data (git-ignored). Override with SYLLABUSLENS_DATA_DIR.
DATA_DIR = Path(os.environ.get("SYLLABUSLENS_DATA_DIR", Path(__file__).parent / "data")).resolve()
DB_FILENAME = "syllabuslens.sqlite3"
UPLOAD_DIRNAME = "uploads"

# Upload safety limits
MAX_UPLOAD_BYTES = int(os.environ.get("SYLLABUSLENS_MAX_UPLOAD_MB", "25")) * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = (".pdf", ".txt", ".md", ".pptx")

# Document kinds: key -> (label, is_question_paper, default_eligible_as_notes)
DOCUMENT_KINDS = {
    "lecture_notes":  ("Lecture notes", False, True),
    "lecture_slides": ("Lecture slides", False, True),
    "question_paper": ("Previous-year question paper", True, False),
    "assignment":     ("Assignment", False, False),
    "tutorial":       ("Tutorial", False, True),
    "syllabus":       ("Syllabus document", False, False),
    "reference":      ("Reference material", False, True),
    "other":          ("Other academic document", False, False),
}
QUESTION_PAPER_KIND = "question_paper"

# Subject suggestion (transparent TF-IDF profile matching; scores are NOT probabilities)
SUGGEST_MIN_SCORE = 0.05        # below this, the classifier reports "insufficient evidence"
SUGGEST_AMBIGUITY_MARGIN = 0.25  # relative gap between #1 and #2 below which result is "ambiguous"
