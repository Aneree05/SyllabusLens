"""
SQLite persistence for SyllabusLens.

One connection per operation (Streamlit runs scripts on multiple threads), foreign keys
enforced, WAL journal. Sections carry a denormalised ``subject_id`` so retrieval can
filter by subject directly in SQL as a second line of defence behind the service layer.
Replaceable: services depend on the Database methods only, not on sqlite3.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS subjects (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    course_code  TEXT NOT NULL DEFAULT '',
    semester     TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_subject_name ON subjects (lower(name));

CREATE TABLE IF NOT EXISTS subject_topics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    topic      TEXT NOT NULL,
    position   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_topics_subject ON subject_topics (subject_id);

CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    subject_id     TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    filename       TEXT NOT NULL,
    title          TEXT NOT NULL,
    doc_kind       TEXT NOT NULL,
    file_ext       TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    file_hash      TEXT NOT NULL,      -- sha256 of the raw bytes
    text_hash      TEXT NOT NULL,      -- sha256 of normalised extracted text
    status         TEXT NOT NULL,      -- processed | failed | no_text
    status_detail  TEXT NOT NULL DEFAULT '',
    page_count     INTEGER NOT NULL DEFAULT 0,
    stored_name    TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT '[]',
    academic_year  TEXT NOT NULL DEFAULT '',
    semester       TEXT NOT NULL DEFAULT '',
    exam_session   TEXT NOT NULL DEFAULT '',
    course_code    TEXT NOT NULL DEFAULT '',
    full_text      TEXT NOT NULL DEFAULT '',
    corrections    TEXT NOT NULL DEFAULT '',   -- JSON: user-corrected question list (papers)
    uploaded_at    TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_documents_subject ON documents (subject_id);
CREATE INDEX IF NOT EXISTS ix_documents_hash ON documents (file_hash);

CREATE TABLE IF NOT EXISTS sections (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    subject_id    TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    page_or_slide TEXT NOT NULL,
    heading       TEXT NOT NULL,
    text          TEXT NOT NULL,
    word_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_sections_subject ON sections (subject_id);
CREATE INDEX IF NOT EXISTS ix_sections_doc ON sections (document_id);

CREATE TABLE IF NOT EXISTS analyses (
    id                  TEXT PRIMARY KEY,
    subject_id          TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    method              TEXT NOT NULL,
    settings            TEXT NOT NULL,   -- JSON
    inputs              TEXT NOT NULL,   -- JSON: frozen list of documents used
    corpus_fingerprint  TEXT NOT NULL,
    cross_subject       INTEGER NOT NULL DEFAULT 0,
    results             TEXT NOT NULL    -- JSON snapshot (independent of later edits)
);
CREATE INDEX IF NOT EXISTS ix_analyses_subject ON analyses (subject_id, created_at);

CREATE TABLE IF NOT EXISTS review_marks (
    subject_id  TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    section_id  TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    PRIMARY KEY (subject_id, section_id)
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Transactional connection: commits on success, rolls back on error."""
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
