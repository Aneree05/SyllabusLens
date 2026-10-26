"""Persisted analysis history. Each record is a self-contained snapshot of inputs and results."""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Set

from services.errors import NotFoundError
from services.library_service import LibraryService
from services.subject_service import now_iso
from storage import Database


class HistoryService:
    def __init__(self, db: Database, library: LibraryService):
        self.db, self.library = db, library

    def save(self, *, subject_id: str, name: str, method: str, settings: Dict[str, Any],
             inputs: Dict[str, Any], corpus_fingerprint: str, cross_subject: bool,
             record: Dict[str, Any]) -> str:
        aid = "ana_" + uuid.uuid4().hex[:12]
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO analyses(id,subject_id,name,created_at,method,settings,inputs,corpus_fingerprint,"
                "cross_subject,results) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (aid, subject_id, name, now_iso(), method, json.dumps(settings), json.dumps(inputs),
                 corpus_fingerprint, int(cross_subject), json.dumps(record)))
        return aid

    @staticmethod
    def _row(r, with_results: bool) -> Dict[str, Any]:
        d = dict(id=r["id"], subject_id=r["subject_id"], name=r["name"], created_at=r["created_at"],
                 method=r["method"], settings=json.loads(r["settings"]), inputs=json.loads(r["inputs"]),
                 corpus_fingerprint=r["corpus_fingerprint"], cross_subject=bool(r["cross_subject"]))
        if with_results:
            d["record"] = json.loads(r["results"])
        return d

    def list(self, subject_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id,subject_id,name,created_at,method,settings,inputs,corpus_fingerprint,cross_subject,results "
                "FROM analyses WHERE subject_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (subject_id, limit)).fetchall()
        return [self._row(r, with_results=False) for r in rows]

    def get(self, analysis_id: str) -> Dict[str, Any]:
        with self.db.connect() as conn:
            r = conn.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        if not r:
            raise NotFoundError(f"Analysis '{analysis_id}' does not exist.")
        return self._row(r, with_results=True)

    def exists(self, analysis_id: str) -> bool:
        with self.db.connect() as conn:
            return conn.execute("SELECT 1 FROM analyses WHERE id=?", (analysis_id,)).fetchone() is not None

    def delete(self, analysis_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))

    def input_status(self, analysis_id: str) -> Dict[str, Any]:
        """Compares the frozen inputs with the library NOW. The stored results are never altered."""
        rec = self.get(analysis_id)
        changes = []
        for group in ("papers", "documents"):
            for item in rec["inputs"].get(group, []):
                try:
                    cur = self.library.get(item["document_id"])
                except NotFoundError:
                    changes.append(f"'{item['filename']}' was deleted.")
                    continue
                if group == "documents" and cur.subject_id != item.get("subject_id", cur.subject_id):
                    changes.append(f"'{item['filename']}' was moved to another subject.")
                if cur.file_hash != item["file_hash"]:
                    changes.append(f"'{item['filename']}' was replaced with different content.")
                if group == "papers" and not cur.is_question_paper:
                    changes.append(f"'{item['filename']}' is no longer a question paper.")
        return dict(status="inputs_changed" if changes else "current", changes=changes)

    # ---- revision checklist state (per subject; section IDs are document-scoped) ----
    def mark_reviewed(self, subject_id: str, section_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO review_marks(subject_id,section_id,reviewed_at) VALUES(?,?,?)",
                         (subject_id, section_id, now_iso()))

    def unmark_reviewed(self, subject_id: str, section_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM review_marks WHERE subject_id=? AND section_id=?", (subject_id, section_id))

    def reviewed_ids(self, subject_id: str) -> Set[str]:
        with self.db.connect() as conn:
            return {r["section_id"] for r in conn.execute(
                "SELECT section_id FROM review_marks WHERE subject_id=?", (subject_id,))}
