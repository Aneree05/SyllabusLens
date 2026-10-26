"""Subject workspaces: create, edit, topics, delete (with typed confirmation)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from services.errors import ConfirmationError, DocumentValidationError, NotFoundError
from storage import Database, FileStore


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Subject:
    id: str
    name: str
    description: str = ""
    course_code: str = ""
    semester: str = ""
    created_at: str = ""
    updated_at: str = ""
    topics: List[str] = field(default_factory=list)


class SubjectService:
    def __init__(self, db: Database, files: FileStore):
        self.db = db
        self.files = files

    # ----- helpers -----
    @staticmethod
    def _clean_topics(topics: Optional[List[str]]) -> List[str]:
        seen, out = set(), []
        for t in topics or []:
            t = " ".join(str(t).split())
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t[:200])
        return out

    def _load_topics(self, conn, subject_id: str) -> List[str]:
        rows = conn.execute(
            "SELECT topic FROM subject_topics WHERE subject_id=? ORDER BY position, id", (subject_id,)
        ).fetchall()
        return [r["topic"] for r in rows]

    def _to_subject(self, conn, row) -> Subject:
        return Subject(
            id=row["id"], name=row["name"], description=row["description"],
            course_code=row["course_code"], semester=row["semester"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            topics=self._load_topics(conn, row["id"]),
        )

    # ----- CRUD -----
    def create(self, name: str, description: str = "", course_code: str = "",
               semester: str = "", topics: Optional[List[str]] = None) -> Subject:
        name = " ".join((name or "").split())
        if not name:
            raise DocumentValidationError("A subject needs a name.")
        if len(name) > 120:
            raise DocumentValidationError("Subject names are limited to 120 characters.")
        sid = "sub_" + uuid.uuid4().hex[:12]
        ts = now_iso()
        with self.db.connect() as conn:
            if conn.execute("SELECT 1 FROM subjects WHERE lower(name)=lower(?)", (name,)).fetchone():
                raise DocumentValidationError(f"A subject named '{name}' already exists. Choose a different name.")
            conn.execute(
                "INSERT INTO subjects(id,name,description,course_code,semester,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (sid, name, description.strip(), course_code.strip(), semester.strip(), ts, ts),
            )
            for pos, t in enumerate(self._clean_topics(topics)):
                conn.execute("INSERT INTO subject_topics(subject_id,topic,position) VALUES(?,?,?)", (sid, t, pos))
        return self.get(sid)

    def get(self, subject_id: str) -> Subject:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM subjects WHERE id=?", (subject_id,)).fetchone()
            if not row:
                raise NotFoundError(f"Subject '{subject_id}' does not exist.")
            return self._to_subject(conn, row)

    def exists(self, subject_id: Optional[str]) -> bool:
        if not subject_id:
            return False
        with self.db.connect() as conn:
            return conn.execute("SELECT 1 FROM subjects WHERE id=?", (subject_id,)).fetchone() is not None

    def list(self) -> List[Subject]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM subjects ORDER BY lower(name)").fetchall()
            return [self._to_subject(conn, r) for r in rows]

    def update(self, subject_id: str, *, name: Optional[str] = None, description: Optional[str] = None,
               course_code: Optional[str] = None, semester: Optional[str] = None,
               topics: Optional[List[str]] = None) -> Subject:
        current = self.get(subject_id)
        new_name = " ".join((name if name is not None else current.name).split())
        if not new_name:
            raise DocumentValidationError("A subject needs a name.")
        with self.db.connect() as conn:
            clash = conn.execute(
                "SELECT 1 FROM subjects WHERE lower(name)=lower(?) AND id<>?", (new_name, subject_id)
            ).fetchone()
            if clash:
                raise DocumentValidationError(f"A subject named '{new_name}' already exists.")
            conn.execute(
                "UPDATE subjects SET name=?, description=?, course_code=?, semester=?, updated_at=? WHERE id=?",
                (new_name,
                 current.description if description is None else description.strip(),
                 current.course_code if course_code is None else course_code.strip(),
                 current.semester if semester is None else semester.strip(),
                 now_iso(), subject_id),
            )
            if topics is not None:
                conn.execute("DELETE FROM subject_topics WHERE subject_id=?", (subject_id,))
                for pos, t in enumerate(self._clean_topics(topics)):
                    conn.execute("INSERT INTO subject_topics(subject_id,topic,position) VALUES(?,?,?)",
                                 (subject_id, t, pos))
        return self.get(subject_id)

    def counts(self, subject_id: str) -> dict:
        with self.db.connect() as conn:
            docs = conn.execute("SELECT COUNT(*) c FROM documents WHERE subject_id=?", (subject_id,)).fetchone()["c"]
            ans = conn.execute("SELECT COUNT(*) c FROM analyses WHERE subject_id=?", (subject_id,)).fetchone()["c"]
        return {"documents": docs, "analyses": ans}

    def delete(self, subject_id: str, confirm_name: str) -> dict:
        """
        Permanently deletes a subject, its documents, extracted sections, stored files and
        saved analyses. Requires the exact subject name as confirmation. Returns what was removed.
        """
        subject = self.get(subject_id)
        if (confirm_name or "").strip() != subject.name:
            raise ConfirmationError(
                f"To delete this subject, type its exact name: '{subject.name}'. Nothing was deleted."
            )
        removed = self.counts(subject_id)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM subjects WHERE id=?", (subject_id,))  # cascades
        self.files.delete_subject(subject_id)
        return removed
