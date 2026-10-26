"""
Document library: ingestion, metadata, duplicate detection, reclassification.

Every section stored here gets an ID of the form ``<document_id>:<ordinal>``. Because
document IDs are random and unique, two subjects can hold files with identical names
(and identical content) without any ID collision.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

import config
from nlp.question_segmentation import QuestionItem, QuestionSegmenter
from services.document_service import DocumentService
from services.errors import (
    DocumentValidationError, DuplicateDocumentError, NotFoundError,
)
from services.subject_service import SubjectService, now_iso
from storage import Database, FileStore

_SORTS = {
    "uploaded_desc": "uploaded_at DESC",
    "uploaded_asc": "uploaded_at ASC",
    "title": "lower(title) ASC",
    "filename": "lower(filename) ASC",
    "size_desc": "size_bytes DESC",
}


@dataclass
class Document:
    id: str
    subject_id: str
    filename: str
    title: str
    doc_kind: str
    file_ext: str
    size_bytes: int
    file_hash: str
    text_hash: str
    status: str
    status_detail: str
    page_count: int
    stored_name: str
    tags: List[str]
    academic_year: str
    semester: str
    exam_session: str
    course_code: str
    uploaded_at: str
    updated_at: str
    has_corrections: bool = False

    @property
    def is_question_paper(self) -> bool:
        return self.doc_kind == config.QUESTION_PAPER_KIND

    @property
    def kind_label(self) -> str:
        return config.DOCUMENT_KINDS.get(self.doc_kind, (self.doc_kind,))[0]


def _from_row(r) -> Document:
    return Document(
        id=r["id"], subject_id=r["subject_id"], filename=r["filename"], title=r["title"],
        doc_kind=r["doc_kind"], file_ext=r["file_ext"], size_bytes=r["size_bytes"],
        file_hash=r["file_hash"], text_hash=r["text_hash"], status=r["status"],
        status_detail=r["status_detail"], page_count=r["page_count"], stored_name=r["stored_name"],
        tags=json.loads(r["tags"] or "[]"), academic_year=r["academic_year"], semester=r["semester"],
        exam_session=r["exam_session"], course_code=r["course_code"],
        uploaded_at=r["uploaded_at"], updated_at=r["updated_at"],
        has_corrections=bool(r["corrections"]),
    )


@dataclass
class UploadPreview:
    """What an upload would look like, computed WITHOUT storing anything."""
    filename: str
    error: Optional[str] = None            # validation failure (type, size, header...)
    ext: str = ""
    size_bytes: int = 0
    file_hash: str = ""
    status: str = ""                       # processed | failed | no_text
    status_detail: str = ""
    text: str = ""
    page_count: int = 0
    kind_suggestion: Optional[Tuple[str, str]] = None
    question_count: int = 0


@dataclass
class IngestResult:
    document: Document
    kind_suggestion: Optional[Tuple[str, str]] = None   # (kind, reason) when the user did not choose
    same_subject_text_duplicates: List[str] = field(default_factory=list)
    other_subject_duplicates: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


_QP_MARKERS = re.compile(
    r"(time allowed|maximum marks|max\.? marks|answer (?:any|all)|attempt (?:any|all)|"
    r"end[- ]?sem(?:ester)?|mid[- ]?sem(?:ester)?|question paper|examination)", re.I)


def suggest_kind(filename: str, ext: str, text: str) -> Tuple[str, str]:
    """Heuristic kind suggestion. Advisory only - the user's choice is authoritative."""
    if ext == "pptx":
        return "lecture_slides", "PowerPoint files are usually lecture slides."
    sample = (text or "")[:6000]
    markers = len(_QP_MARKERS.findall(sample))
    numbered = len(re.findall(r"(?m)^\s*(?:Q\.?\s*)?\d{1,2}\s*[\.\)]", sample))
    marks = len(re.findall(r"\[\s*\d+\s*(?:marks?|m)?\s*\]|\(\s*\d+\s*(?:marks?|m)\s*\)", sample, re.I))
    if (markers >= 1 and numbered >= 3) or marks >= 3:
        return "question_paper", (
            f"Found exam wording ({markers}), numbered items ({numbered}) and mark indicators ({marks}).")
    return "lecture_notes", "No exam-paper structure detected; treating as notes."


def _safe_display_name(filename: str) -> str:
    name = PureWindowsPath(PurePosixPath(filename or "").name).name
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    return name[:200]


class LibraryService:
    def __init__(self, db: Database, files: FileStore, subjects: SubjectService,
                 extractor: Optional[DocumentService] = None):
        self.db, self.files, self.subjects = db, files, subjects
        self.extractor = extractor or DocumentService()
        self.q_segmenter = QuestionSegmenter()

    # ------------------------------------------------------------------ ingest
    def _validate_upload(self, filename: str, data: bytes) -> Tuple[str, str]:
        display = _safe_display_name(filename)
        if not display:
            raise DocumentValidationError("The uploaded file has no usable name.")
        ext = ("." + display.rsplit(".", 1)[-1].lower()) if "." in display else ""
        if ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
            raise DocumentValidationError(
                f"'{display}': unsupported file type '{ext or 'none'}'. "
                f"Allowed types: {', '.join(config.ALLOWED_UPLOAD_EXTENSIONS)}.")
        if not data:
            raise DocumentValidationError(f"'{display}' is empty. Choose a file that has content.")
        if len(data) > config.MAX_UPLOAD_BYTES:
            mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
            raise DocumentValidationError(f"'{display}' exceeds the {mb} MB upload limit.")
        # Verify content matches the claimed type (extension alone is not trusted).
        if ext == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF-"):
            raise DocumentValidationError(f"'{display}' is not a valid PDF (missing PDF header).")
        if ext == ".pptx" and not data.startswith(b"PK"):
            raise DocumentValidationError(f"'{display}' is not a valid .pptx file.")
        if ext in (".txt", ".md") and b"\x00" in data[:4096]:
            raise DocumentValidationError(f"'{display}' looks like a binary file, not text.")
        return display, ext.lstrip(".")

    def _extract(self, data: bytes, display: str):
        ex = self.extractor.extract_from_bytes(data, display)
        if ex.error_message:
            return ex, "failed", ex.error_message
        if not (ex.full_text or "").strip():
            return ex, "no_text", ("No extractable text. This looks like a scanned/image-only file; "
                                    "OCR is not included in this version.")
        if ex.is_likely_scanned:
            return ex, "processed", "Very little text per page: extraction may be incomplete (possible scan)."
        return ex, "processed", ""

    @staticmethod
    def _guess_title(text: str, display: str) -> str:
        for line in (text or "").splitlines():
            line = line.strip().lstrip("#").strip()
            if 4 <= len(line) <= 120 and re.search(r"[A-Za-z]", line):
                return line
        return display.rsplit(".", 1)[0]

    def ingest(self, subject_id: str, filename: str, data: bytes, *,
               doc_kind: Optional[str] = None, title: Optional[str] = None,
               tags: Optional[List[str]] = None, academic_year: str = "", semester: str = "",
               exam_session: str = "", course_code: str = "",
               allow_duplicate: bool = False) -> IngestResult:
        if not self.subjects.exists(subject_id):
            raise NotFoundError("Choose an existing subject before uploading.")
        display, ext = self._validate_upload(filename, data)
        if doc_kind is not None and doc_kind not in config.DOCUMENT_KINDS:
            raise DocumentValidationError(f"Unknown document type '{doc_kind}'.")

        file_hash = sha256_bytes(data)
        with self.db.connect() as conn:
            dup = conn.execute("SELECT id, filename FROM documents WHERE subject_id=? AND file_hash=?",
                               (subject_id, file_hash)).fetchone()
        if dup and not allow_duplicate:
            raise DuplicateDocumentError(
                f"'{display}' is identical to '{dup['filename']}' already in this subject. "
                "Upload skipped; open the existing document or pass allow_duplicate.", dup["id"])

        extracted, status, detail = self._extract(data, display)
        text = extracted.full_text or ""
        suggestion = None
        if doc_kind is None:
            suggestion = suggest_kind(display, ext, text)
            doc_kind = suggestion[0]

        doc_id = "doc_" + uuid.uuid4().hex[:16]
        stored_name = f"{doc_id}.{ext}"
        text_hash = text_fingerprint(text)
        ts = now_iso()
        clean_tags = sorted({t.strip() for t in (tags or []) if t and t.strip()})

        with self.db.connect() as conn:
            same_text = [r["id"] for r in conn.execute(
                "SELECT id FROM documents WHERE subject_id=? AND text_hash=? AND ?<>''",
                (subject_id, text_hash, text.strip())).fetchall()]
            others = [dict(id=r["id"], filename=r["filename"], subject=r["sname"]) for r in conn.execute(
                "SELECT d.id, d.filename, s.name sname FROM documents d JOIN subjects s ON s.id=d.subject_id "
                "WHERE d.subject_id<>? AND d.file_hash=?", (subject_id, file_hash)).fetchall()]
            conn.execute(
                """INSERT INTO documents(id,subject_id,filename,title,doc_kind,file_ext,size_bytes,file_hash,
                   text_hash,status,status_detail,page_count,stored_name,tags,academic_year,semester,
                   exam_session,course_code,full_text,uploaded_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (doc_id, subject_id, display, (title or self._guess_title(text, display)).strip()[:200],
                 doc_kind, ext, len(data), file_hash, text_hash, status, detail,
                 extracted.page_or_slide_count, stored_name, json.dumps(clean_tags), academic_year.strip(),
                 semester.strip(), exam_session.strip(), course_code.strip(), text[:2_000_000], ts, ts))
            if status == "processed" and doc_kind != config.QUESTION_PAPER_KIND:
                self._store_sections(conn, doc_id, subject_id, extracted.sections)
        self.files.save(subject_id, stored_name, data)

        warnings = []
        if status != "processed" or detail:
            warnings.append(detail)
        if same_text:
            warnings.append("Extracted text matches another document in this subject (different file bytes).")
        if others:
            warnings.append("The same file also exists in another subject: " +
                            ", ".join(f"{o['filename']} ({o['subject']})" for o in others))
        return IngestResult(self.get(doc_id), suggestion, same_text, others, [w for w in warnings if w])

    @staticmethod
    def _store_sections(conn, doc_id: str, subject_id: str, sections) -> None:
        for i, sec in enumerate(sections, 1):
            conn.execute(
                "INSERT INTO sections(id,document_id,subject_id,ordinal,page_or_slide,heading,text,word_count) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (f"{doc_id}:{i}", doc_id, subject_id, i, sec.page_or_slide, sec.heading, sec.text, sec.word_count))

    # ----------------------------------------------------------------- preview
    def preview_upload(self, filename: str, data: bytes) -> UploadPreview:
        """Validate and extract an upload for review. Nothing is written to the database or disk."""
        try:
            display, ext = self._validate_upload(filename, data)
        except DocumentValidationError as exc:
            return UploadPreview(filename=_safe_display_name(filename) or "(unnamed)", error=str(exc),
                                 size_bytes=len(data or b""))
        extracted, status, detail = self._extract(data, display)
        text = extracted.full_text or ""
        return UploadPreview(
            filename=display, ext=ext, size_bytes=len(data), file_hash=sha256_bytes(data), status=status,
            status_detail=detail, text=text, page_count=extracted.page_or_slide_count,
            kind_suggestion=suggest_kind(display, ext, text),
            question_count=len(self.q_segmenter.segment(text)) if text.strip() else 0)

    def duplicates_of(self, file_hash: str) -> List[Dict[str, str]]:
        """Existing documents with identical bytes, across all subjects (for upload warnings)."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT d.id, d.filename, d.subject_id, s.name AS subject_name FROM documents d "
                "JOIN subjects s ON s.id=d.subject_id WHERE d.file_hash=?", (file_hash,)).fetchall()
        return [dict(r) for r in rows]

    def get_section(self, section_id: str) -> Optional[Dict[str, Any]]:
        """Current text of one section, or None if it no longer exists."""
        with self.db.connect() as conn:
            r = conn.execute(
                "SELECT s.*, d.filename FROM sections s JOIN documents d ON d.id=s.document_id WHERE s.id=?",
                (section_id,)).fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------------ read
    def get(self, doc_id: str) -> Document:
        with self.db.connect() as conn:
            r = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not r:
            raise NotFoundError(f"Document '{doc_id}' does not exist.")
        return _from_row(r)

    def get_text(self, doc_id: str) -> str:
        with self.db.connect() as conn:
            r = conn.execute("SELECT full_text FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not r:
            raise NotFoundError(f"Document '{doc_id}' does not exist.")
        return r["full_text"]

    def list_documents(self, subject_id: Optional[str] = None, *, kind: Optional[str] = None,
                       status: Optional[str] = None, ext: Optional[str] = None,
                       query: Optional[str] = None, search_content: bool = False,
                       sort: str = "uploaded_desc", limit: int = 100, offset: int = 0) -> List[Document]:
        """
        Library listing. ``subject_id=None`` lists across subjects and is intended ONLY for the
        library/dashboard views; retrieval never uses this method (see RetrievalService).
        """
        where, args = [], []
        if subject_id:
            where.append("subject_id=?"); args.append(subject_id)
        if kind:
            where.append("doc_kind=?"); args.append(kind)
        if status:
            where.append("status=?"); args.append(status)
        if ext:
            where.append("file_ext=?"); args.append(ext.lstrip(".").lower())
        if query and query.strip():
            like = "%" + query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            cols = ["title LIKE ? ESCAPE '\\'", "filename LIKE ? ESCAPE '\\'", "tags LIKE ? ESCAPE '\\'"]
            args += [like, like, like]
            if search_content:
                cols.append("full_text LIKE ? ESCAPE '\\'"); args.append(like)
            where.append("(" + " OR ".join(cols) + ")")
        sql = "SELECT * FROM documents" + (" WHERE " + " AND ".join(where) if where else "")
        sql += f" ORDER BY {_SORTS.get(sort, _SORTS['uploaded_desc'])} LIMIT ? OFFSET ?"
        args += [max(1, min(limit, 1000)), max(0, offset)]
        with self.db.connect() as conn:
            return [_from_row(r) for r in conn.execute(sql, args).fetchall()]

    def sections_of(self, doc_id: str) -> List[Dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM sections WHERE document_id=? ORDER BY ordinal", (doc_id,)).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- questions
    def questions_of(self, doc_id: str) -> List[QuestionItem]:
        """User-corrected questions if present, otherwise automatic segmentation."""
        doc = self.get(doc_id)
        if not doc.is_question_paper:
            raise DocumentValidationError(f"'{doc.filename}' is not classified as a question paper.")
        with self.db.connect() as conn:
            r = conn.execute("SELECT corrections, full_text FROM documents WHERE id=?", (doc_id,)).fetchone()
        if r["corrections"]:
            return [QuestionItem(**q) for q in json.loads(r["corrections"])]
        return self.q_segmenter.segment(r["full_text"])

    def save_question_corrections(self, doc_id: str, questions: List[QuestionItem]) -> None:
        cleaned = [q for q in questions if (q.clean_text or "").strip()]
        if not cleaned:
            raise DocumentValidationError("At least one non-empty question is required.")
        ids = [q.question_id for q in cleaned]
        if len(set(ids)) != len(ids):
            raise DocumentValidationError("Question IDs must be unique within a paper.")
        payload = json.dumps([
            dict(question_id=q.question_id, raw_text=q.raw_text, clean_text=q.clean_text, marks=q.marks,
                 parent_id=q.parent_id, sub_label=q.sub_label, metadata=q.metadata) for q in cleaned])
        with self.db.connect() as conn:
            conn.execute("UPDATE documents SET corrections=?, updated_at=? WHERE id=?", (payload, now_iso(), doc_id))

    def clear_question_corrections(self, doc_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("UPDATE documents SET corrections='', updated_at=? WHERE id=?", (now_iso(), doc_id))

    # ------------------------------------------------------------------ mutate
    def rename(self, doc_id: str, title: str) -> Document:
        title = " ".join((title or "").split())
        if not title:
            raise DocumentValidationError("A document title cannot be empty.")
        with self.db.connect() as conn:
            conn.execute("UPDATE documents SET title=?, updated_at=? WHERE id=?", (title[:200], now_iso(), doc_id))
        return self.get(doc_id)

    def update_metadata(self, doc_id: str, *, tags: Optional[List[str]] = None, academic_year: Optional[str] = None,
                        semester: Optional[str] = None, exam_session: Optional[str] = None,
                        course_code: Optional[str] = None) -> Document:
        cur = self.get(doc_id)
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE documents SET tags=?, academic_year=?, semester=?, exam_session=?, course_code=?, updated_at=? WHERE id=?",
                (json.dumps(sorted({t.strip() for t in (tags if tags is not None else cur.tags) if t.strip()})),
                 cur.academic_year if academic_year is None else academic_year.strip(),
                 cur.semester if semester is None else semester.strip(),
                 cur.exam_session if exam_session is None else exam_session.strip(),
                 cur.course_code if course_code is None else course_code.strip(),
                 now_iso(), doc_id))
        return self.get(doc_id)

    def _rebuild_sections(self, doc: Document, kind: str) -> None:
        data = self.files.read(doc.subject_id, doc.stored_name)
        extracted, status, detail = self._extract(data, doc.filename)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM sections WHERE document_id=?", (doc.id,))
            if status == "processed" and kind != config.QUESTION_PAPER_KIND:
                self._store_sections(conn, doc.id, doc.subject_id, extracted.sections)
            conn.execute("UPDATE documents SET status=?, status_detail=?, updated_at=? WHERE id=?",
                         (status, detail, now_iso(), doc.id))

    def set_kind(self, doc_id: str, kind: str) -> Document:
        if kind not in config.DOCUMENT_KINDS:
            raise DocumentValidationError(f"Unknown document type '{kind}'.")
        doc = self.get(doc_id)
        if kind == doc.doc_kind:
            return doc
        qp_changed = (kind == config.QUESTION_PAPER_KIND) != doc.is_question_paper
        with self.db.connect() as conn:
            conn.execute("UPDATE documents SET doc_kind=?, corrections='', updated_at=? WHERE id=?",
                         (kind, now_iso(), doc_id))
        if qp_changed:
            self._rebuild_sections(doc, kind)
        return self.get(doc_id)

    def move_to_subject(self, doc_id: str, new_subject_id: str) -> Document:
        """Reassigns a document (and its sections and file). Changes corpus fingerprints of both subjects."""
        if not self.subjects.exists(new_subject_id):
            raise NotFoundError("The target subject does not exist.")
        doc = self.get(doc_id)
        if doc.subject_id == new_subject_id:
            return doc
        with self.db.connect() as conn:
            conn.execute("UPDATE documents SET subject_id=?, updated_at=? WHERE id=?", (new_subject_id, now_iso(), doc_id))
            conn.execute("UPDATE sections SET subject_id=? WHERE document_id=?", (new_subject_id, doc_id))
            self.files.move(doc.subject_id, new_subject_id, doc.stored_name)
        return self.get(doc_id)

    def replace(self, doc_id: str, filename: str, data: bytes) -> Document:
        """Replace file contents, keeping the document ID, subject, kind and metadata."""
        doc = self.get(doc_id)
        display, ext = self._validate_upload(filename, data)
        extracted, status, detail = self._extract(data, display)
        text = extracted.full_text or ""
        new_stored = f"{doc.id}.{ext}"
        with self.db.connect() as conn:
            conn.execute(
                """UPDATE documents SET filename=?, file_ext=?, size_bytes=?, file_hash=?, text_hash=?, status=?,
                   status_detail=?, page_count=?, stored_name=?, full_text=?, corrections='', updated_at=? WHERE id=?""",
                (display, ext, len(data), sha256_bytes(data), text_fingerprint(text), status, detail,
                 extracted.page_or_slide_count, new_stored, text[:2_000_000], now_iso(), doc.id))
            conn.execute("DELETE FROM sections WHERE document_id=?", (doc.id,))
            if status == "processed" and doc.doc_kind != config.QUESTION_PAPER_KIND:
                self._store_sections(conn, doc.id, doc.subject_id, extracted.sections)
        if new_stored != doc.stored_name:
            self.files.delete(doc.subject_id, doc.stored_name)
        self.files.save(doc.subject_id, new_stored, data)
        return self.get(doc.id)

    def delete(self, doc_id: str) -> None:
        doc = self.get(doc_id)
        with self.db.connect() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))  # sections cascade
        self.files.delete(doc.subject_id, doc.stored_name)

    def bulk_update(self, doc_ids: List[str], *, kind: Optional[str] = None,
                    add_tags: Optional[List[str]] = None, move_to: Optional[str] = None) -> int:
        """Apply the same safe change to several documents; returns the number changed."""
        changed = 0
        for d in doc_ids:
            if kind:
                self.set_kind(d, kind)
            if add_tags:
                cur = self.get(d)
                self.update_metadata(d, tags=list(set(cur.tags) | set(add_tags)))
            if move_to:
                self.move_to_subject(d, move_to)
            changed += 1
        return changed

    # ------------------------------------------------------------------- stats
    def stats(self, subject_id: Optional[str] = None) -> Dict[str, Any]:
        args = (subject_id,) if subject_id else ()
        w = "WHERE subject_id=?" if subject_id else ""
        with self.db.connect() as conn:
            by_kind = {r["doc_kind"]: r["c"] for r in conn.execute(
                f"SELECT doc_kind, COUNT(*) c FROM documents {w} GROUP BY doc_kind", args)}
            by_status = {r["status"]: r["c"] for r in conn.execute(
                f"SELECT status, COUNT(*) c FROM documents {w} GROUP BY status", args)}
            sections = conn.execute(f"SELECT COUNT(*) c FROM sections {w}", args).fetchone()["c"]
        return {"by_kind": by_kind, "by_status": by_status, "sections": sections,
                "documents": sum(by_kind.values()),
                "failures": by_status.get("failed", 0) + by_status.get("no_text", 0)}
