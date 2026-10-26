"""
Controlled storage for uploaded files.

Files are written under <data_dir>/uploads/<subject_id>/<document_id><ext>. The user's
original filename is metadata only and is never used to build a filesystem path, which
rules out path traversal and name collisions between subjects.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path


class FileStore:
    _SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _check(self, *parts: str) -> None:
        for p in parts:
            if not self._SAFE_ID.match(p):
                raise ValueError(f"Unsafe storage identifier: {p!r}")

    def _path(self, subject_id: str, stored_name: str) -> Path:
        stem = stored_name.rsplit(".", 1)[0]
        self._check(subject_id, stem)
        path = (self.root / subject_id / stored_name).resolve()
        if self.root not in path.parents:
            raise ValueError("Resolved path escapes the upload directory.")
        return path

    def save(self, subject_id: str, stored_name: str, data: bytes) -> Path:
        path = self._path(subject_id, stored_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read(self, subject_id: str, stored_name: str) -> bytes:
        return self._path(subject_id, stored_name).read_bytes()

    def move(self, old_subject: str, new_subject: str, stored_name: str) -> None:
        src = self._path(old_subject, stored_name)
        dst = self._path(new_subject, stored_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))

    def delete(self, subject_id: str, stored_name: str) -> None:
        path = self._path(subject_id, stored_name)
        if path.exists():
            path.unlink()

    def delete_subject(self, subject_id: str) -> None:
        self._check(subject_id)
        d = (self.root / subject_id).resolve()
        if self.root in d.parents and d.exists():
            shutil.rmtree(d)
