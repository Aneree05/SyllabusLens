"""Helpers for driving individual pages with Streamlit's AppTest against an isolated Services instance."""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from streamlit.testing.v1 import AppTest


class FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""
    def __init__(self, name: str, data: bytes):
        self.name, self._data, self.size, self.type = name, data, len(data), "application/octet-stream"

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


def _page_script(module: str):
    import streamlit as st
    from importlib import import_module
    from components import theme
    theme.inject_theme()
    fake = st.session_state.get("_fake_files")
    if fake is not None:
        st.file_uploader = lambda *a, **k: list(fake) if k.get("accept_multiple_files") else (fake[0] if fake else None)
    import_module(module).render()


def open_page(svc, module: str, *, active: Optional[str] = None, files: Optional[Iterable[FakeUpload]] = None) -> AppTest:
    at = AppTest.from_function(_page_script, args=(module,), default_timeout=90)
    at.session_state["_services_override"] = svc
    if active:
        at.session_state["active_subject_id"] = active
    if files is not None:
        at.session_state["_fake_files"] = list(files)
    return at.run()


def md(at: AppTest) -> str:
    """All rendered markdown/HTML text on the page, for assertions."""
    return "\n".join(m.value for m in at.markdown if not m.value.lstrip().startswith("<style>"))


def short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]
