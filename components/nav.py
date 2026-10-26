"""Page registry and navigation helpers."""
from __future__ import annotations

from typing import Dict

import streamlit as st

PAGES: Dict[str, "st.Page"] = {}      # filled by app.py on every run
REQUEST_KEY = "_nav_requested"        # recorded when no page registry exists (tests / harness)


def goto(key: str) -> None:
    page = PAGES.get(key)
    if page is not None:
        st.switch_page(page)
    else:
        st.session_state[REQUEST_KEY] = key
