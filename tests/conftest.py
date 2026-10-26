import os
import sys
import tempfile
from pathlib import Path

# Tests must never read or write the real data directory: point the default at a throwaway location
# BEFORE config is imported anywhere.
os.environ["SYLLABUSLENS_DATA_DIR"] = tempfile.mkdtemp(prefix="syllabuslens-test-")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.container import build_services  # noqa: E402
from tests.fixtures import NLP, OS, DBMS  # noqa: E402


@pytest.fixture
def svc(tmp_path):
    return build_services(tmp_path / "data")


def _subject_with_notes(svc, name, topics, notes):
    s = svc.subjects.create(name, description=notes["description"], topics=topics)
    docs = [svc.library.ingest(s.id, fn, txt.encode(), doc_kind="lecture_notes").document for fn, txt in notes["files"].items()]
    return s, docs


@pytest.fixture
def world(svc):
    """NLP + OS subjects with SYNTHETIC test notes and papers (see tests/fixtures.py)."""
    nlp, nlp_docs = _subject_with_notes(svc, "Natural Language Processing", NLP["topics"], NLP)
    osub, os_docs = _subject_with_notes(svc, "Operating Systems", OS["topics"], OS)
    nlp_p1 = svc.library.ingest(nlp.id, "nlp_paper_2023.txt", NLP["paper1"].encode(), doc_kind="question_paper",
                                title="NLP End-Sem 2023").document
    nlp_p2 = svc.library.ingest(nlp.id, "nlp_paper_2024.txt", NLP["paper2"].encode(), doc_kind="question_paper",
                                title="NLP End-Sem 2024").document
    os_p1 = svc.library.ingest(osub.id, "os_paper_2023.txt", OS["paper1"].encode(), doc_kind="question_paper",
                               title="OS End-Sem 2023").document
    return dict(nlp=nlp, os=osub, nlp_docs=nlp_docs, os_docs=os_docs, nlp_p1=nlp_p1, nlp_p2=nlp_p2, os_p1=os_p1)
