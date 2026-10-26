"""Phase 1-2 UI behaviour: onboarding, subject management, upload flow, library. Synthetic data only."""
import pytest

from services.sample_data_service import SAMPLE_SUBJECT_NAME
from tests.fixtures import DBMS, NLP, OS
from tests.ui_harness import FakeUpload, md, open_page, short_hash

SUBJECTS = "pages_content.subjects"
LIBRARY = "pages_content.library"


def _no_errors(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


# ------------------------------------------------------------------ onboarding & subjects
def test_first_run_shows_onboarding_not_empty_metric_grid(svc):
    at = open_page(svc, SUBJECTS)
    _no_errors(at)
    assert "Create your first subject" in md(at) and "sl-metric" not in md(at)
    assert [b.label for b in at.button] == ["Create subject", "Load demo subject"]


def test_create_subject_via_form_sets_it_active(svc):
    at = open_page(svc, SUBJECTS)
    at.text_input[0].input("Natural Language Processing")
    at.text_area[1].input("Tokenization\nTF-IDF\n\n  Parsing  ")
    at.button[0].click().run()
    _no_errors(at)
    s = svc.subjects.list()[0]
    assert s.name == "Natural Language Processing" and s.topics == ["Tokenization", "TF-IDF", "Parsing"]
    assert at.session_state["active_subject_id"] == s.id


def test_duplicate_subject_name_shows_error_not_exception(svc):
    svc.subjects.create("Operating Systems")
    at = open_page(svc, SUBJECTS)
    at.text_input[0].input("operating systems")
    next(b for b in at.button if b.label == "Create subject").click().run()
    _no_errors(at)
    assert "already exists" in md(at) and len(svc.subjects.list()) == 1


def test_demo_subject_is_labelled_and_idempotent(svc):
    at = open_page(svc, SUBJECTS)
    at.button[1].click().run(); _no_errors(at)
    s = svc.subjects.list()[0]
    assert s.name == SAMPLE_SUBJECT_NAME and at.session_state["active_subject_id"] == s.id
    assert "Not real course content" in svc.subjects.get(s.id).description


def test_delete_subject_button_disabled_until_exact_name_typed(svc, world):
    at = open_page(svc, SUBJECTS, active=world["os"].id)
    btn = at.button(key=f"del_btn_{world['os'].id}")
    assert btn.disabled
    at.text_input(key=f"del_confirm_{world['os'].id}").input("operating systems").run()          # wrong case
    assert at.button(key=f"del_btn_{world['os'].id}").disabled
    at.text_input(key=f"del_confirm_{world['os'].id}").input("Operating Systems").run()
    at.button(key=f"del_btn_{world['os'].id}").click().run(); _no_errors(at)
    assert not svc.subjects.exists(world["os"].id) and svc.subjects.exists(world["nlp"].id)
    assert "active_subject_id" not in at.session_state or not at.session_state["active_subject_id"]


def test_deleted_active_subject_does_not_silently_switch(svc, world):
    at = open_page(svc, LIBRARY, active=world["os"].id)
    svc.subjects.delete(world["os"].id, "Operating Systems")
    at.run(); _no_errors(at)
    assert "Select a subject to continue" in md(at)          # NOT silently showing NLP


# ------------------------------------------------------------------ library gating
def test_library_without_subject_prompts_selection(svc, world):
    at = open_page(svc, LIBRARY); _no_errors(at)
    assert "Select a subject to continue" in md(at)
    assert world["nlp"].name not in md(at).split("Select a subject")[0]       # nothing from any subject is shown


# ------------------------------------------------------------------ upload flow
def _uploader_state(at, data, suffix):
    return at.session_state[f"up_0_{short_hash(data)}_{suffix}"]


def test_upload_preview_stores_nothing_until_confirmed(svc):
    s = svc.subjects.create("Natural Language Processing", description=NLP["description"], topics=NLP["topics"])
    data = NLP["files"]["notes.txt"].encode()
    at = open_page(svc, LIBRARY, active=s.id, files=[FakeUpload("week1.txt", data)]); _no_errors(at)
    assert "Automatic suggestion - not confirmed" in md(at) and "Extracted text preview" in [t.label for t in at.text_area]
    assert svc.library.list_documents(None) == []                           # preview only
    at.button(key="up_save_%s_0" % s.id).click().run(); _no_errors(at)
    docs = svc.library.list_documents(s.id)
    assert [d.filename for d in docs] == ["week1.txt"] and docs[0].doc_kind == "lecture_notes" and docs[0].status == "processed"


def test_suggestion_never_overrides_user_choice(svc):
    nlp = svc.subjects.create("Natural Language Processing", description=NLP["description"], topics=NLP["topics"])
    osub = svc.subjects.create("Operating Systems", description=OS["description"], topics=OS["topics"])
    svc.library.ingest(osub.id, "os.txt", OS["files"]["notes.txt"].encode(), doc_kind="lecture_notes")
    svc.library.ingest(nlp.id, "nlp.txt", NLP["files"]["notes.txt"].encode(), doc_kind="lecture_notes")
    data = b"# Deadlock lecture\nCoffman conditions, circular wait, banker algorithm and deadlock avoidance in processes."
    at = open_page(svc, LIBRARY, active=nlp.id, files=[FakeUpload("lecture7.txt", data)]); _no_errors(at)
    key = f"up_0_{short_hash(data)}_subj"
    assert "Operating Systems" in md(at) and "Suggested" in md(at)            # suggestion is shown...
    assert at.session_state[key] == nlp.id                                    # ...but the target is still the active subject
    at.button(key=f"{key}_use").click().run(); _no_errors(at)                 # user accepts the suggestion
    assert at.session_state[key] == osub.id
    at.button(key=f"up_save_{nlp.id}_0").click().run(); _no_errors(at)
    assert [d.filename for d in svc.library.list_documents(osub.id)].count("lecture7.txt") == 1
    assert "lecture7.txt" not in [d.filename for d in svc.library.list_documents(nlp.id)]


def test_malformed_and_unsupported_files_are_rejected_with_guidance(svc):
    s = svc.subjects.create("S")
    files = [FakeUpload("fake.pdf", b"this is not a pdf"), FakeUpload("empty.txt", b"")]
    at = open_page(svc, LIBRARY, active=s.id, files=files); _no_errors(at)
    assert "This file cannot be uploaded" in md(at) and "not a valid PDF" in md(at) and "is empty" in md(at)
    assert at.button(key=f"up_save_{s.id}_0").disabled and svc.library.list_documents(None) == []


def test_duplicate_requires_explicit_allowance(svc):
    s = svc.subjects.create("S")
    data = b"# H\nidentical content for duplicate handling"
    svc.library.ingest(s.id, "a.txt", data, doc_kind="lecture_notes")
    at = open_page(svc, LIBRARY, active=s.id, files=[FakeUpload("copy.txt", data)]); _no_errors(at)
    assert "Already in this subject" in md(at) and at.button(key=f"up_save_{s.id}_0").disabled
    at.checkbox(key=f"up_0_{short_hash(data)}_dup").check().run()
    assert not at.button(key=f"up_save_{s.id}_0").disabled


def test_unreadable_pdf_requires_explicit_save_anyway(svc):
    from pypdf import PdfWriter
    import io
    s = svc.subjects.create("S"); w = PdfWriter(); w.add_blank_page(72, 72); b = io.BytesIO(); w.write(b)
    data = b.getvalue()
    at = open_page(svc, LIBRARY, active=s.id, files=[FakeUpload("scan.pdf", data)]); _no_errors(at)
    assert "No usable text was extracted" in md(at) and at.button(key=f"up_save_{s.id}_0").disabled
    at.checkbox(key=f"up_0_{short_hash(data)}_anyway").check().run()
    at.button(key=f"up_save_{s.id}_0").click().run(); _no_errors(at)
    assert svc.library.list_documents(s.id)[0].status == "no_text"


def test_hostile_filename_and_text_are_escaped_in_html(svc):
    s = svc.subjects.create("<b>Evil</b> $x$")
    evil = "<img src=x onerror=alert(1)>.txt"
    svc.library.ingest(s.id, evil, b"# <script>alert(2)</script>\nbody words here for the section text", doc_kind="lecture_notes", title="<script>alert(3)</script>")
    at = open_page(svc, LIBRARY, active=s.id); _no_errors(at)
    page = md(at)
    for raw in ("<img src=x", "<script>alert", "<b>Evil</b>"):
        assert raw not in page
    assert "&lt;img src=x" in page and "&lt;script&gt;alert(3)" in page


# ------------------------------------------------------------------ library listing & destructive actions
def test_library_search_filter_and_empty_filter_state(svc):
    s = svc.subjects.create("S")
    svc.library.ingest(s.id, "alpha.txt", b"# T\nquantum words here", doc_kind="lecture_notes", title="Alpha notes")
    svc.library.ingest(s.id, "beta.txt", b"1. Explain things. [5 marks]\n2. Describe more. [5 marks]\n3. Define x. [5 marks]", doc_kind="question_paper", title="Beta paper")
    at = open_page(svc, LIBRARY, active=s.id); _no_errors(at)
    assert "Alpha notes" in md(at) and "Beta paper" in md(at)
    at.selectbox(key=f"lib_kind_{s.id}").select("question_paper").run()
    assert "Beta paper" in md(at) and "Alpha notes" not in md(at)
    at.text_input(key=f"lib_q_{s.id}").input("zzz-no-match").run()
    assert "No documents match these filters" in md(at)


def test_library_is_scoped_to_active_subject_unless_all_subjects_ticked(svc, world):
    at = open_page(svc, LIBRARY, active=world["nlp"].id); _no_errors(at)
    assert "NLP End-Sem 2023" in md(at) and "OS End-Sem 2023" not in md(at)
    at.checkbox(key=f"lib_all_{world['nlp'].id}").check().run()
    assert "OS End-Sem 2023" in md(at)


def test_document_delete_requires_confirmation_checkbox(svc, world):
    d = world["nlp_docs"][0]
    at = open_page(svc, LIBRARY, active=world["nlp"].id); _no_errors(at)
    assert at.button(key=f"rm_btn_{d.id}").disabled
    at.checkbox(key=f"rm_ok_{d.id}").check().run()
    at.button(key=f"rm_btn_{d.id}").click().run(); _no_errors(at)
    assert svc.library.list_documents(world["nlp"].id, kind="lecture_notes") == []
    assert svc.library.list_documents(world["os"].id, kind="lecture_notes")     # other subject untouched


def test_move_document_requires_acknowledgement_and_changes_subject(svc, world):
    d = world["nlp_docs"][0]
    at = open_page(svc, LIBRARY, active=world["nlp"].id); _no_errors(at)
    assert at.button(key=f"mv_btn_{d.id}").disabled
    at.selectbox(key=f"mv_{d.id}").select(world["os"].id).run()
    at.checkbox(key=f"mv_ok_{d.id}").check().run()
    at.button(key=f"mv_btn_{d.id}").click().run(); _no_errors(at)
    assert svc.library.get(d.id).subject_id == world["os"].id
