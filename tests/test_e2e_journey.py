"""
Acceptance journey, driven through the real pages (AppTest). Numbers refer to the Phase E acceptance list.
All documents are SYNTHETIC (tests/fixtures.py).
"""
from services import record_export as rx
from tests.fixtures import NLP, OS
from tests.ui_harness import FakeUpload, md, open_page, short_hash

SUBJ, LIB, ANA, EXP, COV, HIS, PLAN = (f"pages_content.{m}" for m in ("subjects", "library", "analysis", "explorer", "coverage", "history", "planner"))
TRAP = b"Trap paper\n1. Explain virtual memory paging and page replacement using LRU. [10 marks]\n2. Explain tokenization and why whitespace tokenization fails on punctuation. [5 marks]\n"


def ok(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


def create_subject(svc, name, topics=""):
    at = open_page(svc, SUBJ)
    at.text_input[0].input(name)
    if topics:
        at.text_area[1].input(topics)
    next(b for b in at.button if b.label == "Create subject").click().run(); ok(at)
    return svc.subjects.list()[-1] if len(svc.subjects.list()) == 1 else next(s for s in svc.subjects.list() if s.name == name)


def upload(svc, subject_id, files):
    at = open_page(svc, LIB, active=subject_id, files=files); ok(at)
    assert "Automatic suggestion - not confirmed" in md(at)                      # (3) suggestion is shown, labelled unconfirmed
    at.button(key=f"up_save_{subject_id}_0").click().run(); ok(at)
    return at


def test_full_user_journey(svc):
    # (1) first-time user creates an NLP subject
    assert "Create your first subject" in md(open_page(svc, SUBJ))
    nlp = create_subject(svc, "Natural Language Processing", "Tokenization\nTF-IDF weighting\nSyntactic parsing\nWord embeddings")
    # (2)+(3) upload notes and a paper; kinds are auto-detected, assignment is confirmed by the Save click
    upload(svc, nlp.id, [FakeUpload("nlp_notes.txt", NLP["files"]["notes.txt"].encode()), FakeUpload("nlp_2023.txt", NLP["paper1"].encode())])
    docs = {d.filename: d for d in svc.library.list_documents(nlp.id)}
    assert docs["nlp_notes.txt"].doc_kind == "lecture_notes" and docs["nlp_2023.txt"].doc_kind == "question_paper"
    assert all(d.subject_id == nlp.id for d in docs.values())

    # (4) analysis uses only eligible NLP documents
    at = open_page(svc, ANA, active=nlp.id)
    at.multiselect(key=f"an_papers_{nlp.id}").set_value([docs["nlp_2023.txt"].id]).run()
    at.button(key=f"an_run_{nlp.id}").click().run(); ok(at)
    saved = svc.history.list(nlp.id)[0]
    rec = svc.history.get(saved["id"])["record"]
    assert {d["subject_id"] for d in rec["documents"]} == {nlp.id} and "Scope: Natural Language Processing only" in md(at)
    # (5) questions and matching sections can be inspected
    ex = open_page(svc, EXP, active=nlp.id); ok(ex)
    assert "Shared terms driving this score" in md(ex) and "Tokenization" in md(ex)
    # (6) coverage and charts reflect the real results
    cv = open_page(svc, COV, active=nlp.id); ok(cv)
    assert len(cv.get("plotly_chart")) >= 3 and f'{rec["metrics"]["total_sections"]}' in md(cv)

    # (7) OS subject + notes through the UI
    osub = create_subject(svc, "Operating Systems", "Deadlocks\nVirtual memory and paging")
    assert at.session_state is not None and svc.subjects.exists(osub.id)
    upload(svc, osub.id, [FakeUpload("os_notes.txt", OS["files"]["notes.txt"].encode())])
    os_doc_ids = {d.id for d in svc.library.list_documents(osub.id)}
    assert len(os_doc_ids) == 1

    # (8) NLP analysis with an adversarial paging question never retrieves OS sections
    trap = svc.library.ingest(nlp.id, "trap.txt", TRAP, doc_kind="question_paper").document
    at = open_page(svc, ANA, active=nlp.id)
    at.multiselect(key=f"an_papers_{nlp.id}").set_value([trap.id]).run()
    at.button(key=f"an_run_{nlp.id}").click().run(); ok(at)
    newest = svc.history.get(svc.history.list(nlp.id)[0]["id"])["record"]
    assert not ({m["document_id"] for q in newest["questions"] for m in q["matches"]} & os_doc_ids)

    # (9) explicit cross-subject works only when requested, and is labelled
    at.toggle(key=f"an_cross_{nlp.id}").set_value(True).run()
    assert at.button(key=f"an_run_{nlp.id}").disabled                           # needs explicit subject choice
    at.multiselect(key=f"an_extra_{nlp.id}").set_value([osub.id]).run()
    at.button(key=f"an_run_{nlp.id}").click().run(); ok(at)
    cross = svc.history.get(svc.history.list(nlp.id)[0]["id"])
    assert cross["cross_subject"] and set(cross["record"]["settings"]["subjects_in_corpus"]) == {nlp.id, osub.id}
    assert "Cross-subject analysis" in md(at)

    # (10) switching subject shows no stale results
    sw = open_page(svc, ANA, active=nlp.id)
    sw.session_state[f"scoped.{nlp.id}.last_analysis"] = saved["id"]; sw.run()
    assert "Scope: Natural Language Processing only" in md(sw)
    sw.session_state["active_subject_id"] = osub.id; sw.run(); ok(sw)
    assert "Scope: Natural Language Processing" not in md(sw) and "Tokenization" not in md(sw)

    # (11) reclassification does not mutate the saved snapshot, and history says so
    before = svc.history.get(saved["id"])["record"]
    svc.library.move_to_subject(docs["nlp_notes.txt"].id, osub.id)
    assert svc.history.get(saved["id"])["record"] == before
    h = open_page(svc, HIS, active=nlp.id); ok(h)
    assert "Inputs changed since" in md(h)

    # (12) exports retain injection protection
    inj = svc.subjects.create("Injection")
    svc.library.ingest(inj.id, "n.txt", b"# Heading\ntokenization of text and words", doc_kind="lecture_notes")
    pp = svc.library.ingest(inj.id, "p.txt", b'1. =HYPERLINK("http://evil") tokenization of text [5 marks]\n2. Describe words.', doc_kind="question_paper").document
    out = svc.runner.run(inj.id, [pp.id])
    csv_text = rx.matches_csv(out.record)
    assert "'=HYPERLINK" in csv_text and "\n=HYPERLINK" not in csv_text and ",=HYPERLINK" not in csv_text
    page = open_page(svc, ANA, active=inj.id)
    page.session_state[f"scoped.{inj.id}.last_analysis"] = out.analysis_id; page.run(); ok(page)
    labels = [b.proto.label for b in page.get("download_button")]
    assert "Question matches (CSV)" in labels
