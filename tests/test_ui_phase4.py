"""Explorer, Coverage Lab, Compare, Planner and History pages against real analysis snapshots. Synthetic/demo data."""
import pytest

from services.sample_data_service import install_sample_subject
from tests.ui_harness import md, open_page

PAGES = {k: f"pages_content.{k}" for k in ("explorer", "coverage", "compare", "planner", "history")}


def _ok(at):
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.fixture
def two_paper_run(svc, world):
    return world, svc.runner.run(world["nlp"].id, [world["nlp_p1"].id, world["nlp_p2"].id], name="NLP both papers")


@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_shows_empty_state_when_subject_has_no_analysis(svc, world, page):
    at = open_page(svc, PAGES[page], active=world["nlp"].id); _ok(at)
    page_md = md(at)
    if page == "planner":
        assert any("Topic-wise Revision Notes" in item.value for item in at.subheader)
        return
    assert ("Nothing to show yet" in page_md or "No saved analyses" in page_md or "Two question papers" in page_md or "needed" in page_md) \
        and "Traceback" not in page_md


@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_requires_a_subject(svc, world, page):
    at = open_page(svc, PAGES[page]); _ok(at)
    assert "Select a subject to continue" in md(at)


@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_renders_with_a_real_two_paper_analysis(svc, two_paper_run, page):
    world, out = two_paper_run
    at = open_page(svc, PAGES[page], active=world["nlp"].id); _ok(at)
    assert "Scope" in md(at) or "Subject scope" in md(at)
    assert "Operating Systems" not in md(at).replace("Other subjects are excluded", "")


def test_explorer_shows_evidence_and_terms(svc, two_paper_run):
    world, out = two_paper_run
    at = open_page(svc, PAGES["explorer"], active=world["nlp"].id); _ok(at)
    page = md(at)
    assert "Shared terms driving this score" in page and "does not prove" in page.replace("a high score does not prove", "does not prove") or "not understanding" in page
    assert "Section text (current document, unchanged since the analysis)" in [e.label for e in at.expander]


def test_explorer_falls_back_to_snapshot_excerpt_when_document_changed(svc, two_paper_run):
    world, out = two_paper_run
    svc.library.replace(world["nlp_docs"][0].id, "notes.txt", b"# Different\ncompletely replaced content here")
    at = open_page(svc, PAGES["explorer"], active=world["nlp"].id); _ok(at)
    assert "Source documents changed after this analysis" in md(at)
    assert any("excerpt saved with the analysis" in e.label for e in at.expander)


def test_explorer_filters_narrow_the_question_list(svc, two_paper_run):
    world, out = two_paper_run
    at = open_page(svc, PAGES["explorer"], active=world["nlp"].id)
    at.text_input(key=f"ex_text_{out.analysis_id}").input("zzzz-nothing").run(); _ok(at)
    assert "No questions match these filters" in md(at)


def test_coverage_lab_charts_and_topic_statuses_on_demo_data(svc):
    sub = install_sample_subject(svc)
    paper = next(d for d in svc.library.list_documents(sub.id) if d.is_question_paper)
    out = svc.runner.run(sub.id, [paper.id])
    at = open_page(svc, PAGES["coverage"], active=sub.id); _ok(at)
    assert len(at.get("plotly_chart")) >= 3                       # heatmap, section bars, topic bars (+ sankey)
    page = md(at)
    assert "Not examined" in page or "not examined" in page
    topics = {t["topic"]: t["status"] for t in out.record["topic_coverage"]}
    assert "Quantum computing algorithms" not in topics            # Q7 is out of syllabus, so it is not a syllabus topic
    assert topics["TF-IDF weighting"] == "detected" and topics["Cosine similarity"] == "detected"
    assert topics["Precision and recall metrics"] == "not_detected"
    assert topics["Machine translation"] == "not_assessable"
    unmatched = [q for q in out.record["questions"] if q["best_tier"] == "Unmatched"]
    assert any(q["display_id"] == "Q7" for q in unmatched)        # the out-of-syllabus quantum question


def test_compare_requires_papers_of_same_subject_and_shows_basis(svc, two_paper_run):
    world, out = two_paper_run
    at = open_page(svc, PAGES["compare"], active=world["nlp"].id); _ok(at)
    page = md(at)
    assert "not forecasts" in page and "Papers compared" in page
    sel = at.multiselect(key=f"cmp_sel_{world['nlp'].id}")
    opts = " ".join(sel.options)
    assert "NLP End-Sem 2023" in opts and "NLP End-Sem 2024" in opts and "OS End-Sem" not in opts   # other subject's paper is not offered


def test_compare_run_form_creates_a_saved_comparison(svc, world):
    sid = world["nlp"].id
    at = open_page(svc, PAGES["compare"], active=sid); _ok(at)
    at.multiselect(key=f"cmp_sel_{sid}").set_value([world["nlp_p1"].id, world["nlp_p2"].id]).run()
    at.button(key=f"cmp_run_{sid}").click().run(); _ok(at)
    saved = svc.history.list(sid)
    assert len(saved) == 1 and saved[0]["inputs"]["papers"].__len__() == 2 and "Papers compared" in md(at)


def test_planner_checklist_persists_review_state_per_subject(svc, two_paper_run):
    world, out = two_paper_run
    sid = world["nlp"].id
    at = open_page(svc, PAGES["planner"], active=sid); _ok(at)
    assert "does not predict future exams" in md(at)
    boxes = [c for c in at.checkbox if c.key and c.key.startswith("rev_")]
    assert boxes
    boxes[0].check().run(); _ok(at)
    assert len(svc.history.reviewed_ids(sid)) == 1 and svc.history.reviewed_ids(world["os"].id) == set()
    again = open_page(svc, PAGES["planner"], active=sid)
    assert any(c.value for c in again.checkbox if c.key and c.key.startswith("rev_"))


def test_planner_shows_selectable_topic_notes_before_analysis(svc, world):
    sid = world["nlp"].id
    at = open_page(svc, PAGES["planner"], active=sid); _ok(at)
    page = md(at)
    assert any("Topic-wise Revision Notes" in item.value for item in at.subheader)
    assert at.selectbox(key=f"plan_standalone_topic_{sid}").options


def test_explorer_generated_practice_questions_render_without_historical_match_fields(svc, two_paper_run):
    world, out = two_paper_run
    at = open_page(svc, PAGES["explorer"], active=world["nlp"].id); _ok(at)
    at.radio(key=f"ex_mode_{out.analysis_id}").set_value("Generated Practice Questions").run(); _ok(at)
    page = md(at)
    assert "Generated Practice Question (Not from Past Papers)" in page
    assert any(e.label.startswith("Grounded Answer Outline") for e in at.expander)
    assert any("matching scores are only available for historical questions" in c.value for c in at.caption)


def test_explorer_keeps_historical_paper_year_and_session(svc, world):
    svc.library.update_metadata(world["nlp_p1"].id, academic_year="2024", exam_session="Spring")
    out = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id])
    assert all(q["academic_year"] == "2024" and q["exam_session"] == "Spring" for q in out.record["questions"])
    at = open_page(svc, PAGES["explorer"], active=world["nlp"].id); _ok(at)
    assert any("Historical source:" in c.value and "2024 · Spring" in c.value for c in at.caption)


def test_planner_never_claims_prediction(svc, two_paper_run):
    world, out = two_paper_run
    at = open_page(svc, PAGES["planner"], active=world["nlp"].id); _ok(at)
    low = md(at).lower()
    for banned in ("likely to appear", "will appear", "probability of", "guarantee marks", "predicted"):
        assert banned not in low.replace("does not predict future exams or guarantee marks", "")


def test_history_lists_opens_and_keeps_snapshot_after_reclassification(svc, two_paper_run):
    world, out = two_paper_run
    sid = world["nlp"].id
    before = svc.history.get(out.analysis_id)["record"]
    at = open_page(svc, PAGES["history"], active=sid); _ok(at)
    assert "NLP both papers" in md(at) and "Inputs unchanged" in md(at)
    svc.library.move_to_subject(world["nlp_docs"][0].id, world["os"].id)
    at.run(); _ok(at)
    assert "Inputs changed since" in md(at)
    assert svc.history.get(out.analysis_id)["record"] == before
    at.button(key=f"hist_open_{out.analysis_id}").click().run(); _ok(at)
    assert "Snapshot: NLP both papers" in md(at) and "Source documents changed after this analysis" in md(at)


def test_history_is_scoped_to_active_subject(svc, two_paper_run):
    world, out = two_paper_run
    at = open_page(svc, PAGES["history"], active=world["os"].id); _ok(at)
    assert "No saved analyses" in md(at) and "NLP both papers" not in md(at)
    at.session_state["scoped.%s.history_open" % world["os"].id] = out.analysis_id      # foreign pointer
    at.run(); _ok(at)
    assert "Snapshot:" not in md(at)


def test_history_delete_removes_only_the_snapshot(svc, two_paper_run):
    world, out = two_paper_run
    at = open_page(svc, PAGES["history"], active=world["nlp"].id)
    at.button(key=f"hist_del_{out.analysis_id}").click().run(); _ok(at)
    assert svc.history.list(world["nlp"].id) == [] and svc.library.list_documents(world["nlp"].id)


def test_single_generic_word_cannot_make_a_topic_look_detected(svc):
    """Regression: 'Sentiment analysis' matched Q2.b only through the word 'analysis' (notes never mention sentiment)."""
    sub = install_sample_subject(svc)
    svc.subjects.update(sub.id, topics=["Sentiment analysis", "Cosine similarity"])
    paper = next(d for d in svc.library.list_documents(sub.id) if d.is_question_paper)
    rec = svc.runner.run(sub.id, [paper.id], save=False).record
    t = {x["topic"]: x for x in rec["topic_coverage"]}
    assert t["Sentiment analysis"]["status"] == "not_assessable" and t["Sentiment analysis"]["vocab_coverage"] <= 0.5
    assert t["Cosine similarity"]["status"] == "detected" and t["Cosine similarity"]["vocab_coverage"] == 1.0


def test_hyphenated_topics_use_the_vectorizer_tokenisation(svc):
    sub = install_sample_subject(svc)
    svc.subjects.update(sub.id, topics=["Part-of-speech tagging", "Self-attention"])
    paper = next(d for d in svc.library.list_documents(sub.id) if d.is_question_paper)
    rec = svc.runner.run(sub.id, [paper.id], save=False).record
    assert {x["topic"]: x["vocab_coverage"] for x in rec["topic_coverage"]} == {"Part-of-speech tagging": 1.0, "Self-attention": 1.0}


def test_below_threshold_sections_are_never_presented_as_matches(svc):
    """Regression: unmatched questions still carry their top-3 sections (scores like 0.05); the UI must call them reference-only."""
    sub = install_sample_subject(svc)
    paper = next(d for d in svc.library.list_documents(sub.id) if d.is_question_paper)
    out = svc.runner.run(sub.id, [paper.id])
    q7 = next(q for q in out.record["questions"] if q["display_id"] == "Q7")
    assert q7["best_tier"] == "Unmatched" and q7["matches"]
    at = open_page(svc, PAGES["explorer"], active=sub.id)
    at.selectbox(key=f"ex_q_{out.analysis_id}").select(q7["question_id"]).run(); _ok(at)
    page = md(at)
    assert "shown for reference only" in page and "not matches" in page
    assert "Best section:" not in page.split("Unmatched")[1][:600] if "Best section:" in page else True
    from components import ui
    assert "No match. Closest section, shown for reference only" in ui.question_card_html(q7)
