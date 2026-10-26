"""Primary workflow from the product spec, run end to end on SYNTHETIC data (tests/fixtures.py)."""
import csv
import io

from services import record_export as rx
from services.analysis_runner import AnalysisSettings
from services.study_plan_service import build_plan
from tests.fixtures import NLP, OS


def test_primary_workflow(svc):
    # 1. Create an NLP subject (with syllabus topics)
    nlp = svc.subjects.create("Natural Language Processing", description=NLP["description"], topics=NLP["topics"] + ["Hidden Markov models"])
    # 2. Upload NLP notes
    notes = svc.library.ingest(nlp.id, "notes.txt", NLP["files"]["notes.txt"].encode(), doc_kind="lecture_notes").document
    assert notes.status == "processed" and len(svc.library.sections_of(notes.id)) == 4
    # 3. Upload an NLP question paper
    p1 = svc.library.ingest(nlp.id, "p23.txt", NLP["paper1"].encode()).document      # kind auto-suggested
    assert p1.is_question_paper
    # 4. Run analysis
    out = svc.runner.run(nlp.id, [p1.id], name="NLP 2023")
    q = {x["display_id"]: x for x in out.record["questions"]}
    # 5. Relevant NLP sections returned
    assert "Tokenization" in q["Q1"]["matches"][0]["heading"]
    assert "TF-IDF" in q["Q2"]["matches"][0]["heading"]
    assert "parsing" in q["Q3"]["matches"][0]["heading"].lower()
    assert all(m["document_id"] == notes.id for x in q.values() for m in x["matches"])
    # 6. OS question paper in a separate subject
    osub = svc.subjects.create("Operating Systems", topics=OS["topics"])
    svc.library.ingest(osub.id, "notes.txt", OS["files"]["notes.txt"].encode(), doc_kind="lecture_notes")
    osp = svc.library.ingest(osub.id, "os.txt", OS["paper1"].encode(), doc_kind="question_paper").document
    # 7. NLP analysis still never retrieves OS material
    again = svc.runner.run(nlp.id, [p1.id], save=False)
    assert again.record["matrix"] == out.record["matrix"]
    assert {d["subject_id"] for d in again.record["documents"]} == {nlp.id}
    os_run = svc.runner.run(osub.id, [osp.id], save=False)
    assert {d["subject_id"] for d in os_run.record["documents"]} == {osub.id}
    # 8. Compare two NLP papers
    p2 = svc.library.ingest(nlp.id, "p24.txt", NLP["paper2"].encode(), doc_kind="question_paper").document
    cmp_out = svc.runner.run(nlp.id, [p1.id, p2.id], name="NLP 2023 vs 2024")
    comp = cmp_out.record["comparison"]
    assert len(comp["papers"]) == 2 and comp["pairwise"][0]["jaccard"] is not None
    repeated = {s["heading"] for s in comp["repeated_sections"]}
    assert any("TF-IDF" in h for h in repeated) and any("Tokenization" in h for h in repeated)
    assert "not forecasts" in comp["basis"]
    # Syllabus gap: "Hidden Markov models" is in no note section -> not assessable, never invented as covered
    hmm = next(t for t in cmp_out.record["topic_coverage"] if t["topic"] == "Hidden Markov models")
    assert hmm["status"] == "not_assessable"
    # 9. Export
    meta = {"id": cmp_out.analysis_id, "created_at": "t"}
    rows = list(csv.reader(io.StringIO(rx.matches_csv(cmp_out.record, meta))))
    assert rows[0][0].startswith("# SyllabusLens export")
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Paper")
    assert len(rows) - header_idx - 1 == 6 * 3          # 6 questions x top-3
    assert "Method:" in rx.coverage_csv(cmp_out.record, meta) and "thresholds" in rx.coverage_csv(cmp_out.record, meta)
    assert "Papers Hit" in rx.comparison_csv(cmp_out.record, meta)
    assert "Hidden Markov models" in rx.topics_csv(cmp_out.record, meta)
    assert "Limitations" in rx.markdown_report(cmp_out.record, meta)
    # 10. Revisit the saved analysis: identical to what was computed, available without recomputation
    saved = svc.history.get(cmp_out.analysis_id)
    assert saved["record"]["matrix"] == cmp_out.record["matrix"]
    assert saved["subject_id"] == nlp.id and saved["name"] == "NLP 2023 vs 2024"
    assert {p["document_id"] for p in saved["inputs"]["papers"]} == {p1.id, p2.id}
    assert svc.history.input_status(cmp_out.analysis_id)["status"] == "current"
    assert len(svc.history.list(nlp.id)) == 2 and svc.history.list(osub.id) == []


def test_history_is_immutable_when_inputs_change(svc, world):
    out = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], name="before")
    snapshot_before = svc.history.get(out.analysis_id)["record"]
    svc.library.replace(world["nlp_docs"][0].id, "notes.txt", b"# Totally different\nunrelated replacement text here")
    svc.library.move_to_subject(world["nlp_p1"].id, world["os"].id)
    snapshot_after = svc.history.get(out.analysis_id)["record"]
    assert snapshot_after == snapshot_before                                   # stored results never change
    status = svc.history.input_status(out.analysis_id)
    assert status["status"] == "inputs_changed" and status["changes"]          # but staleness is reported


def test_study_plan_rules_carry_evidence_and_review_state(svc, world):
    out = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id, world["nlp_p2"].id])
    plan = build_plan(out.record)
    assert plan and all(r["evidence"] and r["rule_id"] and r["rule"] for r in plan)
    assert not any("likely to appear" in (r["title"] + r["rule"]).lower() for r in plan)   # no forecasts
    sec = next(r["section_id"] for r in plan if r["section_id"])
    svc.history.mark_reviewed(world["nlp"].id, sec)
    assert any(r["reviewed"] for r in build_plan(out.record, svc.history.reviewed_ids(world["nlp"].id)))
    assert "Reviewed" in rx.checklist_csv(out.record, svc.history.reviewed_ids(world["nlp"].id))


def test_csv_export_neutralises_formula_injection(svc):
    s = svc.subjects.create("Inj")
    svc.library.ingest(s.id, "n.txt", b"# Heading\ntokenization of text and words", doc_kind="lecture_notes")
    p = svc.library.ingest(s.id, "p.txt", b"1. =HYPERLINK(\"http://evil\") tokenization of text [5 marks]\n2. Describe words.", doc_kind="question_paper").document
    out = svc.runner.run(s.id, [p.id], save=False)
    assert ",'=HYPERLINK" in rx.matches_csv(out.record).replace('"', "") or "'=HYPERLINK" in rx.matches_csv(out.record)


def test_settings_validation_and_bow_runs(svc, world):
    import pytest
    with pytest.raises(ValueError):
        svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], settings=AnalysisSettings(strong_threshold=0.1), save=False)
    out = svc.runner.run(world["nlp"].id, [world["nlp_p1"].id], settings=AnalysisSettings(vectorizer="bow"), save=False)
    assert "Bag-of-Words" in out.record["method"]
