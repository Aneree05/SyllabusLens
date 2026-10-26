"""
Regression tests for Question Categorization, Practice Mode, Numerical Solver, and Diagrams.
Verifies honest provenance (PYQ vs synthetic), deterministic calculations, missing data alerts,
and grounded answer outlines with source citations.
"""
from __future__ import annotations

import pytest
from nlp.question_classifier import (
    QuestionClassifier,
    CAT_SHORT_ANSWER,
    CAT_LONG_ANSWER,
    CAT_COMPARISON,
    CAT_DERIVATION,
    CAT_NUMERICAL,
    CAT_DIAGRAM,
)
from services.numerical_solver import NumericalSolver
from services.diagram_service import DiagramService
from services.question_practice_service import QuestionPracticeService
from services import record_export as rx


def test_question_classifier_all_six_categories():
    qc = QuestionClassifier()

    # 1. Short-Answer & Definition
    r1 = qc.classify("Define the Bag of Words (BoW) model.", "2 marks")
    assert r1.category == CAT_SHORT_ANSWER
    assert "definition" in r1.rationale.lower() or "define" in r1.rationale.lower()

    # 2. Long-Answer & Theory
    r2 = qc.classify("Explain in detail the architecture and forward computation of Recurrent Neural Networks.", "10 marks")
    assert r2.category == CAT_LONG_ANSWER

    # 3. Comparison & Distinction
    r3 = qc.classify("Differentiate between Stemming and Lemmatization with examples.", "5 marks")
    assert r3.category == CAT_COMPARISON
    assert "differentiate" in r3.rationale.lower()

    # 4. Derivation & Proof
    r4 = qc.classify("Formulate the mathematical equation for IDF and derive its lower bound.", "5 marks")
    assert r4.category == CAT_DERIVATION

    # 5. Numerical Problem
    r5 = qc.classify("Calculate the IDF of a term that appears in 10 documents given that |D| = 100.", "5 marks")
    assert r5.category == CAT_NUMERICAL
    assert "calculate" in r5.rationale.lower() or "numerical" in r5.rationale.lower()

    # 6. Diagram-Based
    r6 = qc.classify("Draw a neat labelled diagram of the LSTM cell architecture and explain its gates.", "10 marks")
    assert r6.category == CAT_DIAGRAM
    assert "diagram" in r6.rationale.lower() or "draw" in r6.rationale.lower()


def test_numerical_solver_idf_with_steps_and_verification():
    solver = NumericalSolver()
    q = "Calculate the IDF value of term 'tokenization' if total documents |D| = 100 and document frequency DF = 10."
    sol = solver.analyze_and_solve(q)

    assert sol is not None
    assert sol.is_solvable
    assert sol.given_data["Total documents (|D|)"] == 100
    assert sol.given_data["Document frequency (DF)"] == 10
    assert "IDF(t) = ln((1 + |D|) / (1 + DF(t))) + 1" in sol.formula
    assert len(sol.steps) >= 5
    assert sol.final_answer is not None
    # (1+100)/(1+10) = 101/11 = 9.1818; ln(9.1818) + 1 ≈ 2.2172 + 1 = 3.2172
    assert "3.2172" in sol.final_answer
    assert "Sanity verification" in sol.verification


def test_numerical_solver_cosine_similarity():
    solver = NumericalSolver()
    q = "Compute the cosine similarity between vector q = [1, 2, 0] and document vector d = [2, 1, 0]."
    sol = solver.analyze_and_solve(q)

    assert sol is not None
    assert sol.is_solvable
    assert sol.given_data["Vector 1 (q)"] == [1.0, 2.0, 0.0]
    assert sol.given_data["Vector 2 (d)"] == [2.0, 1.0, 0.0]
    # dot = 4; norm1 = sqrt(5); norm2 = sqrt(5); cos = 4/5 = 0.8
    assert "0.8000" in sol.final_answer
    assert sol.verification is not None


def test_numerical_solver_missing_data_honesty_never_invents_numbers():
    solver = NumericalSolver()
    # Problem prompt requests calculation but does not provide numbers
    q = "Calculate the Inverse Document Frequency (IDF) of the word across the course corpus."
    sol = solver.analyze_and_solve(q)

    assert sol is not None
    assert not sol.is_solvable
    assert sol.final_answer is None
    # Must explicitly list what is missing
    assert len(sol.missing_data) >= 2
    assert any("|D|" in m for m in sol.missing_data)
    assert any("DF(t)" in m for m in sol.missing_data)


def test_diagram_service_deterministic_graphviz_models():
    diag_svc = DiagramService()

    # Transformer
    d_trans = diag_svc.get_diagram_for_text("Explain Transformer self-attention and multi-head dot product.")
    assert d_trans is not None
    assert "digraph TransformerAttention" in d_trans.dot_source
    assert "Scaled Dot Product" in d_trans.dot_source

    # LSTM
    d_lstm = diag_svc.get_diagram_for_text("Draw the gating mechanism of LSTM cells.")
    assert d_lstm is not None
    assert "Forget Gate" in d_lstm.dot_source
    assert "Cell Memory Carousel" in d_lstm.dot_source

    # Cosine
    d_cos = diag_svc.get_diagram_for_text("What is Cosine Similarity distance in vector space?")
    assert d_cos is not None
    assert "digraph CosineSim" in d_cos.dot_source


def test_question_practice_separates_pyq_from_synthetic(svc, world):
    qp_svc = svc.question_practice
    nlp_sub = world["nlp"]
    p1 = world["nlp_p1"]

    # 1. PYQ analysis
    out = svc.runner.run(nlp_sub.id, [p1.id], save=False)
    pyqs = qp_svc.prepare_pyq_practice(out.record["questions"])
    assert len(pyqs) > 0
    for q in pyqs:
        assert q.is_pyq is True
        assert "Past Paper:" in q.source_label
        assert q.category in (CAT_SHORT_ANSWER, CAT_LONG_ANSWER, CAT_COMPARISON, CAT_DERIVATION, CAT_NUMERICAL, CAT_DIAGRAM)

    # 2. Generated practice questions
    gen_qs = qp_svc.generate_practice_questions(nlp_sub.id)
    assert len(gen_qs) > 0
    for q in gen_qs:
        assert q.is_pyq is False
        assert "Generated Practice Question" in q.source_label
        assert "Not from Past Papers" in q.source_label
        # Grounded in notes
        assert q.answer_outline.is_grounded is True
        assert len(q.answer_outline.citations) > 0


def test_grounded_answer_outline_and_insufficient_data_notice(svc, world):
    from services.sample_data_service import install_sample_subject

    nlp_sub = install_sample_subject(svc)
    p1 = next(d for d in svc.library.list_documents(nlp_sub.id) if d.is_question_paper)
    out = svc.runner.run(nlp_sub.id, [p1.id], save=False)
    questions = out.record["questions"]

    # Question on BoW (Question 1a in sample_question_paper)
    bow_q = next(q for q in questions if "bag of words" in q["text"].lower())
    assert bow_q["answer_outline"]["is_grounded"] is True
    assert len(bow_q["answer_outline"]["citations"]) > 0
    assert bow_q["answer_outline"]["core_answer"] != ""

    # Question 7 is Quantum Computing Shor's algorithm (Unmatched in sample notes)
    unmatched_q = next(q for q in questions if q["display_id"] == "Q7")
    assert unmatched_q["best_tier"] == "Unmatched"
    assert unmatched_q["answer_outline"]["is_grounded"] is False
    assert unmatched_q["answer_outline"]["insufficient_reason"] is not None


def test_practice_questions_export_csv(svc, world):
    nlp_sub = world["nlp"]
    p1 = world["nlp_p1"]
    out = svc.runner.run(nlp_sub.id, [p1.id], save=False)

    csv_data = rx.practice_questions_csv(out.record)
    assert "Question ID" in csv_data
    assert "Category" in csv_data
    assert "Core Answer" in csv_data
    assert "Formula" in csv_data
    assert "Missing Material" in csv_data
