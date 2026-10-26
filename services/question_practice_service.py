"""
Categorized Question Practice Service for SyllabusLens.
Manages previous-year questions (PYQs) and generated practice questions.
Categorises questions into 6 categories, provides grounded answer outlines with source
citations, identifies insufficient study material, and connects with numerical solver and diagrams.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nlp.question_classifier import QuestionClassifier, ClassificationResult, CAT_NUMERICAL
from services.diagram_service import DiagramService, DiagramSpec
from services.numerical_solver import NumericalSolution, NumericalSolver
from services.retrieval_service import Corpus, RetrievalService
from services.subject_service import SubjectService


@dataclass
class GroundedAnswerOutline:
    is_grounded: bool
    core_answer: str
    key_points: List[str]
    formula: Optional[str] = None
    citations: List[Dict[str, str]] = field(default_factory=list)
    insufficient_material_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "core_answer": self.core_answer,
            "key_points": self.key_points,
            "formula": self.formula,
            "citations": self.citations,
            "insufficient_material_reason": self.insufficient_material_reason,
        }


@dataclass
class PracticeQuestion:
    question_id: str
    display_id: str
    text: str
    marks: Optional[str]
    category: str
    category_name: str
    category_rationale: str
    is_pyq: bool                        # True = Extracted Past Paper; False = Generated Practice
    source_label: str                   # e.g., "End-Sem 2024 [Paper]" or "Generated Practice Question"
    paper_doc_id: Optional[str] = None
    year: Optional[str] = None
    answer_outline: Optional[GroundedAnswerOutline] = None
    numerical_solution: Optional[NumericalSolution] = None
    diagram: Optional[DiagramSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "display_id": self.display_id,
            "text": self.text,
            "marks": self.marks or "",
            "category": self.category,
            "category_name": self.category_name,
            "category_rationale": self.category_rationale,
            "is_pyq": self.is_pyq,
            "source_label": self.source_label,
            "paper_doc_id": self.paper_doc_id,
            "year": self.year,
            "answer_outline": self.answer_outline.to_dict() if self.answer_outline else None,
            "numerical_solution": self.numerical_solution.to_dict() if self.numerical_solution else None,
            "diagram": self.diagram.title if self.diagram else None,
        }


class QuestionPracticeService:
    def __init__(
        self,
        subjects: SubjectService,
        retrieval: RetrievalService,
        classifier: Optional[QuestionClassifier] = None,
        solver: Optional[NumericalSolver] = None,
        diagrams: Optional[DiagramService] = None,
    ):
        self.subjects = subjects
        self.retrieval = retrieval
        self.classifier = classifier or QuestionClassifier()
        self.solver = solver or NumericalSolver()
        self.diagrams = diagrams or DiagramService()

    def build_grounded_answer(
        self,
        question_text: str,
        top_matches: Sequence[Any],
        weak_threshold: float = 0.08,
    ) -> GroundedAnswerOutline:
        """
        Builds a structured answer outline grounded strictly in matching note sections.
        Where evidence is below threshold, explains what is missing rather than inventing facts.
        """
        valid_matches = [m for m in top_matches if getattr(m, "score", getattr(m, "similarity_score", 0.0)) >= weak_threshold]

        if not valid_matches:
            return GroundedAnswerOutline(
                is_grounded=False,
                core_answer="",
                key_points=[],
                formula=None,
                citations=[],
                insufficient_material_reason=(
                    "Uploaded study material does not contain sufficient coverage for this question. "
                    "Vocabulary similarity fell below the weak match threshold. "
                    "Recommend consulting recommended textbooks for full syllabus mastery."
                ),
            )

        citations = []
        key_points = []
        core_answer = ""
        formula = None

        for m in valid_matches[:3]:
            doc_name = getattr(m, "doc_name", "")
            locator = getattr(m, "locator", getattr(m, "page_or_slide", ""))
            heading = getattr(m, "heading", "")
            sec_id = getattr(m, "section_id", "")
            snippet = getattr(m, "snippet", getattr(m, "section_snippet", ""))

            citations.append({
                "doc_name": doc_name,
                "locator": locator,
                "heading": heading,
                "section_id": sec_id,
            })

            # Extract distinct sentences
            sentences = [s.strip() for s in snippet.replace("\n", " ").split(".") if len(s.strip()) > 15]
            for s in sentences:
                if not core_answer and any(w in s.lower() for w in ("is defined as", "refers to", "is a", "denotes")):
                    core_answer = s + "."
                elif len(key_points) < 5 and s not in key_points:
                    key_points.append(s + ".")

            if not formula and any(c in snippet for c in ("=", "≈", "IDF", "Sim", "tanh")):
                for line in snippet.splitlines():
                    if any(c in line for c in ("=", "≈", "∑", "√")) and len(line.strip()) < 120:
                        formula = line.strip()
                        break

        if not core_answer and key_points:
            core_answer = key_points.pop(0)

        return GroundedAnswerOutline(
            is_grounded=True,
            core_answer=core_answer or "Relevant supporting discussion found in uploaded notes.",
            key_points=key_points,
            formula=formula,
            citations=citations,
            insufficient_material_reason=None,
        )

    def prepare_pyq_practice(self, record_questions: List[Dict[str, Any]]) -> List[PracticeQuestion]:
        """Annotates extracted exam questions with category, grounded answer, solver, and diagrams."""
        out = []
        for q in record_questions:
            text = q.get("text", "")
            marks = q.get("marks", "")
            cls_res: ClassificationResult = self.classifier.classify(text, marks)

            # Grounded answer outline
            matches = q.get("matches", [])
            answer = self.build_grounded_answer(text, matches)

            # Numerical solver
            num_sol = self.solver.analyze_and_solve(text) if cls_res.category == CAT_NUMERICAL or any(w in text.lower() for w in ("calculate", "compute", "value")) else None

            # Diagram matching
            diag = self.diagrams.get_diagram_for_text(text)

            out.append(PracticeQuestion(
                question_id=q.get("question_id", ""),
                display_id=q.get("display_id", q.get("question_id", "")),
                text=text,
                marks=marks,
                category=cls_res.category,
                category_name=cls_res.category_name,
                category_rationale=cls_res.rationale,
                is_pyq=True,
                source_label=f"Past Paper: {q.get('paper_label', 'Exam Paper')}",
                paper_doc_id=q.get("paper_doc_id"),
                year=q.get("paper_label"),
                answer_outline=answer,
                numerical_solution=num_sol,
                diagram=diag,
            ))
        return out

    def generate_practice_questions(self, subject_id: str) -> List[PracticeQuestion]:
        """
        Generates synthetic practice questions deterministically from uploaded lecture notes.
        Explicitly marked as generated practice questions (never fabricating past paper provenance).
        """
        corpus: Corpus = self.retrieval.build_note_corpus(subject_id)
        sections = list(corpus.sections)
        questions: List[PracticeQuestion] = []
        seen_prompts = set()

        for sec in sections:
            lines = [l.strip() for l in (sec.text or "").splitlines() if l.strip()]
            section_question_start = len(questions)
            for line in lines:
                # 1. Definition question check
                if any(w in line.lower() for w in ("is defined as", "refers to as", "is a technique")):
                    term_match = line.split("is defined as")[0] if "is defined as" in line else line.split("refers to")[0]
                    term = " ".join(term_match.split()[-4:]).strip(" :.,-")
                    if 3 <= len(term) <= 40 and term.lower() not in seen_prompts:
                        seen_prompts.add(term.lower())
                        prompt = f"Define {term} and explain its fundamental importance in course concepts."
                        cls_res = self.classifier.classify(prompt, "5 marks")
                        ans = GroundedAnswerOutline(
                            is_grounded=True,
                            core_answer=line,
                            key_points=[f"Sourced directly from lecture section: {sec.heading}."],
                            formula=None,
                            citations=[{"doc_name": sec.doc_name, "locator": sec.page_or_slide, "heading": sec.heading, "section_id": sec.section_id}],
                        )
                        questions.append(PracticeQuestion(
                            question_id=f"GEN_{len(questions)+1}",
                            display_id=f"PRACTICE-{len(questions)+1}",
                            text=prompt,
                            marks="5 marks",
                            category=cls_res.category,
                            category_name=cls_res.category_name,
                            category_rationale="Generated revision prompt from lecture note definition.",
                            is_pyq=False,
                            source_label="Generated Practice Question (Not from Past Papers)",
                            answer_outline=ans,
                            diagram=self.diagrams.get_diagram_for_text(term),
                        ))

                # 2. Distinction question check
                if any(w in line.lower() for w in ("in contrast", "difference between", "differs from")):
                    prompt = f"Differentiate between key approaches discussed in: {sec.heading}."
                    if prompt.lower() not in seen_prompts and len(questions) < 12:
                        seen_prompts.add(prompt.lower())
                        cls_res = self.classifier.classify(prompt, "5 marks")
                        ans = GroundedAnswerOutline(
                            is_grounded=True,
                            core_answer=line,
                            key_points=[f"Refer to comparative section in notes: {sec.heading}."],
                            formula=None,
                            citations=[{"doc_name": sec.doc_name, "locator": sec.page_or_slide, "heading": sec.heading, "section_id": sec.section_id}],
                        )
                        questions.append(PracticeQuestion(
                            question_id=f"GEN_{len(questions)+1}",
                            display_id=f"PRACTICE-{len(questions)+1}",
                            text=prompt,
                            marks="5 marks",
                            category=cls_res.category,
                            category_name=cls_res.category_name,
                            category_rationale="Generated distinction prompt from comparative note sentence.",
                            is_pyq=False,
                            source_label="Generated Practice Question (Not from Past Papers)",
                            answer_outline=ans,
                            diagram=self.diagrams.get_diagram_for_text(sec.heading),
                        ))

            if len(questions) == section_question_start and len(questions) < 15 and sec.heading.strip():
                source_line = next((line for line in lines if len(line) >= 30), "")
                prompt = f"Explain {sec.heading} based on the uploaded study notes."
                if source_line and prompt.lower() not in seen_prompts:
                    seen_prompts.add(prompt.lower())
                    cls_res = self.classifier.classify(prompt, "5 marks")
                    answer = GroundedAnswerOutline(
                        is_grounded=True,
                        core_answer=source_line,
                        key_points=[f"Use the cited passage from {sec.heading} as supporting material."],
                        citations=[{
                            "doc_name": sec.doc_name,
                            "locator": sec.page_or_slide,
                            "heading": sec.heading,
                            "section_id": sec.section_id,
                        }],
                    )
                    questions.append(PracticeQuestion(
                        question_id=f"GEN_{len(questions)+1}",
                        display_id=f"PRACTICE-{len(questions)+1}",
                        text=prompt,
                        marks="5 marks",
                        category=cls_res.category,
                        category_name=cls_res.category_name,
                        category_rationale="Generated from an uploaded note heading; not a historical exam question.",
                        is_pyq=False,
                        source_label="Generated Practice Question (Not from Past Papers)",
                        answer_outline=answer,
                        diagram=self.diagrams.get_diagram_for_text(sec.heading),
                    ))

            if len(questions) >= 15:
                break

        return questions
