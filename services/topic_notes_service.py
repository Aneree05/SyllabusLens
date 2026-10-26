"""
Topic-Wise Revision Notes Service for SyllabusLens.
Synthesizes comprehensive, source-grounded exam preparation notes organized by topic.
Aggregates content across multiple documents, slides, and pages, extracting definitions,
formulas, processes, derivations, advantages/limitations, and explicitly identifying missing material.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nlp.document_segmentation import clean_heading
from services.diagram_service import DiagramService, DiagramSpec
from services.retrieval_service import Corpus, RetrievalService
from services.subject_service import SubjectService


# Cues for section element extraction
_RE_DEF = re.compile(
    r"\b(is\s+defined\s+as|refers\s+to|denotes|is\s+a\s+technique|is\s+an\s+approach|"
    r"is\s+the\s+process\s+of|represents\s+the|conceptually\s+means|defined\s+by)\b|"
    r"^(?:[A-Z][a-zA-Z\s\-]{2,30}):\s+[A-Z]",
    re.IGNORECASE,
)

_RE_FORMULA = re.compile(
    r"(=|≈|\bIDF\b|\bTF-IDF\b|\bSim\(|\btanh\(|\bsoftmax\(|\b\w+\s*=\s*[^;\.\n]{3,}|"
    r"\\sum|\\frac|\bCount\(|\|\s*\|\s*\w+\s*\|\s*\|)",
)

_RE_PROCESS = re.compile(
    r"^(?:\d+[\.\)]|\b(?:Step\s*\d+|First|Second|Third|Finally|Next|Then)\b|"
    r"\b(?:Architecture|Pipeline|Phase|Algorithm|Workflow)\b)",
    re.IGNORECASE,
)

_RE_PRO_CON = re.compile(
    r"\b(advantage|disadvantage|limitation|drawback|benefit|weakness|strength|"
    r"fails\s+to|cannot\s+capture|suffers\s+from|challenges?|trade[\-\s]*off|application)\b",
    re.IGNORECASE,
)

_RE_EXAMPLE = re.compile(
    r"\b(for\s+example|e\.g\.|illustrate|for\s+instance|consider\s+the|such\s+as|"
    r"demonstrated\s+by|example:)\b",
    re.IGNORECASE,
)


@dataclass
class SourceCitation:
    doc_name: str
    locator: str
    heading: str
    section_id: str

    def to_str(self) -> str:
        return f"{self.doc_name} ({self.locator}) - {self.heading}"


@dataclass
class TopicRevisionNote:
    topic: str
    status: str                         # "covered" | "partially_covered" | "uncovered"
    summary: str
    definitions: List[Tuple[str, SourceCitation]] = field(default_factory=list)
    explanations: List[Tuple[str, SourceCitation]] = field(default_factory=list)
    formulas: List[Tuple[str, SourceCitation]] = field(default_factory=list)
    derivations_examples: List[Tuple[str, SourceCitation]] = field(default_factory=list)
    pros_cons_limitations: List[Tuple[str, SourceCitation]] = field(default_factory=list)
    tables_diagrams: List[Tuple[str, SourceCitation]] = field(default_factory=list)
    diagram: Optional[DiagramSpec] = None
    sources: List[SourceCitation] = field(default_factory=list)
    missing_material: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "status": self.status,
            "summary": self.summary,
            "definitions": [{"text": t, "source": s.to_str()} for t, s in self.definitions],
            "explanations": [{"text": t, "source": s.to_str()} for t, s in self.explanations],
            "formulas": [{"text": t, "source": s.to_str()} for t, s in self.formulas],
            "derivations_examples": [{"text": t, "source": s.to_str()} for t, s in self.derivations_examples],
            "pros_cons_limitations": [{"text": t, "source": s.to_str()} for t, s in self.pros_cons_limitations],
            "tables_diagrams": [{"text": t, "source": s.to_str()} for t, s in self.tables_diagrams],
            "diagram": self.diagram.title if self.diagram else None,
            "sources": [s.to_str() for s in self.sources],
            "missing_material": self.missing_material,
        }


class TopicNotesService:
    def __init__(self, subjects: SubjectService, retrieval: RetrievalService, diagram_svc: Optional[DiagramService] = None):
        self.subjects = subjects
        self.retrieval = retrieval
        self.diagram_svc = diagram_svc or DiagramService()

    def build_topic_notes(
        self,
        subject_id: str,
        note_doc_ids: Optional[Sequence[str]] = None,
        min_section_similarity: float = 0.10,
    ) -> List[TopicRevisionNote]:
        """
        Builds topic-wise revision notes by aggregating all note sections belonging
        to the given subject. Preserves strict subject isolation.
        """
        subject = self.subjects.get(subject_id)
        corpus: Corpus = self.retrieval.build_note_corpus(subject_id, note_doc_ids)
        sections = list(corpus.sections)

        if not sections:
            return []

        # 1. Topic discovery: use configured subject topics, or derive major unit headings
        topics = list(subject.topics)
        if not topics:
            topics = self._discover_topics_from_sections(sections)
        if not topics:
            topics = ["General Course Concepts"]

        # 2. Vectorize sections and topics using TF-IDF for reliable multi-document aggregation
        from nlp.vectorization import EmptyVocabularyError, VectorizerEngine
        vec = VectorizerEngine(ngram_range=(1, 2), min_df=1)
        sec_texts = [s.text or "" for s in sections]
        sec_headings = [s.heading or "" for s in sections]
        combined_sec = [f"{h} {t}" for h, t in zip(sec_headings, sec_texts)]

        try:
            vec.fit(combined_sec)
            S_mat = vec.transform_tfidf(combined_sec)
            T_mat = vec.transform_tfidf(topics)
            sim_mat = cosine_similarity(T_mat, S_mat)
        except EmptyVocabularyError:
            sim_mat = np.zeros((len(topics), len(sections)))

        out: List[TopicRevisionNote] = []

        for t_idx, topic in enumerate(topics):
            topic_lower = topic.lower()
            matched_sections: List[Tuple[Any, float]] = []

            for s_idx, sec in enumerate(sections):
                score = float(sim_mat[t_idx, s_idx])
                # Direct heading match or lexical score
                heading_match = topic_lower in (sec.heading or "").lower() or (sec.heading or "").lower() in topic_lower
                if score >= min_section_similarity or heading_match:
                    matched_sections.append((sec, max(score, 0.40 if heading_match else score)))

            # Sort matched sections by relevance
            matched_sections.sort(key=lambda x: x[1], reverse=True)

            if not matched_sections:
                out.append(TopicRevisionNote(
                    topic=topic,
                    status="uncovered",
                    summary="No material in the uploaded notes matches this topic.",
                    missing_material=["Complete topic content missing from uploaded lecture notes and textbooks."],
                ))
                continue

            # Synthesize elements across all matched sections
            definitions: List[Tuple[str, SourceCitation]] = []
            explanations: List[Tuple[str, SourceCitation]] = []
            formulas: List[Tuple[str, SourceCitation]] = []
            derivations: List[Tuple[str, SourceCitation]] = []
            pros_cons: List[Tuple[str, SourceCitation]] = []
            tables_diagrams: List[Tuple[str, SourceCitation]] = []
            seen_sentences = set()
            sources_dict: Dict[str, SourceCitation] = {}

            for sec, sc in matched_sections:
                cit = SourceCitation(
                    doc_name=sec.doc_name,
                    locator=sec.page_or_slide,
                    heading=clean_heading(sec.heading),
                    section_id=sec.section_id,
                )
                sources_dict[sec.section_id] = cit

                lines = [l.strip() for l in (sec.text or "").split("\n") if l.strip()]
                for line in lines:
                    norm = " ".join(line.lower().split())
                    if norm in seen_sentences or len(line) < 15:
                        continue
                    seen_sentences.add(norm)

                    is_formula = bool(_RE_FORMULA.search(line)) and any(c in line for c in "=≈∑√/")
                    is_def = bool(_RE_DEF.search(line))
                    is_pro_con = bool(_RE_PRO_CON.search(line))
                    is_ex = bool(_RE_EXAMPLE.search(line))
                    is_process = bool(_RE_PROCESS.search(line))

                    if is_formula:
                        formulas.append((line, cit))
                    elif is_def and len(definitions) < 5:
                        definitions.append((line, cit))
                    elif is_pro_con:
                        pros_cons.append((line, cit))
                    elif is_ex:
                        derivations.append((line, cit))
                    elif is_process or len(explanations) < 6:
                        explanations.append((line, cit))

            # Missing material analysis
            missing: List[str] = []
            if not definitions:
                missing.append("Explicit formal definitions not identified in uploaded text.")
            if not formulas and any(w in topic_lower for w in ("tfidf", "cosine", "vector", "metric", "probability", "bayes", "regression", "neural")):
                missing.append("Mathematical formula equations not found in matching sections.")
            if not pros_cons:
                missing.append("Advantages, disadvantages, and trade-off comparisons not discussed.")
            if not derivations:
                missing.append("Concrete examples, derivations, or step-by-step calculations omitted.")

            status = "covered" if (len(definitions) + len(explanations) >= 3 and not missing) else "partially_covered"

            # Diagram matching
            diag = self.diagram_svc.get_diagram_for_text(f"{topic} {' '.join(s[0].heading for s in matched_sections[:3])}")

            # Synthesis summary
            summary_sentences = []
            if definitions:
                summary_sentences.append(definitions[0][0])
            elif explanations:
                summary_sentences.append(explanations[0][0])
            if pros_cons:
                summary_sentences.append(pros_cons[0][0])
            summary = " ".join(summary_sentences) if summary_sentences else f"Covers {topic} based on {len(matched_sections)} note section(s)."

            out.append(TopicRevisionNote(
                topic=topic,
                status=status,
                summary=summary,
                definitions=definitions[:6],
                explanations=explanations[:8],
                formulas=formulas[:5],
                derivations_examples=derivations[:5],
                pros_cons_limitations=pros_cons[:6],
                tables_diagrams=tables_diagrams,
                diagram=diag,
                sources=list(sources_dict.values()),
                missing_material=missing,
            ))

        return out

    def _discover_topics_from_sections(self, sections: List[Any]) -> List[str]:
        """Discovers major topics from prominent unit/chapter headings."""
        candidates = []
        for s in sections:
            h = clean_heading(s.heading)
            if re.search(r"\b(Unit|Chapter|Module|Part)\s*\d+", h, re.I):
                clean = re.sub(r"^(?:Unit|Chapter|Module|Part)\s*\d+[\s\:\-]+", "", h, flags=re.I).strip()
                if clean and clean not in candidates:
                    candidates.append(clean)
        return candidates[:10]

    def export_notes_markdown(self, notes: List[TopicRevisionNote], subject_name: str) -> str:
        """Renders comprehensive, structured markdown notes for exam revision."""
        lines = [
            f"# {subject_name} - Topic-Wise Revision Notes",
            "Generated by SyllabusLens Exam Intelligence Workspace.",
            "All definitions, explanations, formulas, and derivations are grounded in uploaded study material.",
            "",
            "---",
            "",
        ]

        for note in notes:
            lines.append(f"## {note.topic}")
            lines.append(f"**Coverage Status**: `{note.status.replace('_', ' ').title()}`")
            lines.append(f"**Core Concept**: {note.summary}")
            lines.append("")

            if note.definitions:
                lines.append("### Key Definitions & Foundations")
                for text, cit in note.definitions:
                    lines.append(f"- **Definition**: {text}  \n  *Source: [{cit.to_str()}]*")
                lines.append("")

            if note.explanations:
                lines.append("### Explanations, Classifications & Algorithms")
                for text, cit in note.explanations:
                    lines.append(f"- {text}  \n  *Source: [{cit.to_str()}]*")
                lines.append("")

            if note.formulas:
                lines.append("### Formulas & Governing Equations")
                for text, cit in note.formulas:
                    lines.append(f"- `{text}`  \n  *Source: [{cit.to_str()}]*")
                lines.append("")

            if note.derivations_examples:
                lines.append("### Derivations, Examples & Calculations")
                for text, cit in note.derivations_examples:
                    lines.append(f"- {text}  \n  *Source: [{cit.to_str()}]*")
                lines.append("")

            if note.pros_cons_limitations:
                lines.append("### Advantages, Limitations & Comparisons")
                for text, cit in note.pros_cons_limitations:
                    lines.append(f"- {text}  \n  *Source: [{cit.to_str()}]*")
                lines.append("")

            if note.diagram:
                lines.append(f"### Conceptual Diagram: {note.diagram.title}")
                lines.append(f"*{note.diagram.caption}*")
                lines.append("")

            if note.missing_material:
                lines.append("### Missing Material Alerts")
                for m in note.missing_material:
                    lines.append(f"- ⚠️ {m}")
                lines.append("")

            if note.sources:
                lines.append("### Source Documents & Page References")
                for s in note.sources:
                    lines.append(f"- {s.to_str()}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)
