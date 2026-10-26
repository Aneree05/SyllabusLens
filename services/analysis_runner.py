"""
Orchestrates one full analysis run:

  subject -> eligible corpus -> vectorize/rank -> coverage -> comparison -> topics -> snapshot -> history

The runner never touches documents directly; it asks RetrievalService for an already
subject-filtered corpus, so unrelated subjects cannot enter the NLP pipeline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import config
from nlp.document_segmentation import clean_heading
from nlp.vectorization import EmptyVocabularyError
from services.analysis_service import AnalysisService, CompleteAnalysisResult
from services.comparison_service import compare_papers
from services.coverage_service import CoverageService
from services.errors import DocumentValidationError
from services.history_service import HistoryService
from services.retrieval_service import Corpus, PaperSet, RetrievalService
from services.subject_service import SubjectService
from services.topic_service import topic_coverage

LIMITATIONS = (
    "Scores are lexical (shared vocabulary), not semantic understanding. A high score does not prove "
    "that a section fully answers a question, and a low score does not prove the topic is absent. "
    "Match tiers use fixed, uncalibrated similarity thresholds. Frequencies describe the selected "
    "past papers only and are not predictions of future examinations."
)


@dataclass(frozen=True)
class AnalysisSettings:
    vectorizer: str = "tfidf"                       # "tfidf" | "bow"
    ngram_range: Tuple[int, int] = config.DEFAULT_NGRAM_RANGE
    use_stemming: bool = False
    strong_threshold: float = config.DEFAULT_STRONG_THRESHOLD
    moderate_threshold: float = config.DEFAULT_MODERATE_THRESHOLD
    weak_threshold: float = config.DEFAULT_WEAK_THRESHOLD
    top_k: int = 3

    def validate(self) -> None:
        if self.vectorizer not in ("tfidf", "bow"):
            raise ValueError("vectorizer must be 'tfidf' or 'bow'")
        if not (self.strong_threshold > self.moderate_threshold > self.weak_threshold > 0):
            raise ValueError("Thresholds must satisfy strong > moderate > weak > 0.")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")


@dataclass
class AnalysisOutcome:
    result: CompleteAnalysisResult
    record: Dict[str, Any]
    corpus: Corpus
    papers: PaperSet
    cache_hit: bool
    analysis_id: Optional[str] = None
    reused_saved: bool = False       # identical inputs were already saved; no duplicate history entry written


class AnalysisCache:
    """In-process cache. Keys contain the subject IDs, the corpus fingerprint and the paper fingerprint."""

    def __init__(self):
        self._store: Dict[tuple, Tuple[CompleteAnalysisResult, Dict[str, Any]]] = {}
        self.saved: Dict[tuple, str] = {}     # cache key -> history analysis id

    @staticmethod
    def key(corpus: Corpus, papers: PaperSet, settings: AnalysisSettings) -> tuple:
        sh = hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()[:16]
        return (tuple(sorted(corpus.subject_ids)), corpus.fingerprint, papers.fingerprint, sh)

    def get(self, key): return self._store.get(key)
    def put(self, key, value): self._store[key] = value
    def clear(self): self._store.clear(); self.saved.clear()
    def __len__(self): return len(self._store)


class AnalysisRunner:
    def __init__(self, subjects: SubjectService, retrieval: RetrievalService, history: HistoryService,
                 cache: Optional[AnalysisCache] = None,
                 topic_notes_svc: Optional[Any] = None,
                 question_practice_svc: Optional[Any] = None):
        self.subjects, self.retrieval, self.history = subjects, retrieval, history
        self.cache = cache or AnalysisCache()
        self.coverage = CoverageService()
        self.topic_notes_svc = topic_notes_svc
        self.question_practice_svc = question_practice_svc

    def run(self, subject_id: Optional[str], paper_ids: Sequence[str],
            note_document_ids: Optional[Sequence[str]] = None, *,
            settings: Optional[AnalysisSettings] = None, name: Optional[str] = None,
            allow_cross_subject: bool = False, extra_subject_ids: Sequence[str] = (),
            save: bool = True, force_new: bool = False) -> AnalysisOutcome:
        settings = settings or AnalysisSettings()
        settings.validate()
        # 1-3: subject validation + eligible corpus (filter happens HERE, before any NLP)
        corpus = self.retrieval.build_note_corpus(subject_id, note_document_ids,
                                                  allow_cross_subject=allow_cross_subject,
                                                  extra_subject_ids=extra_subject_ids)
        papers = self.retrieval.build_paper_set(subject_id, paper_ids)
        subject = self.subjects.get(subject_id)  # type: ignore[arg-type]

        key = self.cache.key(corpus, papers, settings)
        cached = self.cache.get(key)
        svc = AnalysisService(settings.strong_threshold, settings.moderate_threshold, settings.weak_threshold,
                              settings.ngram_range, settings.use_stemming)
        if cached:
            result, record = cached
            hit = True
        else:
            try:
                result = svc.run_analysis(list(papers.questions), list(corpus.sections),
                                          vectorizer_type=settings.vectorizer, top_k=settings.top_k)
            except EmptyVocabularyError as exc:
                raise DocumentValidationError(str(exc)) from exc
            record = self._build_record(result, svc, corpus, papers, settings, subject)
            self.cache.put(key, (result, record))
            hit = False

        analysis_id, reused = None, False
        if save:
            existing = self.cache.saved.get(key)
            if existing and not force_new and self.history.exists(existing):
                # Same subject, corpus version, papers and settings were already saved this session:
                # an accidental re-run must not create a duplicate history entry.
                return AnalysisOutcome(result, record, corpus, papers, hit, existing, True)
            label = name or f"{subject.name} - {len(papers.papers)} paper(s)"
            analysis_id = self.history.save(
                subject_id=subject.id, name=label, method=result.method_name, settings=record["settings"],
                inputs={"papers": list(papers.papers), "documents": list(corpus.documents)},
                corpus_fingerprint=corpus.fingerprint, cross_subject=corpus.cross_subject, record=record)
            self.cache.saved[key] = analysis_id
        return AnalysisOutcome(result, record, corpus, papers, hit, analysis_id, reused)

    # ------------------------------------------------------------------ snapshot
    def _build_record(self, result, svc, corpus, papers, settings, subject) -> Dict[str, Any]:
        from nlp.question_classifier import QuestionClassifier
        from services.numerical_solver import NumericalSolver
        from services.diagram_service import DiagramService

        classifier = QuestionClassifier()
        solver = NumericalSolver()
        diagram_svc = DiagramService()

        agg, sec_stats = self.coverage.compute_coverage(result)
        comparison = compare_papers(result, list(papers.papers)) if len(papers.papers) > 1 else None
        topics = topic_coverage(result, svc, subject.topics, settings.vectorizer) if subject.topics else []
        paper_label = {p["document_id"]: p["title"] for p in papers.papers}
        paper_facts = {p["document_id"]: p for p in papers.papers}

        sec_meta = {s.section_id: s for s in result.note_sections}
        questions = []
        for q, qr in zip(result.questions, result.question_results):
            text = qr.question_text
            marks = qr.marks or ""
            paper = paper_facts.get(q.metadata.get("paper_doc_id"), {})
            cls_res = classifier.classify(text, marks)

            matches = [dict(rank=m.rank, section_id=m.section_id,
                            document_id=sec_meta[m.section_id].metadata.get("document_id"),
                            doc_name=m.doc_name, locator=m.page_or_slide, heading=clean_heading(m.heading),
                            score=round(m.similarity_score, 4), tier=m.match_tier,
                            snippet=m.section_snippet, explanation=m.explanation,
                            key_terms=[t["term"] for t in m.matching_terms[:5]]) for m in qr.top_matches]

            # Grounded answer outline
            valid_m = [m for m in qr.top_matches if m.similarity_score >= settings.weak_threshold]
            if valid_m:
                core_ans = ""
                key_pts = []
                formula = None
                for vm in valid_m[:3]:
                    snip = vm.section_snippet or ""
                    for sent in [s.strip() for s in snip.split(".") if len(s.strip()) > 15]:
                        if not core_ans and any(w in sent.lower() for w in ("is defined as", "refers to", "is a")):
                            core_ans = sent + "."
                        elif len(key_pts) < 4 and sent not in key_pts:
                            key_pts.append(sent + ".")
                    if not formula and any(c in snip for c in ("=", "≈", "IDF", "Sim", "tanh")):
                        for line in snip.splitlines():
                            if any(c in line for c in ("=", "≈")) and len(line.strip()) < 100:
                                formula = line.strip()
                                break
                answer_outline = dict(
                    is_grounded=True,
                    core_answer=core_ans or (key_pts.pop(0) if key_pts else "Supported by lecture notes."),
                    key_points=key_pts,
                    formula=formula,
                    citations=[dict(doc_name=vm.doc_name, locator=vm.page_or_slide, heading=clean_heading(vm.heading)) for vm in valid_m[:3]],
                    insufficient_reason=None)
            else:
                answer_outline = dict(
                    is_grounded=False,
                    core_answer="",
                    key_points=[],
                    formula=None,
                    citations=[],
                    insufficient_reason="No lecture note section shares sufficient vocabulary with this question. Material may be missing from your notes.")

            # Numerical solver
            num_sol = solver.analyze_and_solve(text) if cls_res.category == "numerical" or any(w in text.lower() for w in ("calculate", "compute", "value")) else None
            diag = diagram_svc.get_diagram_for_text(text)

            questions.append(dict(
                question_id=qr.question_id, display_id=q.metadata.get("display_id", qr.question_id),
                paper_doc_id=q.metadata.get("paper_doc_id"), paper_label=paper_label.get(q.metadata.get("paper_doc_id"), ""),
                academic_year=paper.get("academic_year", ""), exam_session=paper.get("exam_session", ""),
                text=text, marks=marks, best_score=round(qr.best_score, 4), best_tier=qr.best_tier,
                category=cls_res.category, category_name=cls_res.category_name, category_rationale=cls_res.rationale,
                is_pyq=True, answer_outline=answer_outline,
                numerical_solution=num_sol.to_dict() if num_sol else None,
                diagram_spec=dict(id=diag.diagram_id, title=diag.title, caption=diag.caption, dot=diag.dot_source) if diag else None,
                matches=matches))

        sections = []
        for st in sec_stats:
            s = sec_meta[st.section_id]
            sections.append(dict(section_id=st.section_id, document_id=s.metadata.get("document_id"),
                                 doc_name=st.doc_name, locator=st.page_or_slide, heading=clean_heading(st.heading),
                                 max_similarity=round(st.max_similarity, 4), top1_match_count=st.top1_match_count,
                                 top3_match_count=st.top3_match_count, coverage_status=st.coverage_status,
                                 matched_questions=st.matched_questions, word_count=s.word_count))

        # Topic revision notes synthesis
        topic_notes = []
        if self.topic_notes_svc:
            note_ids = [d["document_id"] for d in corpus.documents if d["subject_id"] == subject.id]
            if note_ids:
                t_notes = self.topic_notes_svc.build_topic_notes(subject.id, note_ids)
                topic_notes = [tn.to_dict() for tn in t_notes]

        return dict(
            version=1, subject=dict(id=subject.id, name=subject.name, course_code=subject.course_code),
            method=result.method_name, settings=dict(result.settings, syllabus_topics=list(subject.topics),
                                                     cross_subject=corpus.cross_subject,
                                                     subjects_in_corpus=list(corpus.subject_ids)),
            thresholds=dict(strong=settings.strong_threshold, moderate=settings.moderate_threshold, weak=settings.weak_threshold),
            corpus_fingerprint=corpus.fingerprint, papers=list(papers.papers), documents=list(corpus.documents),
            questions=questions, sections=sections, metrics=asdict(agg), comparison=comparison,
            topic_coverage=topics, topic_revision_notes=topic_notes,
            matrix=dict(question_ids=[q["question_id"] for q in questions],
                        section_ids=[s["section_id"] for s in sections],
                        values=[[round(float(v), 4) for v in row] for row in result.similarity_matrix]),
            limitations=LIMITATIONS)
