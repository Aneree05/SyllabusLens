"""Syllabus-topic coverage: which configured topics are detected in the selected papers."""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nlp.document_segmentation import clean_heading
from services.analysis_service import AnalysisService, CompleteAnalysisResult


_TOKEN = re.compile(r"(?u)\b\w+\b")
MIN_VOCAB_COVERAGE = 0.5   # share of a topic's words that must occur in the notes for it to be assessable


def topic_coverage(analysis: CompleteAnalysisResult, svc: AnalysisService, topics: List[str],
                   vectorizer_type: str = "tfidf") -> List[Dict[str, Any]]:
    """
    Each topic is scored against every question and every note section with the same fitted
    vectorizer. Statuses:
      detected        - some question has similarity >= weak threshold
      not_detected    - topic words exist in the notes vocabulary but no question reaches the threshold
      not_assessable  - half or fewer of the topic's words occur in the notes vocabulary, so a lexical score
                        would rest on a few generic words (e.g. only "analysis" of "sentiment analysis")
    """
    if not topics:
        return []
    vec = analysis.vectorizer
    tf = vec.transform_bow if vectorizer_type == "bow" else vec.transform_tfidf
    t_clean = [svc.preprocessor.process(t).cleaned_text for t in topics]
    q_clean = [qr.clean_text for qr in analysis.question_results]
    s_clean = [svc.preprocessor.process(s.text).cleaned_text for s in analysis.note_sections]
    T, Q, S = tf(t_clean), tf(q_clean), tf(s_clean)
    tq = np.nan_to_num(cosine_similarity(T, Q)) if Q.shape[0] else np.zeros((len(topics), 0))
    ts = np.nan_to_num(cosine_similarity(T, S))
    out = []
    vocab = analysis.vectorizer.vocabulary or {}
    for i, topic in enumerate(topics):
        words = list(dict.fromkeys(_TOKEN.findall(t_clean[i])))   # same tokenisation as the vectoriser
        vocab_cov = (sum(1 for w in words if w in vocab) / len(words)) if words else 0.0
        in_vocab = vocab_cov > MIN_VOCAB_COVERAGE and T[i].nnz > 0
        best_q = int(tq[i].argmax()) if tq.shape[1] else None
        best_s = int(ts[i].argmax()) if ts.shape[1] else None
        qscore = float(tq[i, best_q]) if best_q is not None else 0.0
        papers = sorted({analysis.questions[j].metadata.get("paper_doc_id")
                         for j in range(tq.shape[1]) if tq[i, j] >= analysis.weak_threshold})
        status = ("not_assessable" if not in_vocab
                  else "detected" if qscore >= analysis.weak_threshold else "not_detected")
        out.append(dict(
            topic=topic, status=status, vocab_coverage=round(vocab_cov, 2), best_question_score=round(qscore, 4),
            best_question_id=analysis.questions[best_q].question_id if best_q is not None and qscore > 0 else None,
            best_question_display=analysis.questions[best_q].metadata.get("display_id") if best_q is not None and qscore > 0 else None,
            papers_detected=[p for p in papers if p],
            best_section_id=analysis.note_sections[best_s].section_id if best_s is not None and ts[i, best_s] > 0 else None,
            best_section_heading=clean_heading(analysis.note_sections[best_s].heading) if best_s is not None and ts[i, best_s] > 0 else None,
            best_section_score=round(float(ts[i, best_s]), 4) if best_s is not None else 0.0))
    return out
