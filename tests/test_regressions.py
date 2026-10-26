"""Each test pins a defect found during the audit (and its fix)."""
import numpy as np
import pytest

from nlp.document_segmentation import DocumentSegmenter
from nlp.preprocessing import TextPreprocessor
from nlp.question_segmentation import QuestionItem, QuestionSegmenter
from nlp.vectorization import EmptyVocabularyError, VectorizerEngine
from services.analysis_service import AnalysisService
from services.coverage_service import CoverageService


def test_eg_ie_continuations_are_not_subquestions():
    qs = QuestionSegmenter().segment("1. Explain tokenization.\ne.g. whitespace methods.\n2. Describe TF-IDF.\ni.e. term weighting.")
    assert [q.question_id for q in qs] == ["Q1", "Q2"] and "e.g." in qs[0].clean_text


def test_real_subquestions_still_parsed():
    qs = QuestionSegmenter().segment("1. Answer:\n(a) Define IDF.\n(b) Compute it.\nc) Discuss.\n2(i) Explain parsing.")
    assert [q.question_id for q in qs] == ["Q1.a", "Q1.b", "Q1.c", "Q2.i"]


def test_contraction_stopwords_removed_and_possessives_stripped():
    t = TextPreprocessor().process("Don't explain what isn't important; the network’s edge").tokens_no_stops
    assert t == ["important", "network", "edge"]


def test_vocabulary_independent_of_question_paper():
    secs = DocumentSegmenter().segment_txt_document("n.txt", "# A\ntokenization splits text\n# B\nparsing builds trees\n# C\nstemming reduces words")
    def idf(texts):
        qs = [QuestionItem(question_id=f"Q{i}", raw_text=t, clean_text=t) for i, t in enumerate(texts)]
        return AnalysisService().run_analysis(qs, secs).vectorizer.idf_weights
    assert idf(["tokenization"]) == idf(["parsing parsing", "stemming reduces"])


def test_section_coverage_uses_full_matrix_not_top_k():
    seg = DocumentSegmenter()
    secs = seg.segment_txt_document("n.txt", "\n".join(f"# S{i}\ntokenization tokens text splits {'filler ' * i}" for i in range(1, 7)))
    r = AnalysisService().run_analysis([QuestionItem(question_id="Q1", raw_text="x", clean_text="tokenization tokens text splits")], secs, top_k=3)
    _, stats = CoverageService().compute_coverage(r)
    sims = r.similarity_matrix[0]
    assert all((s.coverage_status != "Uncovered in Exam") == (sims[i] >= r.weak_threshold) for i, s in enumerate(stats))


def test_sankey_nodes_keyed_by_id_not_heading():
    secs = DocumentSegmenter().segment_txt_document("n.txt", "# Same long heading text repeated for both sections here\ntokens\n# Same long heading text repeated for both sections here\nstems")
    r = AnalysisService().run_analysis([QuestionItem(question_id="Q1", raw_text="x", clean_text="tokens stems")], secs)
    d = CoverageService().prepare_sankey_data(r, min_threshold=0.0)
    assert len(d["node_labels"]) == 1 + len(secs) and len(set(d["targets"])) == len(d["targets"])


def test_empty_vocabulary_raises_instead_of_fabricating_terms():
    for corpus in ([], ["", "  "]):
        with pytest.raises(EmptyVocabularyError):
            VectorizerEngine().fit(corpus)
    with pytest.raises(EmptyVocabularyError):
        VectorizerEngine(min_df=5).fit(["alpha beta", "gamma delta"])


def test_zero_vectors_and_unseen_terms_give_zero_not_nan():
    secs = DocumentSegmenter().segment_txt_document("n.txt", "# A\nalpha beta gamma")
    r = AnalysisService().run_analysis([QuestionItem(question_id="Q1", raw_text="x", clean_text="zzz qqq")], secs)
    assert np.isfinite(r.similarity_matrix).all() and r.question_results[0].best_tier == "Unmatched"


def test_duplicate_section_ids_rejected():
    s = DocumentSegmenter().segment_txt_document("n.txt", "# A\nalpha\n# B\nbeta")
    s[1].section_id = s[0].section_id
    with pytest.raises(ValueError, match="Duplicate section IDs"):
        AnalysisService().run_analysis([QuestionItem(question_id="Q1", raw_text="a", clean_text="alpha")], s)


def test_all_stopword_notes_give_clear_error(svc):
    from services.errors import DocumentValidationError
    s = svc.subjects.create("S")
    svc.library.ingest(s.id, "n.txt", b"# H\nthe and of to in", doc_kind="lecture_notes")
    p = svc.library.ingest(s.id, "p.txt", b"1. Explain tokenization. [5 marks]\n2. Describe parsing.", doc_kind="question_paper").document
    with pytest.raises(DocumentValidationError, match="No vocabulary|no usable text"):
        svc.runner.run(s.id, [p.id], save=False)


def test_long_stem_is_kept_but_bare_instruction_stem_is_dropped():
    keep = QuestionSegmenter().segment("1. Consider a CPU scheduler using round robin with quantum four.\n(a) Compute waiting time.\n(b) Compute turnaround time.")
    assert [q.question_id for q in keep] == ["Q1", "Q1.a", "Q1.b"]
    drop = QuestionSegmenter().segment("1. Answer:\n(a) Compute waiting time.")
    assert [q.question_id for q in drop] == ["Q1.a"]


def test_headings_are_cleaned_and_truncated_at_word_boundaries():
    from nlp.document_segmentation import clean_heading, truncate_title
    assert clean_heading("Sec 12: Unit 3: Parsing") == "Unit 3: Parsing" and clean_heading("Page 2: Intro") == "Page 2: Intro"
    long = "Unit 2: Vector Space Models, Bag of Words, and TF-IDF Retrieval Systems Explained In Depth For All Students"
    t = truncate_title(long)
    assert t.endswith("\u2026") and len(t) <= 91 and not t.rstrip("\u2026").endswith(" ") and long.startswith(t[:-1])
    secs = DocumentSegmenter().segment_txt_document("n.txt", "# " + long + "\ntext here\n# Second\nmore text")
    assert secs[0].heading.endswith("\u2026") and "Retrie" not in secs[0].heading.split()[-1]


def test_marks_are_not_doubled_in_question_cards():
    from components import ui
    q = dict(best_tier="Strong", display_id="Q1", best_score=0.5, text="x", matches=[], marks="10 marks", paper_label="")
    assert ui.question_card_html(q).count("marks") == 1 and "10 marks" in ui.question_card_html(q)
    assert "5 marks" in ui.question_card_html({**q, "marks": "5"})
