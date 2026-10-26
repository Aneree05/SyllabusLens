"""NLP Lab: inspect every real stage of the pipeline on text you type or on your active subject's corpus."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components import state, ui
from nlp.document_segmentation import DocumentSegmenter, clean_heading
from nlp.optional_linguistics import RuleBasedLinguisticsEngine
from nlp.preprocessing import TextPreprocessor
from nlp.question_segmentation import QuestionItem, QuestionSegmenter
from nlp.vectorization import EmptyVocabularyError, VectorizerEngine
from services.errors import SyllabusLensError

DEFAULT_TEXT = ("Define the Bag of Words (BoW) model and explain Term Frequency-Inverse Document Frequency (TF-IDF). "
                "State its primary limitations in capturing word order.")

NOT_IMPLEMENTED = [
    ("Word embeddings (word2vec, GloVe, transformers)", "Not implemented. Retrieval is purely lexical (TF-IDF / Bag of Words), so it cannot match synonyms or paraphrases."),
    ("Lemmatization", "Not implemented. Only Porter stemming is available."),
    ("Trained named-entity recognition", "Not implemented. The Lab offers a small dictionary lookup of NLP/CS terms (below), which is not NER."),
    ("Statistical part-of-speech tagging", "Not implemented. The Lab offers a suffix-rule heuristic (below) that makes errors."),
    ("OCR for scanned PDFs", "Not implemented. Image-only files are flagged as having no text."),
]


def _preprocessing_tab() -> None:
    text = st.text_area("Text to trace through the pipeline", DEFAULT_TEXT, height=90, key="lab_text")
    c1, c2 = st.columns(2)
    stem = c1.checkbox("Apply Porter stemming", True, key="lab_stem")
    bigrams = c2.checkbox("Include word pairs (bigrams)", True, key="lab_bi")
    res = TextPreprocessor(use_stemming=stem).process(text)
    st.markdown("**1. Normalise** - Unicode normalisation and lowercasing")
    st.code(res.normalized_text or "(empty)", language=None)
    st.markdown(f"**2. Tokenise** - {len(res.tokens)} tokens"); st.write(res.tokens)
    st.markdown(f"**3. Remove stop words** - {len(res.tokens_no_stops)} kept")
    st.caption("Removes grammatical filler and exam-instruction words such as 'explain' or 'define', so they do not create false matches.")
    st.write(res.tokens_no_stops)
    if stem:
        st.markdown("**4. Stem** - Porter (1980) suffix stripping, so related word forms share one term"); st.write(res.stemmed_tokens)
    st.markdown("**5. N-grams** - the features the vectoriser counts")
    a, b = st.columns(2)
    a.write(res.unigrams)
    if bigrams:
        b.write(res.bigrams)
    return res


def _vector_tab() -> None:
    text = st.session_state.get("lab_text", DEFAULT_TEXT)
    stem = st.session_state.get("lab_stem", True)
    ngram = (1, 2) if st.session_state.get("lab_bi", True) else (1, 1)
    cleaned = TextPreprocessor(use_stemming=stem).process(text).cleaned_text
    st.markdown("**Bag of Words** counts each term; **TF-IDF** weights a term by how rare it is across the corpus: "
                "tf-idf(t, d) = tf(t, d) x idf(t), with idf(t) = ln((1 + N) / (1 + df(t))) + 1, then each vector is length-normalised.")
    st.caption("With only one text there is no corpus, so every IDF is equal. Rank against your subject's notes in the next tab to see real IDF weights.")
    eng = VectorizerEngine(ngram_range=ngram)
    try:
        eng.fit([cleaned])
    except EmptyVocabularyError as exc:
        ui.notice("warning", "No terms left to count", str(exc))
        return
    rows = eng.inspect_text_vector(cleaned, top_n=25)
    st.metric("Distinct terms", len(eng.feature_names))
    st.dataframe(rows, width="stretch", hide_index=True)


def _ranking_tab(svc, subject) -> None:
    st.markdown("Rank a typed question against the **notes of the active subject**. This uses the same subject-filtered corpus and engine as Question Analysis.")
    if subject is None:
        ui.empty_state("Select a subject", "Choose a subject in the sidebar to rank text against its lecture material.")
        return
    q = st.text_area("Question or phrase", "Explain how cosine similarity compares document vectors.", height=70, key=f"lab_q_{subject.id}")
    method = st.radio("Vectoriser", ["tfidf", "bow"], horizontal=True, key=f"lab_vec_{subject.id}", format_func={"tfidf": "TF-IDF", "bow": "Bag of Words"}.get)
    if not st.button("Rank against this subject's notes", key=f"lab_rank_{subject.id}", disabled=not q.strip()):
        return
    from services.analysis_service import AnalysisService
    try:
        corpus = svc.retrieval.build_note_corpus(subject.id)                  # subject filter happens here
        item = QuestionItem(question_id="LAB", raw_text=q, clean_text=q)
        res = AnalysisService().run_analysis([item], list(corpus.sections), vectorizer_type=method, top_k=5)
    except SyllabusLensError as exc:
        ui.notice("error", "Could not rank", str(exc)); return
    qr = res.question_results[0]
    ui.notice("info", f"Searched {len(corpus.sections)} sections in {subject.name} only", f"Corpus version {corpus.fingerprint[:10]}. Best tier: {qr.best_tier} (heuristic).")
    if not qr.top_matches or qr.best_tier == "Unmatched":
        ui.notice("warning", "No section shares enough vocabulary", "Nothing reached the weak-match threshold. Any sections listed below are the closest by vocabulary, "
                  "shown for reference only. Try different wording, or check that the notes cover this topic.")
    for m in qr.top_matches:
        ui.render(f'<div class="sl-card"><div class="sl-card-title">#{m.rank} {ui.esc(clean_heading(m.heading))}</div><div>{ui.tier_badge(m.match_tier)} {ui.badge("similarity %.3f" % m.similarity_score)}</div>'
                  f'<div class="sl-card-meta"><span>{ui.esc(m.doc_name)}</span><span>{ui.esc(m.page_or_slide)}</span></div></div>')
        if m.matching_terms:
            st.caption("Shared terms (contribution to the score): " + ", ".join(f"{t['term']} ({t.get('contribution', 0):.3f})" for t in m.matching_terms[:6]))
    with st.expander("Inspect the query vector against the real corpus vocabulary"):
        st.dataframe(res.vectorizer.inspect_text_vector(qr.clean_text, top_n=20), width="stretch", hide_index=True)
    st.caption("Cosine similarity = (q . d) / (|q| |d|). It ranges 0 to 1 here and ignores length, but it only sees shared words.")


def _segmentation_tab() -> None:
    kind = st.radio("What to segment", ["Question paper text", "Lecture notes text"], horizontal=True, key="lab_seg_kind")
    sample = ("1. (a) Define stemming. [5 marks]\n(b) Compare it with lemmatization. [5 marks]\n2. Explain TF-IDF. e.g. with an example. [10 marks]"
              if kind.startswith("Question") else "# Tokenization\nSplitting text into tokens.\n# Parsing\nBuilding parse trees from grammar rules.")
    text = st.text_area("Paste text", sample, height=150, key=f"lab_seg_{kind}")
    if kind.startswith("Question"):
        qs = QuestionSegmenter().segment(text)
        st.dataframe([{"ID": q.question_id, "Marks": q.marks or "", "Question": q.clean_text} for q in qs], width="stretch", hide_index=True) if qs else st.caption("No questions detected.")
        st.caption("Numbered items, (a)/(b) parts and mark indicators are recognised. Wrong splits can be corrected on the Question Analysis page.")
    else:
        secs = DocumentSegmenter().segment_txt_document("pasted.txt", text)
        st.dataframe([{"Section": s.heading, "Location": s.page_or_slide, "Words": s.word_count} for s in secs], width="stretch", hide_index=True) if secs else st.caption("No sections detected.")
        st.caption("Headings (#, Unit, Chapter...) start sections; long sections are split into overlapping chunks.")


def _linguistics_tab() -> None:
    ui.notice("warning", "Illustrative heuristics, not trained models", "Neither tool below is used in retrieval or scoring. They exist to demonstrate the ideas and make mistakes.")
    text = st.text_area("Text", "The parser builds trees quickly. BERT and LSTM outperform TF-IDF at Stanford University.", height=70, key="lab_ling")
    eng = RuleBasedLinguisticsEngine()
    tokens = TextPreprocessor().process(text).tokens
    a, b = st.columns(2)
    with a:
        st.markdown("**Part-of-speech guess** (suffix rules and a small word list)")
        st.dataframe([{"Token": w, "Tag": t} for w, t in eng.tag_pos(tokens)], width="stretch", hide_index=True)
        st.caption("Known weakness: any word ending in -s is tagged a plural noun, so verbs such as 'builds' are wrong; capitalised names are not detected.")
    with b:
        st.markdown("**Concept spotter** (dictionary of about 28 NLP/CS terms)")
        ents = eng.extract_academic_entities(text)
        st.dataframe([{"Term": e["concept"], "Category": e["category"]} for e in ents], width="stretch", hide_index=True) if ents else st.caption("No known terms.")
        st.caption("This is a lookup, not named-entity recognition: people, places and organisations such as 'Stanford University' are not found.")


def render() -> None:
    svc = state.get_services()
    ui.page_header("NLP Lab", "Inspect every stage of the matching pipeline. What you see here is what the analysis really does.")
    subject = state.get_active_subject(svc)
    t1, t2, t3, t4, t5, t6 = st.tabs(["Preprocessing", "Vectors", "Rank against notes", "Segmentation", "Linguistics demos", "Not implemented"])
    with t1:
        _preprocessing_tab()
    with t2:
        _vector_tab()
    with t3:
        _ranking_tab(svc, subject)
    with t4:
        _segmentation_tab()
    with t5:
        _linguistics_tab()
    with t6:
        st.caption("These are not available in this version and are not used anywhere in the app.")
        for name, why in NOT_IMPLEMENTED:
            ui.render(f'<div class="sl-card"><div class="sl-card-title">{ui.esc(name)}</div><p class="sl-caption" style="margin:0">{ui.esc(why)}</p></div>')
