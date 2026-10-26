"""
Deterministic Concept Diagram Module for SyllabusLens.
Renders structured, accurate, labeled concept diagrams using Graphviz DOT.
These diagrams represent verified conceptual models and architectures grounded in
course syllabi. No OCR or automated image hallucination is performed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DiagramSpec:
    diagram_id: str
    title: str
    caption: str
    dot_source: str
    concept_domain: str


class DiagramService:
    """
    Supplies deterministic, publication-quality Graphviz diagrams for core
    academic computer science, NLP, and system architectures.
    """

    def get_diagram_for_text(self, text: str) -> Optional[DiagramSpec]:
        """Inspects text (question, topic, or section heading) and matches a relevant diagram."""
        lower = text.lower()

        if any(k in lower for k in ("transformer", "self-attention", "multi-head", "scaled dot-product")):
            return self.diagram_transformer_attention()

        if any(k in lower for k in ("lstm", "long short-term memory", "gating mechanism")):
            return self.diagram_lstm_cell()

        if any(k in lower for k in ("recurrent neural network", "rnn", "backpropagation through time", "bptt")):
            return self.diagram_rnn_architecture()

        if any(k in lower for k in ("cosine similarity", "cosine distance", "vector space retrieval")):
            return self.diagram_cosine_similarity()

        if any(k in lower for k in ("tf-idf", "term frequency", "bag of words", "bow model", "vector space model")):
            return self.diagram_bow_vs_tfidf()

        if any(k in lower for k in ("stemming", "lemmatization", "porter stemmer", "morphological analysis")):
            return self.diagram_stemming_vs_lemmatization()

        if any(k in lower for k in ("text normalization", "tokenization", "sentence segmentation", "stop word")):
            return self.diagram_text_normalization()

        if any(k in lower for k in ("pos tagging", "part-of-speech", "ner", "named entity recognition")):
            return self.diagram_sequence_labeling()

        if any(k in lower for k in ("precision", "recall", "confusion matrix", "evaluation metric")):
            return self.diagram_ir_evaluation()

        return None

    def diagram_text_normalization(self) -> DiagramSpec:
        dot = """digraph TextNorm {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    raw [label="Raw Text Corpus\\n(Unstructured)", fillcolor="#EFF6FF", color="#3B82F6", penwidth=2];
    seg [label="1. Sentence\\nSegmentation", fillcolor="#F1F5F9"];
    tok [label="2. Word\\nTokenization", fillcolor="#F1F5F9"];
    norm [label="3. Case Folding &\\nUnicode Norm", fillcolor="#F1F5F9"];
    stop [label="4. Stop Word\\nRemoval", fillcolor="#F1F5F9"];
    stem [label="5. Stemming /\\nLemmatization", fillcolor="#F1F5F9"];
    tokens [label="Clean Vocabulary\\nTokens", fillcolor="#ECFDF5", color="#10B981", penwidth=2];

    raw -> seg [label="Sentence boundaries"];
    seg -> tok [label="Tokens"];
    tok -> norm [label="Normalized text"];
    norm -> stop [label="Filtered"];
    stop -> stem [label="Base forms"];
    stem -> tokens [label="Indexed lexicon"];
}"""
        return DiagramSpec(
            diagram_id="text_norm_pipeline",
            title="Text Normalization and Linguistic Preprocessing Pipeline",
            caption="Sequential steps converting raw unstructured character streams into indexed lexical tokens.",
            dot_source=dot,
            concept_domain="Text Normalization",
        )

    def diagram_bow_vs_tfidf(self) -> DiagramSpec:
        dot = """digraph BoWvsTFIDF {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    corpus [label="Input Documents\\n{d1, d2, ..., dN}", fillcolor="#EFF6FF", color="#3B82F6", penwidth=2];
    vocab [label="Fixed Vocabulary Set V\\n(Unique unigrams / bigrams)", fillcolor="#F1F5F9"];

    subgraph cluster_bow {
        label="Bag-of-Words (BoW)";
        style="filled,dashed";
        fillcolor="#F8FAFC";
        color="#CBD5E1";
        bow_count [label="Term Frequency\\nCount: c(t, d)"];
        bow_vec [label="BoW Vector\\nHigh sparsity, unweighted", fillcolor="#FEF3C7", color="#F59E0B"];
    }

    subgraph cluster_tfidf {
        label="TF-IDF Model";
        style="filled,dashed";
        fillcolor="#F8FAFC";
        color="#CBD5E1";
        tf [label="TF(t, d) = f(t, d)"];
        idf [label="IDF(t) = ln((1+|D|)/(1+DF(t))) + 1"];
        tfidf_vec [label="TF-IDF Feature Vector\\nDistinctive terms boosted", fillcolor="#ECFDF5", color="#10B981"];
    }

    corpus -> vocab;
    vocab -> bow_count;
    bow_count -> bow_vec;
    vocab -> tf;
    corpus -> idf [label="Corpus statistics"];
    tf -> tfidf_vec [label="TF * IDF"];
    idf -> tfidf_vec;
}"""
        return DiagramSpec(
            diagram_id="bow_vs_tfidf",
            title="Comparison: Bag-of-Words (BoW) vs. TF-IDF Feature Projection",
            caption="BoW counts raw token occurrences; TF-IDF down-weights ubiquitous terms across the collection.",
            dot_source=dot,
            concept_domain="Vector Space Models",
        )

    def diagram_cosine_similarity(self) -> DiagramSpec:
        dot = """digraph CosineSim {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    q [label="Query Vector q\\n[q1, q2, ..., qV]", fillcolor="#EFF6FF", color="#3B82F6"];
    d [label="Document Vector d\\n[d1, d2, ..., dV]", fillcolor="#EFF6FF", color="#3B82F6"];
    
    dot [label="Dot Product\\nq · d = ∑(q_i * d_i)", fillcolor="#F1F5F9"];
    norm_q [label="L2 Norm ||q||\\n√(∑ q_i²)", fillcolor="#F1F5F9"];
    norm_d [label="L2 Norm ||d||\\n√(∑ d_i²)", fillcolor="#F1F5F9"];
    
    sim [label="Cosine Similarity\\ncos(θ) = (q · d) / (||q|| * ||d||)\\nRange: [0, 1.0]", fillcolor="#ECFDF5", color="#10B981", penwidth=2];

    q -> dot;
    d -> dot;
    q -> norm_q;
    d -> norm_d;
    dot -> sim;
    norm_q -> sim;
    norm_d -> sim;
}"""
        return DiagramSpec(
            diagram_id="cosine_similarity",
            title="Vector Space Cosine Similarity Geometry",
            caption="Normalizes vector magnitude, rendering similarity invariant to document text length variations.",
            dot_source=dot,
            concept_domain="Vector Space Geometry",
        )

    def diagram_stemming_vs_lemmatization(self) -> DiagramSpec:
        dot = """digraph StemVsLemma {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    word [label="Surface Word\\n(e.g., 'running', 'better')", fillcolor="#EFF6FF", color="#3B82F6", penwidth=2];

    subgraph cluster_stem {
        label="Rule-Based Stemmer (e.g., Porter)";
        style="filled,dashed";
        fillcolor="#FFFBEB";
        color="#FDE68A";
        s_rules [label="Suffix Stripping Rules\\n(-ing, -ed, -s)"];
        s_out [label="Stem: 'run'\\n(May not be valid dictionary word)", fillcolor="#FEF3C7"];
    }

    subgraph cluster_lemma {
        label="Morphological Lemmatizer";
        style="filled,dashed";
        fillcolor="#F0FDF4";
        color="#BBF7D0";
        l_pos [label="Part-of-Speech Context\\n(e.g., ADJ, VERB)"];
        l_dict [label="Full Morphological\\nDictionary Lookup"];
        l_out [label="Lemma: 'good'\\n(Guaranteed valid lexical entry)", fillcolor="#DCFCE7"];
    }

    word -> s_rules;
    s_rules -> s_out;
    word -> l_pos;
    word -> l_dict;
    l_pos -> l_dict [label="Disambiguation"];
    l_dict -> l_out;
}"""
        return DiagramSpec(
            diagram_id="stem_vs_lemma",
            title="Process Distinction: Rule-Based Stemming vs. Lemmatization",
            caption="Stemming chops suffixes using heuristic rewrite rules; lemmatization requires full vocabulary and POS context.",
            dot_source=dot,
            concept_domain="Morphological Analysis",
        )

    def diagram_rnn_architecture(self) -> DiagramSpec:
        dot = """digraph RNN {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    subgraph cluster_step1 {
        label="Step t-1";
        style=dashed;
        x0 [label="Input x_{t-1}"];
        h0 [label="Hidden h_{t-1}\\n= tanh(W_h h_{t-2} + W_x x_{t-1})", fillcolor="#EFF6FF", color="#3B82F6"];
        y0 [label="Output y_{t-1}"];
        x0 -> h0;
        h0 -> y0;
    }

    subgraph cluster_step2 {
        label="Step t";
        style=dashed;
        x1 [label="Input x_t"];
        h1 [label="Hidden h_t\\n= tanh(W_h h_{t-1} + W_x x_t)", fillcolor="#EFF6FF", color="#3B82F6"];
        y1 [label="Output y_t"];
        x1 -> h1;
        h1 -> y1;
    }

    subgraph cluster_step3 {
        label="Step t+1";
        style=dashed;
        x2 [label="Input x_{t+1}"];
        h2 [label="Hidden h_{t+1}\\n= tanh(W_h h_t + W_x x_{t+1})", fillcolor="#EFF6FF", color="#3B82F6"];
        y2 [label="Output y_{t+1}"];
        x2 -> h2;
        h2 -> y2;
    }

    h0 -> h1 [label="Recurrent W_h", color="#2563EB", penwidth=2];
    h1 -> h2 [label="Recurrent W_h", color="#2563EB", penwidth=2];
}"""
        return DiagramSpec(
            diagram_id="rnn_unfolded",
            title="Recurrent Neural Network (RNN) Forward Computation Unfolded Over Time",
            caption="Hidden state vector h_t passes sequential context forward across successive token time steps.",
            dot_source=dot,
            concept_domain="Sequential Neural Models",
        )

    def diagram_lstm_cell(self) -> DiagramSpec:
        dot = """digraph LSTMCell {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    inputs [label="Inputs: [h_{t-1}, x_t]", fillcolor="#EFF6FF", color="#3B82F6", penwidth=2];
    
    f_gate [label="Forget Gate (f_t)\\nσ(W_f · [h_{t-1}, x_t] + b_f)\\nDecides what history to drop", fillcolor="#FEF3C7", color="#F59E0B"];
    i_gate [label="Input Gate (i_t)\\nσ(W_i · [h_{t-1}, x_t] + b_i)\\nDecides what new info to store", fillcolor="#FEF3C7", color="#F59E0B"];
    c_cand [label="Candidate State (~C_t)\\ntanh(W_c · [h_{t-1}, x_t] + b_c)", fillcolor="#F1F5F9"];
    
    cell_state [label="Cell Memory Carousel (C_t)\\nC_t = f_t * C_{t-1} + i_t * ~C_t\\nMitigates Vanishing Gradients", fillcolor="#ECFDF5", color="#10B981", penwidth=2];
    
    o_gate [label="Output Gate (o_t)\\nσ(W_o · [h_{t-1}, x_t] + b_o)", fillcolor="#FEF3C7", color="#F59E0B"];
    h_out [label="Updated Hidden State (h_t)\\nh_t = o_t * tanh(C_t)", fillcolor="#EFF6FF", color="#3B82F6", penwidth=2];

    inputs -> f_gate;
    inputs -> i_gate;
    inputs -> c_cand;
    inputs -> o_gate;
    f_gate -> cell_state;
    i_gate -> cell_state;
    c_cand -> cell_state;
    cell_state -> h_out;
    o_gate -> h_out;
}"""
        return DiagramSpec(
            diagram_id="lstm_cell",
            title="Long Short-Term Memory (LSTM) Cell and Gating Architecture",
            caption="Constant error carousel and additive gates resolve the vanishing gradient bottleneck of vanilla RNNs.",
            dot_source=dot,
            concept_domain="Deep Sequence Learning",
        )

    def diagram_transformer_attention(self) -> DiagramSpec:
        dot = """digraph TransformerAttention {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    emb [label="Input Embeddings + Positional Encodings X", fillcolor="#EFF6FF", color="#3B82F6", penwidth=2];

    q [label="Query Matrix (Q = X W_Q)", fillcolor="#F1F5F9"];
    k [label="Key Matrix (K = X W_K)", fillcolor="#F1F5F9"];
    v [label="Value Matrix (V = X W_V)", fillcolor="#F1F5F9"];

    matmul1 [label="Scaled Dot Product\\nQ * K^T / √(d_k)", fillcolor="#FEF3C7"];
    smax [label="Softmax Normalization\\n(Attention Weights A)", fillcolor="#FEF3C7", color="#F59E0B"];
    matmul2 [label="Weighted Summation\\nOutput = A * V", fillcolor="#ECFDF5", color="#10B981", penwidth=2];

    head_concat [label="Multi-Head Concatenation\\nConcat(head_1, ..., head_h) W_O", fillcolor="#ECFDF5", color="#10B981"];

    emb -> q;
    emb -> k;
    emb -> v;
    q -> matmul1;
    k -> matmul1;
    matmul1 -> smax;
    smax -> matmul2;
    v -> matmul2;
    matmul2 -> head_concat;
}"""
        return DiagramSpec(
            diagram_id="transformer_self_attention",
            title="Transformer Architecture: Scaled Dot-Product & Multi-Head Self-Attention",
            caption="Eliminates recurrent bottlenecks by computing parallel contextual dependencies across queries, keys, and values.",
            dot_source=dot,
            concept_domain="Transformer Architectures",
        )

    def diagram_sequence_labeling(self) -> DiagramSpec:
        dot = """digraph SeqLabeling {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    tokens [label="Sentence:\\n['Google', 'acquired', 'DeepMind', 'in', 'London']", fillcolor="#EFF6FF", color="#3B82F6"];
    pos [label="Part-of-Speech (POS):\\n['NNP', 'VBD', 'NNP', 'IN', 'NNP']", fillcolor="#F1F5F9"];
    bio [label="Named Entity BIO Scheme:\\n['B-ORG', 'O', 'B-ORG', 'O', 'B-LOC']", fillcolor="#ECFDF5", color="#10B981", penwidth=2];

    tokens -> pos [label="Syntactic Role Tagging"];
    tokens -> bio [label="Entity Boundary & Category"];
}"""
        return DiagramSpec(
            diagram_id="sequence_labeling",
            title="Linguistic Sequence Labeling: POS Tagging and BIO Named Entity Recognition",
            caption="Maps token sequences to grammatical roles and domain entity spans.",
            dot_source=dot,
            concept_domain="Linguistic Annotation",
        )

    def diagram_ir_evaluation(self) -> DiagramSpec:
        dot = """digraph IREvaluation {
    rankdir=TB;
    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#F8FAFC", color="#94A3B8"];
    edge [fontname="Helvetica", fontsize=9, color="#64748B"];

    subgraph cluster_grid {
        label="Evaluation Matrix";
        style=dashed;
        tp [label="True Positives (TP)\\nRetrieved & Relevant", fillcolor="#DCFCE7", color="#16A34A"];
        fp [label="False Positives (FP)\\nRetrieved & Not Relevant", fillcolor="#FEE2E2", color="#DC2626"];
        fn [label="False Negatives (FN)\\nNot Retrieved & Relevant", fillcolor="#FEF3C7", color="#D97706"];
        tn [label="True Negatives (TN)\\nNot Retrieved & Not Relevant", fillcolor="#F1F5F9"];
    }

    prec [label="Precision@k = TP / (TP + FP)\\nAccuracy of retrieved set", fillcolor="#EFF6FF", color="#3B82F6"];
    rec [label="Recall@k = TP / (TP + FN)\\nCoverage of relevant corpus", fillcolor="#EFF6FF", color="#3B82F6"];

    tp -> prec;
    fp -> prec;
    tp -> rec;
    fn -> rec;
}"""
        return DiagramSpec(
            diagram_id="ir_evaluation_matrix",
            title="Information Retrieval Evaluation: Precision, Recall, and Confusion Space",
            caption="Measures accuracy (precision) versus comprehensive syllabus coverage (recall).",
            dot_source=dot,
            concept_domain="Evaluation Metrics",
        )
