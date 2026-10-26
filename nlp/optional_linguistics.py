"""
Optional Linguistics Module for SyllabusLens.
Provides rule-based Part-of-Speech (POS) tagging and Academic Concept/Named Entity recognition.
Status: Implemented and tested (Rule-based & Morphological Heuristic Engine).
"""

import re
from typing import List, Tuple, Dict, Any


# Common suffix patterns for Penn Treebank POS tagging
SUFFIX_RULES = [
    (r'.*able$', 'JJ'),        # Adjective (e.g. قابل)
    (r'.*ive$', 'JJ'),         # Adjective
    (r'.*ous$', 'JJ'),         # Adjective
    (r'.*ful$', 'JJ'),         # Adjective
    (r'.*al$', 'JJ'),          # Adjective
    (r'.*ic$', 'JJ'),          # Adjective
    (r'.*ing$', 'VBG'),        # Verb, gerund/present participle
    (r'.*ed$', 'VBD'),         # Verb, past tense
    (r'.*ly$', 'RB'),          # Adverb
    (r'.*tion$', 'NN'),        # Noun, singular
    (r'.*sion$', 'NN'),        # Noun
    (r'.*ment$', 'NN'),        # Noun
    (r'.*ness$', 'NN'),        # Noun
    (r'.*ity$', 'NN'),         # Noun
    (r'.*ance$', 'NN'),        # Noun
    (r'.*ence$', 'NN'),        # Noun
    (r'.*er$', 'NN'),          # Noun (agent)
    (r'.*or$', 'NN'),          # Noun (agent)
    (r'.*ist$', 'NN'),         # Noun
    (r'.*s$', 'NNS'),          # Noun, plural
]

# Closed-class lexicon
CLOSED_CLASS = {
    "the": "DT", "a": "DT", "an": "DT", "this": "DT", "that": "DT", "these": "DT", "those": "DT",
    "in": "IN", "on": "IN", "at": "IN", "by": "IN", "for": "IN", "with": "IN", "about": "IN", "against": "IN",
    "between": "IN", "into": "IN", "through": "IN", "during": "IN", "before": "IN", "after": "IN",
    "above": "IN", "below": "IN", "to": "TO", "from": "IN", "up": "IN", "down": "IN", "off": "IN", "over": "IN",
    "and": "CC", "but": "CC", "or": "CC", "nor": "CC", "so": "CC", "yet": "CC",
    "is": "VBZ", "are": "VBP", "was": "VBD", "were": "VBD", "be": "VB", "been": "VBN", "being": "VBG",
    "have": "VBP", "has": "VBZ", "had": "VBD", "do": "VBP", "does": "VBZ", "did": "VBD",
    "can": "MD", "could": "MD", "will": "MD", "would": "MD", "shall": "MD", "should": "MD", "may": "MD", "might": "MD", "must": "MD",
    "what": "WP", "which": "WDT", "who": "WP", "whom": "WP", "whose": "WP$", "where": "WRB", "when": "WRB", "why": "WRB", "how": "WRB"
}

# Academic Computer Science / NLP Concept Entity Dictionary
ACADEMIC_ENTITIES = {
    "tf-idf": "METRIC_REPRESENTATION",
    "tfidf": "METRIC_REPRESENTATION",
    "bag of words": "MODEL_REPRESENTATION",
    "bow": "MODEL_REPRESENTATION",
    "cosine similarity": "SIMILARITY_METRIC",
    "tokenization": "PREPROCESSING_TECHNIQUE",
    "stemming": "PREPROCESSING_TECHNIQUE",
    "lemmatization": "PREPROCESSING_TECHNIQUE",
    "n-gram": "LANGUAGE_MODEL",
    "ngrams": "LANGUAGE_MODEL",
    "pos tagging": "LINGUISTIC_TASK",
    "ner": "INFORMATION_EXTRACTION",
    "named entity recognition": "INFORMATION_EXTRACTION",
    "word2vec": "EMBEDDING_ARCHITECTURE",
    "glove": "EMBEDDING_ARCHITECTURE",
    "fasttext": "EMBEDDING_ARCHITECTURE",
    "rnn": "NEURAL_ARCHITECTURE",
    "lstm": "NEURAL_ARCHITECTURE",
    "gru": "NEURAL_ARCHITECTURE",
    "transformer": "NEURAL_ARCHITECTURE",
    "attention": "NEURAL_MECHANISM",
    "encoder-decoder": "NEURAL_ARCHITECTURE",
    "bert": "PRETRAINED_MODEL",
    "gpt": "PRETRAINED_MODEL",
    "precision": "EVALUATION_METRIC",
    "recall": "EVALUATION_METRIC",
    "f1 score": "EVALUATION_METRIC",
    "mrr": "EVALUATION_METRIC"
}


class RuleBasedLinguisticsEngine:
    """
    Lightweight rule-based POS tagger and Academic Entity Extractor.
    Designed for educational transparency in the NLP Lab.
    """

    def __init__(self):
        self.compiled_suffixes = [(re.compile(pat, re.IGNORECASE), tag) for pat, tag in SUFFIX_RULES]

    def tag_pos(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Assigns Penn Treebank Part-of-Speech tags to tokens using
        lexicon lookups, capitalization heuristics, and morphological suffix rules.
        """
        tagged: List[Tuple[str, str]] = []
        for word in tokens:
            low = word.lower()
            # 1. Closed class lookup
            if low in CLOSED_CLASS:
                tagged.append((word, CLOSED_CLASS[low]))
                continue

            # 2. Number check
            if re.match(r'^\d+(?:\.\d+)?$', word):
                tagged.append((word, 'CD'))
                continue

            # 3. Proper noun check (capitalized inside sentence or uppercase abbreviation)
            if word.isupper() and len(word) > 1:
                tagged.append((word, 'NNP'))
                continue

            # 4. Morphological suffix check
            matched_tag = None
            for pattern, tag in self.compiled_suffixes:
                if pattern.match(low):
                    matched_tag = tag
                    break

            if matched_tag:
                tagged.append((word, matched_tag))
            else:
                # Default noun for content words
                tagged.append((word, 'NN'))

        return tagged

    def extract_academic_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts NLP & Computer Science concepts, models, and metrics from text.
        """
        entities: List[Dict[str, Any]] = []
        text_lower = text.lower()

        for concept, category in ACADEMIC_ENTITIES.items():
            pattern = r'\b' + re.escape(concept) + r'\b'
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                matched_text = text[start:end]
                entities.append({
                    "concept": matched_text,
                    "normalized": concept,
                    "category": category,
                    "start": start,
                    "end": end
                })

        return entities
