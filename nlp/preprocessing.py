"""
Text Preprocessing module for SyllabusLens.
Provides normalization, tokenization, stop-word removal, stemming, and n-gram extraction.
"""

import re
import unicodedata
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass, field
from nlp.stemmer import stem_word

# Standard English Stop Words (comprehensive curated list for academic IR)
STANDARD_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Academic/Question filler words that add little topical discrimination
    "discuss", "explain", "describe", "write", "briefly", "short", "note", "notes",
    "illustrate", "define", "differentiate", "compare", "contrast", "state", "list"
}


@dataclass
class PreprocessedResult:
    """Stores intermediate and final outputs of text preprocessing."""
    original_text: str
    normalized_text: str
    tokens: List[str] = field(default_factory=list)
    tokens_no_stops: List[str] = field(default_factory=list)
    stemmed_tokens: List[str] = field(default_factory=list)
    unigrams: List[str] = field(default_factory=list)
    bigrams: List[str] = field(default_factory=list)
    cleaned_text: str = ""


class TextPreprocessor:
    """
    Modular text preprocessor supporting academic text normalization,
    tokenization, stop-word removal, stemming, and n-grams.
    """

    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        use_stemming: bool = False,
        min_token_len: int = 2,
        custom_stopwords: Optional[Set[str]] = None
    ):
        self.stop_words = set(stop_words) if stop_words is not None else set(STANDARD_STOP_WORDS)
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        self.use_stemming = use_stemming
        self.min_token_len = min_token_len
        # Token pattern: alphanumeric tokens, keeping internal hyphens/underscores if relevant
        # Allows one internal apostrophe so contractions ("don't") can be matched
        # against the stop-word list before apostrophes are stripped.
        self.token_pattern = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]*(?:'[a-zA-Z]+)?\b")

    def normalize_text(self, text: str) -> str:
        """Applies Unicode normalization, case folding, and whitespace cleanup."""
        if not text:
            return ""
        # Unicode normalization (NFKD separates accents from letters)
        normalized = unicodedata.normalize('NFKD', text)
        # Convert to lower case
        lowered = normalized.lower().replace("\u2019", "'").replace("\u2018", "'")
        # Replace non-printable / control chars and strange whitespace with space
        cleaned = re.sub(r'[\r\t\f\v]+', ' ', lowered)
        cleaned = re.sub(r' +', ' ', cleaned)
        return cleaned.strip()

    def tokenize(self, text: str) -> List[str]:
        """Extracts words matching token pattern and meeting minimum length."""
        if not text:
            return []
        raw_tokens = self.token_pattern.findall(text)
        return [tok for tok in raw_tokens if len(tok) >= self.min_token_len]

    @staticmethod
    def _strip_apostrophe(tok: str) -> str:
        """Possessive "'s" is dropped; other internal apostrophes are removed."""
        if tok.endswith("'s"):
            return tok[:-2]
        return tok.replace("'", "")

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Removes stop words (matched with apostrophes intact), then strips apostrophes."""
        kept = [self._strip_apostrophe(t) for t in tokens if t not in self.stop_words]
        return [t for t in kept if len(t) >= self.min_token_len and t not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Applies Porter Stemmer to tokens."""
        return [stem_word(tok) for tok in tokens]

    def generate_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Generates n-grams as space-separated strings."""
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def process(self, text: str) -> PreprocessedResult:
        """
        Executes full preprocessing pipeline and records every intermediate transformation.
        """
        original = text or ""
        normalized = self.normalize_text(original)
        tokens = self.tokenize(normalized)
        tokens_no_stops = self.remove_stopwords(tokens)

        if self.use_stemming:
            final_tokens = self.stem_tokens(tokens_no_stops)
        else:
            final_tokens = tokens_no_stops

        unigrams = list(final_tokens)
        bigrams = self.generate_ngrams(final_tokens, 2)
        cleaned_text = " ".join(final_tokens)

        return PreprocessedResult(
            original_text=original,
            normalized_text=normalized,
            tokens=tokens,
            tokens_no_stops=tokens_no_stops,
            stemmed_tokens=self.stem_tokens(tokens_no_stops),
            unigrams=unigrams,
            bigrams=bigrams,
            cleaned_text=cleaned_text
        )


# Global default instance
default_preprocessor = TextPreprocessor()

def preprocess_text(text: str, use_stemming: bool = False) -> str:
    """Convenience helper to get cleaned string for vectorization."""
    p = TextPreprocessor(use_stemming=use_stemming)
    return p.process(text).cleaned_text
