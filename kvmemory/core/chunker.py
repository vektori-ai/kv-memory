"""
chunker.py — Semantic chunker for conversation turns.

Splits text at sentence boundaries targeting 80-150 tokens per chunk.
Falls back to simple regex if spacy is not installed.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

# Attempt spacy import; fall back to regex sentence splitter
try:
    import spacy

    _nlp = None

    def _load_spacy():
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spacy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                _nlp = None
        return _nlp

    def _split_sentences_spacy(text: str) -> list[str]:
        nlp = _load_spacy()
        if nlp is None:
            return _split_sentences_regex(text)
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    _SPACY_AVAILABLE = True

except ImportError:
    _SPACY_AVAILABLE = False
    logger.debug("spacy not installed; using regex sentence splitter")


# Simple regex sentence splitter — adequate for most English text
_SENTENCE_RE = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"
)


def _split_sentences_regex(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    sentences = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_sentences(text: str) -> list[str]:
    if _SPACY_AVAILABLE:
        return _split_sentences_spacy(text)
    return _split_sentences_regex(text)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def chunk_turn(
    text: str,
    tokenizer,
    target_tokens: int = 100,
    min_tokens: int = 20,
) -> list[str]:
    """
    Split a conversation turn into semantic chunks.

    Splits at sentence boundaries. Accumulates sentences until the
    target token count is exceeded, then starts a new chunk.
    Chunks below min_tokens are merged with the previous chunk.

    Args:
        text:          full turn text (prompt + response)
        tokenizer:     must have .encode(str) -> list[int]
        target_tokens: target chunk size in tokens (80-150 recommended)
        min_tokens:    minimum chunk size; smaller chunks are merged

    Returns:
        list of text chunks, each approximately target_tokens long
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current_sents: list[str] = []
    current_count: int = 0

    for sent in sentences:
        n = len(tokenizer.encode(sent))

        if n == 0:
            continue

        # If adding this sentence would exceed target and we have content, flush
        if current_count + n > target_tokens and current_sents:
            chunk_text = " ".join(current_sents)
            chunks.append(chunk_text)
            current_sents = [sent]
            current_count = n
        else:
            current_sents.append(sent)
            current_count += n

    # Flush remainder
    if current_sents:
        remainder = " ".join(current_sents)
        # Merge tiny trailing chunk into previous if below min_tokens
        if current_count < min_tokens and chunks:
            chunks[-1] = chunks[-1] + " " + remainder
        else:
            chunks.append(remainder)

    return chunks
