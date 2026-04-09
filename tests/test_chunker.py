"""
test_chunker.py — Chunker unit tests.

Verifies sentence splitting and token-budget-aware chunking.
"""

from __future__ import annotations

import pytest

from kvmemory.core.chunker import chunk_turn, _split_sentences_regex


class TestSentenceSplitter:
    def test_splits_on_period(self):
        text = "Hello world. This is a test. Another sentence."
        sents = _split_sentences_regex(text)
        assert len(sents) >= 2

    def test_empty_input(self):
        assert _split_sentences_regex("") == []

    def test_single_sentence(self):
        sents = _split_sentences_regex("Just one sentence here.")
        assert len(sents) == 1

    def test_question_mark_split(self):
        text = "What is this? This is a test."
        sents = _split_sentences_regex(text)
        assert len(sents) >= 2

    def test_exclamation_split(self):
        text = "Amazing! That is great."
        sents = _split_sentences_regex(text)
        assert len(sents) >= 2


class TestChunkTurn:
    def test_returns_list(self, fake_tokenizer):
        result = chunk_turn("Hello world. This is a test.", fake_tokenizer)
        assert isinstance(result, list)

    def test_empty_input_returns_empty(self, fake_tokenizer):
        assert chunk_turn("", fake_tokenizer) == []

    def test_whitespace_only_returns_empty(self, fake_tokenizer):
        assert chunk_turn("   ", fake_tokenizer) == []

    def test_chunks_are_strings(self, fake_tokenizer):
        text = "First sentence. Second sentence. Third one here."
        chunks = chunk_turn(text, fake_tokenizer)
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_respects_target_tokens(self, fake_tokenizer):
        """With a tiny target, each sentence should be its own chunk."""
        # Each word becomes one token in FakeTokenizer
        text = "one two three. four five six. seven eight nine."
        chunks = chunk_turn(text, fake_tokenizer, target_tokens=3)
        # Should produce at least 2 chunks since we have ~9 tokens total
        assert len(chunks) >= 2

    def test_large_target_produces_single_chunk(self, fake_tokenizer):
        """With a huge target, all sentences should merge into one chunk."""
        text = "Short. Text. Here."
        chunks = chunk_turn(text, fake_tokenizer, target_tokens=1000)
        assert len(chunks) == 1

    def test_all_content_preserved(self, fake_tokenizer):
        """No words should be dropped across all chunks."""
        text = "The quick brown fox. Jumps over the lazy dog. End of story here."
        original_words = set(text.replace(".", "").split())
        chunks = chunk_turn(text, fake_tokenizer, target_tokens=5)
        combined_words = set(" ".join(chunks).replace(".", "").split())
        assert original_words <= combined_words, "Words lost during chunking"
