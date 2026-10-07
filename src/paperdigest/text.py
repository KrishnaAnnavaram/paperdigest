"""Tokens, sentences, normalization and token-aware chunking.

The chunker packs whole sentences into chunks of at most `max_tokens` word tokens, with an overlap
of whole sentences. It never cuts a word. It cuts a sentence only if the sentence alone is longer
than `max_tokens`.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+(?:[.'-]\w+)*|[^\w\s]")
_WORD = re.compile(r"\w+(?:[.'-]\w+)*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENT.split(text) if s.strip()] if text else []


def normalize_answer(text: str) -> str:
    """Lower case, no punctuation, no articles, single spaces (SQuAD convention)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def metric_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    n_tokens: int
    first_sentence: int


def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[Chunk]:
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("need max_tokens >= 1 and 0 <= overlap_tokens < max_tokens")
    sents = []
    for s in sentences(text):
        w = words(s)
        if len(w) <= max_tokens:
            sents.append(s)
        else:  # a very long sentence: split it on word boundaries
            for k in range(0, len(w), max_tokens):
                sents.append(" ".join(w[k:k + max_tokens]))
    chunks: list[Chunk] = []
    start = 0
    while start < len(sents):
        end, total = start, 0
        while end < len(sents) and total + len(words(sents[end])) <= max_tokens:
            total += len(words(sents[end]))
            end += 1
        end = max(end, start + 1)
        chunks.append(Chunk(len(chunks), " ".join(sents[start:end]), total or len(words(sents[start])), start))
        if end >= len(sents):
            break
        # Step back whole sentences until the overlap budget is used.
        back, budget = end, overlap_tokens
        while back - 1 > start and len(words(sents[back - 1])) <= budget:
            back -= 1
            budget -= len(words(sents[back]))
        start = back
    return chunks
