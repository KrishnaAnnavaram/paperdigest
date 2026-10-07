"""Retrievers and question-answering systems.

- `BM25Retriever`: Okapi BM25 (k1 = 1.5, b = 0.75) on word tokens.
- `TfidfRetriever`: cosine similarity of TF-IDF vectors (an offline stand-in for a dense encoder).
- `DenseRetriever`: sentence-transformers embeddings (optional `dense` extra).
- `HybridRetriever`: reciprocal rank fusion (k = 60) of two or more retrievers.
- `RAGQA`: retrieve the top-k paragraphs, then ask the LLM with the answer prompt.
- `ClosedBookQA`: ask the LLM with the summary of the paper as the only context.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..llm import PROMPTS, LLMClient
from ..text import words


class Retriever(Protocol):
    name: str

    def rank(self, query: str, passages: list[str]) -> list[int]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        docs = [[w.lower() for w in words(p)] for p in passages]
        avg = sum(map(len, docs)) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        out = np.zeros(n)
        for t in {w.lower() for w in words(query)}:
            if t not in df:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            for i, d in enumerate(docs):
                tf = d.count(t)
                if tf:
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(d) / (avg or 1)))
        return out

    def rank(self, query: str, passages: list[str]) -> list[int]:
        return [int(i) for i in np.argsort(-self.scores(query, passages), kind="stable")]


class TfidfRetriever:
    name = "tfidf"

    def rank(self, query: str, passages: list[str]) -> list[int]:
        vec = TfidfVectorizer().fit(passages + [query])
        sims = (vec.transform(passages) @ vec.transform([query]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-sims, kind="stable")]


class DenseRetriever:  # pragma: no cover - needs the dense extra and a download
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = f"dense:{model.split('/')[-1]}"
        self.model = SentenceTransformer(model)

    def rank(self, query: str, passages: list[str]) -> list[int]:
        emb = self.model.encode(passages + [query], normalize_embeddings=True)
        return [int(i) for i in np.argsort(-(emb[:-1] @ emb[-1]), kind="stable")]


class HybridRetriever:
    def __init__(self, retrievers: list, k: int = 60):
        self.retrievers, self.k = retrievers, k
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def rank(self, query: str, passages: list[str]) -> list[int]:
        fused = np.zeros(len(passages))
        for r in self.retrievers:
            for pos, idx in enumerate(r.rank(query, passages)):
                fused[idx] += 1.0 / (self.k + pos + 1)
        return [int(i) for i in np.argsort(-fused, kind="stable")]


@dataclass
class Answer:
    text: str
    ranked: list[int] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


class RAGQA:
    def __init__(self, retriever, llm: LLMClient, k: int = 3):
        self.retriever, self.llm, self.k = retriever, llm, k
        self.name = f"rag:{retriever.name}:k{k}:{llm.name}"

    def answer(self, question: str, paragraphs: list[str], summary: str | None = None) -> Answer:
        ranked = self.retriever.rank(question, paragraphs)
        contexts = [paragraphs[i] for i in ranked[: self.k]]
        text = self.llm.complete(PROMPTS["answer"].format(context="\n\n".join(contexts), question=question), 64)
        return Answer(text.strip(), ranked, contexts)


class ClosedBookQA:
    """Answers from the summary only. It shows what a summary keeps and what it loses."""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.name = f"closed-book-summary:{llm.name}"

    def answer(self, question: str, paragraphs: list[str], summary: str | None = None) -> Answer:
        if summary is None:
            raise ValueError("closed-book QA needs a summary")
        text = self.llm.complete(PROMPTS["answer"].format(context=summary, question=question), 64)
        return Answer(text.strip(), [], [summary])
