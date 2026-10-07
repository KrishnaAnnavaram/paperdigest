"""Summarizers: two offline extractive baselines, an LLM map-reduce summarizer and Hugging Face models.

The Hugging Face registry lists only checkpoints that were fine-tuned (or instruction-tuned) for
summarization. Each entry gives the model prefix and the input limit in tokens. Inputs are
truncated by the tokenizer, never by characters, and the text keeps its case and punctuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..llm import PROMPTS, LLMClient
from ..text import chunk_text, sentences, words


class Summarizer(Protocol):
    name: str

    def summarize(self, document: str) -> str: ...


def _budget(sents: list[str], order: list[int], max_words: int) -> str:
    chosen, used = [], 0
    for i in order:
        n = len(words(sents[i]))
        if used + n > max_words and chosen:
            continue
        chosen.append(i)
        used += n
        if used >= max_words:
            break
    return " ".join(sents[i] for i in sorted(chosen))


class LeadSummarizer:
    """The first sentences up to the word budget. A strong baseline for news, weaker for papers."""

    def __init__(self, max_words: int = 150):
        self.name = f"lead-{max_words}"
        self.max_words = max_words

    def summarize(self, document: str) -> str:
        sents = sentences(document)
        return _budget(sents, list(range(len(sents))), self.max_words)


class TextRankSummarizer:
    """Sentence centrality on a TF-IDF cosine graph (PageRank, damping 0.85), in document order."""

    def __init__(self, max_words: int = 150, damping: float = 0.85, iterations: int = 50):
        self.name = f"textrank-{max_words}"
        self.max_words, self.damping, self.iterations = max_words, damping, iterations

    def summarize(self, document: str) -> str:
        sents = sentences(document)
        if len(sents) <= 2:
            return " ".join(sents)
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
        sim = (tfidf @ tfidf.T).toarray()
        np.fill_diagonal(sim, 0.0)
        rows = sim.sum(axis=1, keepdims=True)
        trans = np.divide(sim, rows, out=np.full_like(sim, 1.0 / len(sents)), where=rows > 0)
        score = np.full(len(sents), 1.0 / len(sents))
        for _ in range(self.iterations):
            score = (1 - self.damping) / len(sents) + self.damping * trans.T @ score
        order = list(np.argsort(-score, kind="stable"))
        return _budget(sents, [int(i) for i in order], self.max_words)


class MapReduceSummarizer:
    """Summarize each token-aware chunk with the LLM, then combine the partial summaries.

    `LLMError` propagates: a failed request stops this document, it never becomes summary text.
    """

    def __init__(self, llm: LLMClient, chunk_tokens: int = 400, overlap: int = 50, max_words: int = 150,
                 map_words: int = 80):
        self.llm = llm
        self.name = f"mapreduce:{llm.name}"
        self.chunk_tokens, self.overlap, self.max_words, self.map_words = chunk_tokens, overlap, max_words, map_words

    def summarize(self, document: str) -> str:
        chunks = chunk_text(document, self.chunk_tokens, self.overlap)
        partial = [self.llm.complete(PROMPTS["map"].format(n=self.map_words, text=c.text), self.map_words * 2)
                   for c in chunks]
        if len(partial) == 1:
            return partial[0]
        return self.llm.complete(PROMPTS["reduce"].format(n=self.max_words, text="\n\n".join(partial)),
                                 self.max_words * 2)


@dataclass(frozen=True)
class HFSpec:
    checkpoint: str
    prefix: str
    max_input_tokens: int
    trained_on: str


HF_REGISTRY: dict[str, HFSpec] = {
    "bart-large-cnn": HFSpec("facebook/bart-large-cnn", "", 1024, "CNN/DailyMail news"),
    "distilbart-cnn-12-6": HFSpec("sshleifer/distilbart-cnn-12-6", "", 1024, "CNN/DailyMail news"),
    "distilbart-cnn-6-6": HFSpec("sshleifer/distilbart-cnn-6-6", "", 1024, "CNN/DailyMail news"),
    "bart-large-xsum": HFSpec("facebook/bart-large-xsum", "", 1024, "XSum news"),
    "pegasus-arxiv": HFSpec("google/pegasus-arxiv", "", 1024, "arXiv papers"),
    "pegasus-pubmed": HFSpec("google/pegasus-pubmed", "", 1024, "PubMed papers"),
    "pegasus-cnn-dailymail": HFSpec("google/pegasus-cnn_dailymail", "", 1024, "CNN/DailyMail news"),
    "pegasus-xsum": HFSpec("google/pegasus-xsum", "", 512, "XSum news"),
    "bigbird-pegasus-arxiv": HFSpec("google/bigbird-pegasus-large-arxiv", "", 4096, "arXiv papers"),
    "bigbird-pegasus-pubmed": HFSpec("google/bigbird-pegasus-large-pubmed", "", 4096, "PubMed papers"),
    "led-large-arxiv": HFSpec("allenai/led-large-16384-arxiv", "", 8192, "arXiv papers"),
    "t5-small": HFSpec("t5-small", "summarize: ", 512, "multi-task, includes CNN/DailyMail"),
    "t5-base": HFSpec("t5-base", "summarize: ", 512, "multi-task, includes CNN/DailyMail"),
    "t5-large": HFSpec("t5-large", "summarize: ", 512, "multi-task, includes CNN/DailyMail"),
    "flan-t5-base": HFSpec("google/flan-t5-base", "Summarize this research paper: ", 512, "instruction-tuned"),
    "flan-t5-large": HFSpec("google/flan-t5-large", "Summarize this research paper: ", 512, "instruction-tuned"),
    "long-t5-pubmed": HFSpec("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps", "", 8192, "PubMed papers"),
}


class HFSummarizer:
    """A Hugging Face seq2seq summarizer (needs the `hf` extra). Beam search, no sampling."""

    def __init__(self, key: str, max_new_tokens: int = 256, num_beams: int = 4, device: str | None = None):
        if key not in HF_REGISTRY:
            raise KeyError(f"{key!r} is not in the registry of summarization checkpoints")
        self.spec = HF_REGISTRY[key]
        self.name = f"hf:{key}"
        self.max_new_tokens, self.num_beams, self.device = max_new_tokens, num_beams, device
        self._model = self._tok = None

    def _load(self):  # pragma: no cover - needs the hf extra and a download
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.spec.checkpoint)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.spec.checkpoint).eval()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)

    def summarize(self, document: str) -> str:  # pragma: no cover - needs the hf extra and a download
        if self._model is None:
            self._load()
        enc = self._tok(self.spec.prefix + document, truncation=True, max_length=self.spec.max_input_tokens,
                        return_tensors="pt").to(self.device)
        kwargs = {}
        if "led" in self.spec.checkpoint:
            mask = self._torch.zeros_like(enc["input_ids"])
            mask[:, 0] = 1
            kwargs["global_attention_mask"] = mask
        with self._torch.no_grad():
            out = self._model.generate(**enc, **kwargs, max_new_tokens=self.max_new_tokens,
                                       num_beams=self.num_beams, do_sample=False, early_stopping=True)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()

    def unload(self) -> None:
        self._model = self._tok = None
