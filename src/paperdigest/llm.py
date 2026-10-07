"""A provider-agnostic LLM client with retries, and a deterministic offline fake.

Rules:
- The API key comes from the environment (`Settings.api_key`). It is never logged or returned.
- Decoding is deterministic: temperature 0 and a fixed seed.
- A failed request raises `LLMError` after the retries. No error text ever flows into a summary.
- Prompts have a version (`PROMPT_VERSION`). Each benchmark record stores it.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol

from .text import sentences, words

PROMPT_VERSION = "2026-10-v1"
PROMPTS = {
    "map": "Summarize this part of a research paper in at most {n} words. Keep numbers exact.\n\n{text}",
    "reduce": "Combine these partial summaries of one research paper into one summary of at most {n} words. "
              "Keep numbers exact.\n\n{text}",
    "answer": "Answer the question with a short span from the context. If the context has no answer, "
              "write 'unanswerable'.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
}
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The LLM request failed after all retries, or the answer was not usable."""


class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str, max_tokens: int = 256) -> str: ...


class FakeLLM:
    """Offline and deterministic. It follows the three prompt templates with simple extractive rules."""

    name = "fake-extractive"

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        if prompt.startswith("Answer the question"):
            context = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
            question = prompt.split("Question:", 1)[1].rsplit("Answer:", 1)[0]
            return extract_answer(question, context)
        body = prompt.split("\n\n", 1)[1] if "\n\n" in prompt else prompt
        budget = min(max_tokens, int(prompt.split("at most ", 1)[1].split()[0]) if "at most " in prompt else max_tokens)
        out, used = [], 0
        for s in sentences(body):
            n = len(words(s))
            if used + n > budget and out:
                break
            out.append(s)
            used += n
        return " ".join(out)


def extract_answer(question: str, context: str) -> str:
    """Offline answer rule: find the sentence with the most question words, then take a short span.

    - "how many" or a question about a value: the first number in the sentence.
    - "which": the first name-like word (capital letter or digit) that is not in the question.
    - other questions: the words after the last question word in the sentence (at most 8).
    """
    q_words = [w.lower() for w in words(question)]
    q = {w for w in q_words if len(w) > 2}
    best, best_score = "", 0
    for s in sentences(context):
        score = len(q & {w.lower() for w in words(s)})
        if score > best_score:
            best, best_score = s, score
    if not best:
        return "unanswerable"
    tokens = words(best)
    lowered = question.lower()
    if lowered.startswith("how many") or any(k in q for k in ("reach", "score", "value", "accuracy")):
        numbers = [t for t in tokens if any(c.isdigit() for c in t) and t.replace(".", "").isdigit()]
        if numbers:
            return numbers[0]
    if lowered.startswith("which"):
        for t in tokens:
            if t.lower() not in q and (t[0].isupper() or any(c.isdigit() for c in t)) and tokens.index(t) > 0:
                return t
    last = max((i for i, t in enumerate(tokens) if t.lower() in q), default=-1)
    span = tokens[last + 1:last + 9]
    return " ".join(span) if span else best


class OpenAICompatibleClient:
    """Chat-completions client for OpenRouter, OpenAI or any compatible server. Standard library only."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: int = 60, max_retries: int = 3,
                 seed: int = 42, transport=urllib.request.urlopen, sleep=time.sleep):
        if not api_key:
            raise LLMError("no API key in the environment for this provider")
        self.name = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self.timeout, self.max_retries, self.seed = timeout, max_retries, seed
        self._transport, self._sleep = transport, sleep

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatibleClient(model={self.name!r}, url={self._url!r})"

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        body = json.dumps({"model": self.name, "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "seed": self.seed, "max_tokens": max_tokens}).encode()
        last = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(self._url, data=body, method="POST", headers={
                "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
            try:
                with self._transport(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise LLMError("the LLM answer was empty")
                return text.strip()
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
                if exc.code not in RETRY_STATUS:
                    raise LLMError(f"LLM request failed: {last}") from None
            except (urllib.error.URLError, TimeoutError) as exc:
                last = type(exc).__name__
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise LLMError(f"unexpected LLM answer format: {type(exc).__name__}") from None
            if attempt < self.max_retries:
                self._sleep(min(30.0, 2.0 ** attempt))
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last}")


def make_client(settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM()
    return OpenAICompatibleClient(settings.llm_base_url, settings.llm_model, settings.api_key,
                                  settings.llm_timeout, settings.llm_max_retries, settings.seed)
