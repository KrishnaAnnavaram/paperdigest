"""Metrics, with no heavy dependency.

- ROUGE-1, ROUGE-2 and ROUGE-L F1 on normalized word tokens (no stemming).
- SQuAD exact match and token F1, with the maximum over all gold answers.
- Retrieval recall@k and MRR on gold evidence paragraphs.
- A number-consistency score: the share of numbers in a summary that also occur in the source.
  It is a cheap faithfulness proxy, not a full factuality metric.
- Bootstrap intervals over documents and a paired bootstrap between two systems.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from .text import metric_tokens


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if n_pred == 0 or n_ref == 0 or overlap == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int) -> float:
    p, r = _ngrams(metric_tokens(prediction), n), _ngrams(metric_tokens(reference), n)
    return _f1(sum((p & r).values()), sum(p.values()), sum(r.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    p, r = metric_tokens(prediction), metric_tokens(reference)
    return _f1(_lcs(p, r), len(p), len(r))


def rouge(prediction: str, reference: str) -> dict[str, float]:
    return {"rouge1": rouge_n(prediction, reference, 1), "rouge2": rouge_n(prediction, reference, 2),
            "rougeL": rouge_l(prediction, reference)}


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred = " ".join(metric_tokens(prediction))
    return float(any(pred == " ".join(metric_tokens(g)) for g in golds))


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    pred = metric_tokens(prediction)
    best = 0.0
    for g in golds:
        gold = metric_tokens(g)
        common = Counter(pred) & Counter(gold)
        best = max(best, _f1(sum(common.values()), len(pred), len(gold)))
    return best


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    for pos, item in enumerate(ranked, 1):
        if item in relevant:
            return 1.0 / pos
    return 0.0


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def number_consistency(summary: str, source: str) -> float:
    """Share of numbers in the summary that occur in the source. 1.0 when the summary has no number."""
    nums = _NUMBER.findall(summary)
    if not nums:
        return 1.0
    present = set(_NUMBER.findall(source))
    return sum(n in present for n in nums) / len(nums)


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 42, level: float = 0.95) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), (n_boot, len(arr)))].mean(axis=1)
    alpha = (1 - level) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(low), float(high)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 42) -> dict:
    """Mean of a - b over the same documents, with a CI and a two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired scores need the same documents in the same order")
    diff = a - b
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return {"delta": float(diff.mean()), "ci": (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))),
            "p_value": float(min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean())))}


def mean_with_ci(values: Sequence[float], n_boot: int, seed: int) -> dict:
    arr = [v for v in values if np.isfinite(v)]
    return {"mean": float(np.mean(arr)) if arr else float("nan"), "ci": bootstrap_ci(arr, n_boot, seed), "n": len(arr)}


Scorer = Callable[[str, str], float]
