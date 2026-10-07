"""Benchmarks: summarization and question answering, with records, failures and leaderboards.

A system that fails to load or fails on a document is recorded with its error. The benchmark goes on
with the next system or document. A failure never becomes a prediction.
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict

import numpy as np

from .data import QAItem, SummItem
from .llm import PROMPT_VERSION
from .metrics import (
    exact_match,
    mean_with_ci,
    number_consistency,
    paired_bootstrap,
    reciprocal_rank,
    recall_at_k,
    rouge,
    token_f1,
)
from .text import words

log = logging.getLogger("paperdigest")
SUMM_METRICS = ("rouge1", "rouge2", "rougeL", "numbers_ok")


def run_summarization(items: list[SummItem], systems: list, n_boot: int = 1000, seed: int = 42) -> dict:
    run_id = uuid.uuid4().hex[:8]
    records, failures = [], defaultdict(list)
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for system in systems:
        for item in items:
            try:
                pred = system.summarize(item.document)
            except Exception as exc:  # noqa: BLE001 - record any failure and go on
                failures[system.name].append(f"{item.id}: {type(exc).__name__}: {exc}")
                log.warning("%s failed on %s: %s", system.name, item.id, exc)
                continue
            s = rouge(pred, item.summary)
            s["numbers_ok"] = number_consistency(pred, item.document)
            s["words"] = float(len(words(pred)))
            scores[system.name][item.id] = s
            records.append({"run_id": run_id, "prompt_version": PROMPT_VERSION, "system": system.name,
                            "doc_id": item.id, "prediction": pred, **s})
    board = []
    for name, per_doc in scores.items():
        row = {"system": name, "n_docs": len(per_doc), "n_failed": len(failures.get(name, [])),
               "mean_words": float(np.mean([s["words"] for s in per_doc.values()]))}
        for m in SUMM_METRICS:
            row[m] = mean_with_ci([s[m] for s in per_doc.values()], n_boot, seed)
        board.append(row)
    board.sort(key=lambda r: -r["rougeL"]["mean"])
    comparison = {}
    if len(board) >= 2:
        a, b = board[0]["system"], board[1]["system"]
        common = sorted(set(scores[a]) & set(scores[b]))
        if common:
            comparison = {"a": a, "b": b, "metric": "rougeL", **paired_bootstrap(
                [scores[a][d]["rougeL"] for d in common], [scores[b][d]["rougeL"] for d in common], n_boot, seed)}
    return {"run_id": run_id, "prompt_version": PROMPT_VERSION, "leaderboard": board, "comparison": comparison,
            "failures": dict(failures), "records": records}


def run_qa(items: list[QAItem], systems: list, summarizer=None, ks=(1, 3, 5), n_boot: int = 1000,
           seed: int = 42) -> dict:
    run_id = uuid.uuid4().hex[:8]
    summaries = {}
    if summarizer is not None:
        summaries = {item.id: summarizer.summarize(item.document) for item in items}
    records, failures, board = [], defaultdict(list), []
    for system in systems:
        em, f1, rr, rec = [], [], [], {k: [] for k in ks}
        for item in items:
            for q in item.questions:
                try:
                    ans = system.answer(q.question, list(item.paragraphs), summaries.get(item.id))
                except Exception as exc:  # noqa: BLE001 - record any failure and go on
                    failures[system.name].append(f"{item.id}: {type(exc).__name__}: {exc}")
                    continue
                em.append(exact_match(ans.text, q.answers))
                f1.append(token_f1(ans.text, q.answers))
                if ans.ranked and q.evidence:
                    relevant = set(q.evidence)
                    rr.append(reciprocal_rank(ans.ranked, relevant))
                    for k in ks:
                        rec[k].append(recall_at_k(ans.ranked, relevant, k))
                records.append({"run_id": run_id, "prompt_version": PROMPT_VERSION, "system": system.name,
                                "doc_id": item.id, "question": q.question, "prediction": ans.text,
                                "gold": list(q.answers), "em": em[-1], "f1": f1[-1]})
        row = {"system": system.name, "n_questions": len(em), "n_failed": len(failures.get(system.name, [])),
               "exact_match": mean_with_ci(em, n_boot, seed), "f1": mean_with_ci(f1, n_boot, seed)}
        if rr:
            row["mrr"] = mean_with_ci(rr, n_boot, seed)
            for k in ks:
                row[f"recall@{k}"] = mean_with_ci(rec[k], n_boot, seed)
        board.append(row)
    board.sort(key=lambda r: -r["f1"]["mean"] if r["n_questions"] else 0.0)
    return {"run_id": run_id, "prompt_version": PROMPT_VERSION, "leaderboard": board, "failures": dict(failures),
            "records": records}


def _fmt(cell) -> str:
    if isinstance(cell, dict) and "mean" in cell:
        lo, hi = cell["ci"]
        return f"{cell['mean']:.3f} ({lo:.3f}-{hi:.3f})"
    return f"{cell:.1f}" if isinstance(cell, float) else str(cell)


def markdown_table(board: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in board:
        lines.append("| " + " | ".join(_fmt(row.get(c, "-")) for c in columns) + " |")
    return "\n".join(lines)
