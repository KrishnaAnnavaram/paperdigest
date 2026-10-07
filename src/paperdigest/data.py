"""Datasets with schema checks, seeded samples and optional loaders (Hugging Face, PDF).

Summarization item: `id`, `document` (the paper body) and `summary` (a real reference, for example the
abstract). A title is never a reference summary. QA item: `id`, `paragraphs` and `questions`, each
question with `question`, `answers` (one or more gold strings) and `evidence` (paragraph indices).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

MIN_SUMMARY_WORDS = 20


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class SummItem:
    id: str
    document: str
    summary: str


@dataclass(frozen=True)
class Question:
    question: str
    answers: tuple[str, ...]
    evidence: tuple[int, ...] = ()


@dataclass(frozen=True)
class QAItem:
    id: str
    paragraphs: tuple[str, ...]
    questions: tuple[Question, ...] = field(default_factory=tuple)

    @property
    def document(self) -> str:
        return "\n\n".join(self.paragraphs)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{n}: not valid JSON") from exc
    return rows


def parse_summ(row: dict, where: str = "") -> SummItem:
    for key in ("id", "document", "summary"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise DataError(f"{where}: field {key!r} must be a non-empty string")
    if len(row["summary"].split()) < MIN_SUMMARY_WORDS:
        raise DataError(f"{where}: the reference summary has fewer than {MIN_SUMMARY_WORDS} words. "
                        "A title is not a summary.")
    return SummItem(row["id"], row["document"], row["summary"])


def parse_qa(row: dict, where: str = "") -> QAItem:
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs or not all(isinstance(p, str) for p in paragraphs):
        raise DataError(f"{where}: 'paragraphs' must be a non-empty list of strings")
    questions = []
    for k, q in enumerate(row.get("questions", [])):
        answers = q.get("answers")
        if not q.get("question") or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
            raise DataError(f"{where}: question {k} needs 'question' and a non-empty 'answers' list")
        evidence = tuple(int(e) for e in q.get("evidence", []))
        if any(e < 0 or e >= len(paragraphs) for e in evidence):
            raise DataError(f"{where}: question {k} has an evidence index out of range")
        questions.append(Question(q["question"], tuple(answers), evidence))
    return QAItem(str(row.get("id", where)), tuple(paragraphs), tuple(questions))


def load_summarization(path: str | Path) -> list[SummItem]:
    return [parse_summ(r, f"{path}:{i + 1}") for i, r in enumerate(_read_jsonl(Path(path)))]


def load_qa(path: str | Path) -> list[QAItem]:
    return [parse_qa(r, f"{path}:{i + 1}") for i, r in enumerate(_read_jsonl(Path(path)))]


def sample(items: list, n: int | None, seed: int = 42) -> list:
    """A seeded random sample, never the first rows of the file."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return [items[i] for i in sorted(rng.sample(range(len(items)), n))]


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def load_hf_summarization(name: str = "ccdv/arxiv-summarization", split: str = "test", n: int = 500,
                          seed: int = 42) -> list[SummItem]:  # pragma: no cover - needs network and the hf extra
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    rows = sample(list(range(len(ds))), n, seed)
    return [SummItem(f"{name}:{i}", ds[i]["article"], ds[i]["abstract"]) for i in rows]


def pdf_text(path: str | Path) -> str:  # pragma: no cover - needs the pdf extra
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError('PDF input needs the optional extra: pip install -e ".[pdf]"') from exc
    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)
