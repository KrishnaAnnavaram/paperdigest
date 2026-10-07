"""Command-line interface: `paperdigest <command>`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bench import markdown_table, run_qa, run_summarization
from .config import Settings
from .data import DataError, load_qa, load_summarization, pdf_text, sample, write_jsonl
from .llm import LLMError, make_client
from .synthetic import generate, to_rows
from .systems.qa import BM25Retriever, ClosedBookQA, HybridRetriever, RAGQA, TfidfRetriever
from .systems.summarizers import HF_REGISTRY, HFSummarizer, LeadSummarizer, MapReduceSummarizer, TextRankSummarizer

SUMM_COLUMNS = ["system", "n_docs", "n_failed", "rouge1", "rouge2", "rougeL", "numbers_ok", "mean_words"]
QA_COLUMNS = ["system", "n_questions", "exact_match", "f1", "recall@1", "recall@3", "mrr"]


def _summarizers(names: list[str], settings: Settings, words: int):
    out = []
    for name in names:
        if name == "lead":
            out.append(LeadSummarizer(words))
        elif name == "textrank":
            out.append(TextRankSummarizer(words))
        elif name == "mapreduce":
            out.append(MapReduceSummarizer(make_client(settings), settings.chunk_tokens, settings.chunk_overlap, words))
        elif name.startswith("hf:"):
            out.append(HFSummarizer(name[3:]))
        else:
            raise ValueError(f"unknown summarizer {name!r}. Use lead, textrank, mapreduce or hf:<key>")
    return out


def _qa_systems(names: list[str], settings: Settings, k: int):
    llm = make_client(settings)
    table = {"bm25": BM25Retriever(), "tfidf": TfidfRetriever()}
    out = []
    for name in names:
        if name == "closed-book":
            out.append(ClosedBookQA(llm))
        elif name == "hybrid":
            out.append(RAGQA(HybridRetriever([BM25Retriever(), TfidfRetriever()]), llm, k))
        elif name in table:
            out.append(RAGQA(table[name], llm, k))
        else:
            raise ValueError(f"unknown QA system {name!r}. Use closed-book, bm25, tfidf or hybrid")
    return out


def _save(result: dict, out: Path | None, columns: list[str]) -> None:
    print(markdown_table(result["leaderboard"], columns))
    for name, errors in result["failures"].items():
        print(f"failures of {name}: {len(errors)} (first: {errors[0]})")
    if out:
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "predictions.jsonl", result["records"])
        summary = {k: v for k, v in result.items() if k != "records"}
        (out / "leaderboard.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        (out / "leaderboard.md").write_text(markdown_table(result["leaderboard"], columns) + "\n", encoding="utf-8")
        print(f"wrote {out / 'predictions.jsonl'} and {out / 'leaderboard.md'}")


def cmd_synth(args, settings) -> int:
    s_rows, q_rows = to_rows(*generate(args.papers, args.seed))
    print(f"wrote {write_jsonl(Path(args.out) / 'summarization.jsonl', s_rows)}")
    print(f"wrote {write_jsonl(Path(args.out) / 'qa.jsonl', q_rows)}")
    return 0


def cmd_models(args, settings) -> int:
    print("offline: lead, textrank, mapreduce (with PAPERDIGEST_LLM_PROVIDER)")
    for key, spec in HF_REGISTRY.items():
        print(f"hf:{key:24s} {spec.checkpoint:55s} max {spec.max_input_tokens:5d} tokens, {spec.trained_on}")
    return 0


def cmd_summarize_bench(args, settings) -> int:
    items = sample(load_summarization(args.data), args.n, settings.seed)
    result = run_summarization(items, _summarizers(args.systems, settings, args.words), args.bootstrap, settings.seed)
    print(f"run {result['run_id']}, prompt version {result['prompt_version']}, {len(items)} documents")
    _save(result, args.out, SUMM_COLUMNS)
    if result["comparison"]:
        c = result["comparison"]
        print(f"{c['a']} minus {c['b']} (ROUGE-L): {c['delta']:+.4f}, 95% CI {c['ci'][0]:+.4f} to {c['ci'][1]:+.4f}, "
              f"p = {c['p_value']:.3f}")
    return 0


def cmd_qa_bench(args, settings) -> int:
    items = sample(load_qa(args.data), args.n, settings.seed)
    summarizer = _summarizers([args.summarizer], settings, args.words)[0] if "closed-book" in args.systems else None
    result = run_qa(items, _qa_systems(args.systems, settings, args.k), summarizer, n_boot=args.bootstrap,
                    seed=settings.seed)
    print(f"run {result['run_id']}, prompt version {result['prompt_version']}, {len(items)} papers")
    _save(result, args.out, QA_COLUMNS)
    return 0


def cmd_digest(args, settings) -> int:
    path = Path(args.input)
    text = pdf_text(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    summarizer = _summarizers([args.summarizer], settings, args.words)[0]
    summary = summarizer.summarize(text)
    print(f"# Summary ({summarizer.name})\n\n{summary}\n")
    if args.question:
        from .text import chunk_text

        passages = [c.text for c in chunk_text(text, settings.chunk_tokens, settings.chunk_overlap)]
        qa = RAGQA(HybridRetriever([BM25Retriever(), TfidfRetriever()]), make_client(settings), args.k)
        for q in args.question:
            ans = qa.answer(q, passages)
            print(f"Q: {q}\nA: {ans.text}\n")
    return 0


def cmd_demo(args, settings) -> int:
    summ, qa = generate(args.papers, args.seed)
    print(f"synthetic data: {args.papers} papers (seed {args.seed}), {sum(len(q.questions) for q in qa)} questions")
    fake = Settings(**{**settings.__dict__, "llm_provider": "fake", "llm_model": "fake-extractive", "api_key": ""})
    result = run_summarization(summ, _summarizers(["lead", "textrank", "mapreduce"], fake, 60), args.bootstrap, fake.seed)
    print(markdown_table(result["leaderboard"], SUMM_COLUMNS))
    c = result["comparison"]
    print(f"{c['a']} minus {c['b']} (ROUGE-L): {c['delta']:+.4f}, 95% CI {c['ci'][0]:+.4f} to {c['ci'][1]:+.4f}, "
          f"p = {c['p_value']:.3f}")
    qa_result = run_qa(qa, _qa_systems(["closed-book", "bm25", "tfidf", "hybrid"], fake, 2),
                       TextRankSummarizer(60), n_boot=args.bootstrap, seed=fake.seed)
    print(markdown_table(qa_result["leaderboard"], QA_COLUMNS))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paperdigest", description="Paper summarization, QA and RAG benchmark")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="write synthetic summarization.jsonl and qa.jsonl")
    s.add_argument("--out", type=Path, default=Path("data/synthetic"))
    s.add_argument("--papers", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)
    m = sub.add_parser("models", help="list the summarizers")
    m.set_defaults(func=cmd_models)
    b = sub.add_parser("summarize-bench", help="score summarizers against reference summaries")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--systems", nargs="+", default=["lead", "textrank", "mapreduce"])
    b.add_argument("--n", type=int, default=500, help="seeded random sample size")
    b.add_argument("--words", type=int, default=150, help="summary word budget (extractive and LLM)")
    b.add_argument("--bootstrap", type=int, default=1000)
    b.add_argument("--out", type=Path)
    b.set_defaults(func=cmd_summarize_bench)
    q = sub.add_parser("qa-bench", help="score QA systems against gold answers and evidence")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--systems", nargs="+", default=["closed-book", "bm25", "tfidf", "hybrid"])
    q.add_argument("--summarizer", default="textrank")
    q.add_argument("--words", type=int, default=150)
    q.add_argument("--k", type=int, default=3)
    q.add_argument("--n", type=int, default=500)
    q.add_argument("--bootstrap", type=int, default=1000)
    q.add_argument("--out", type=Path)
    q.set_defaults(func=cmd_qa_bench)
    d = sub.add_parser("digest", help="summarize one paper (text or PDF) and answer questions")
    d.add_argument("--input", type=Path, required=True)
    d.add_argument("--summarizer", default="textrank")
    d.add_argument("--words", type=int, default=150)
    d.add_argument("--question", action="append")
    d.add_argument("--k", type=int, default=3)
    d.set_defaults(func=cmd_digest)
    e = sub.add_parser("demo", help="offline benchmark on synthetic papers")
    e.add_argument("--papers", type=int, default=100)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--bootstrap", type=int, default=1000)
    e.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        return args.func(args, settings)
    except (DataError, LLMError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
