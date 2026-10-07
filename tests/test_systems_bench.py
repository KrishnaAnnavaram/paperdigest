"""Problems 4 (non-summarization checkpoints), 5 (one failure stops the run), 6 (unused work),
7 (no ground truth) and 8 (retrieval never evaluated)."""
import pytest

from paperdigest.bench import markdown_table, run_qa, run_summarization
from paperdigest.cli import main
from paperdigest.llm import FakeLLM, LLMError, extract_answer
from paperdigest.synthetic import generate
from paperdigest.systems.qa import BM25Retriever, ClosedBookQA, HybridRetriever, RAGQA, TfidfRetriever
from paperdigest.systems.summarizers import (
    HF_REGISTRY,
    HFSummarizer,
    LeadSummarizer,
    MapReduceSummarizer,
    TextRankSummarizer,
)
from paperdigest.text import words


@pytest.fixture(scope="module")
def data():
    return generate(30, seed=3)


def test_registry_has_only_summarization_checkpoints():
    banned = {"facebook/mbart-large-50", "allenai/led-base-16384", "google/pegasus-large", "facebook/bart-base",
              "philschmid/bart-large-cnn-samsum"}
    assert not banned & {s.checkpoint for s in HF_REGISTRY.values()}
    assert len(HF_REGISTRY) + 3 == 20  # 17 checkpoints + lead + textrank + map-reduce
    assert all(s.max_input_tokens >= 512 for s in HF_REGISTRY.values())
    assert HF_REGISTRY["t5-base"].prefix == "summarize: "
    with pytest.raises(KeyError):
        HFSummarizer("mbart-large-50")


def test_extractive_summarizers_respect_budget(data):
    doc = data[0][0].document
    for s in (LeadSummarizer(40), TextRankSummarizer(40)):
        out = s.summarize(doc)
        assert 0 < len(words(out)) <= 60
        assert out == s.summarize(doc)  # deterministic


def test_mapreduce_propagates_llm_errors(data):
    class Broken:
        name = "broken"

        def complete(self, prompt, max_tokens=256):
            raise LLMError("down")

    with pytest.raises(LLMError):
        MapReduceSummarizer(Broken(), chunk_tokens=40, overlap=5).summarize(data[0][0].document)


def test_benchmark_records_failures_and_continues(data):
    class Exploding:
        name = "exploding"

        def summarize(self, document):
            raise RuntimeError("cannot load model")

    result = run_summarization(data[0][:10], [Exploding(), LeadSummarizer(60), TextRankSummarizer(60)], n_boot=50)
    names = [r["system"] for r in result["leaderboard"]]
    assert "exploding" not in names and len(names) == 2
    assert len(result["failures"]["exploding"]) == 10
    assert all("cannot load" not in r["prediction"] for r in result["records"])
    assert {"run_id", "prompt_version"} <= set(result["records"][0])
    assert result["comparison"]["metric"] == "rougeL"


def test_textrank_beats_lead_on_synthetic_papers(data):
    result = run_summarization(data[0], [LeadSummarizer(60), TextRankSummarizer(60)], n_boot=100)
    board = {r["system"]: r for r in result["leaderboard"]}
    assert board["textrank-60"]["rougeL"]["mean"] > board["lead-60"]["rougeL"]["mean"]


def test_retrievers_rank_the_evidence_first():
    passages = ["The cat sat on the mat.", "Transformers use attention layers.", "Rain fell all day."]
    for r in (BM25Retriever(), TfidfRetriever(), HybridRetriever([BM25Retriever(), TfidfRetriever()])):
        assert r.rank("Which layers do transformers use?", passages)[0] == 1


def test_qa_bench_has_gold_answers_and_retrieval_metrics(data):
    llm = FakeLLM()
    systems = [ClosedBookQA(llm), RAGQA(BM25Retriever(), llm, k=2)]
    result = run_qa(data[1][:10], systems, TextRankSummarizer(60), n_boot=50)
    board = {r["system"].split(":")[0]: r for r in result["leaderboard"]}
    assert board["rag"]["n_questions"] == 50
    assert "recall@1" in board["rag"] and "recall@1" not in board["closed-book-summary"]
    assert board["rag"]["f1"]["mean"] > board["closed-book-summary"]["f1"]["mean"]
    with pytest.raises(ValueError):
        ClosedBookQA(llm).answer("q", ["p"], None)


def test_offline_answer_rule():
    ctx = "GraphNet-3 reaches an accuracy of 87.3 on OGB-Mol. The strongest baseline, FluxNet-4, reaches 85.1."
    assert extract_answer("What accuracy does GraphNet-3 reach on OGB-Mol?", ctx) == "87.3"
    assert extract_answer("Which baseline is the strongest?", ctx) == "FluxNet-4"
    assert extract_answer("Who wrote it?", "") == "unanswerable"


def test_markdown_table():
    table = markdown_table([{"system": "x", "f1": {"mean": 0.5, "ci": (0.4, 0.6)}}], ["system", "f1", "mrr"])
    assert "0.500 (0.400-0.600)" in table and "| - |" in table


def test_cli_end_to_end(tmp_path, capsys):
    data_dir = tmp_path / "d"
    assert main(["synth", "--out", str(data_dir), "--papers", "12", "--seed", "1"]) == 0
    out = tmp_path / "s"
    assert main(["summarize-bench", "--data", str(data_dir / "summarization.jsonl"), "--n", "10",
                 "--bootstrap", "50", "--out", str(out)]) == 0
    assert (out / "predictions.jsonl").exists() and (out / "leaderboard.md").exists()
    assert main(["qa-bench", "--data", str(data_dir / "qa.jsonl"), "--n", "5", "--bootstrap", "50"]) == 0
    paper = tmp_path / "paper.txt"
    paper.write_text(generate(1, seed=2)[0][0].document, encoding="utf-8")
    capsys.readouterr()
    assert main(["digest", "--input", str(paper), "--question", "Which baseline is the strongest?"]) == 0
    assert "Q: Which baseline" in capsys.readouterr().out
    assert main(["models"]) == 0


def test_cli_errors(tmp_path, capsys):
    assert main(["summarize-bench", "--data", str(tmp_path / "none.jsonl")]) == 1
    assert main(["summarize-bench", "--data", str(tmp_path / "none.jsonl"), "--systems", "bogus"]) == 1
