"""Problem 8 (character chunking), problem 3 (no confidence) and the metric implementations."""
import numpy as np
import pytest

from paperdigest.metrics import (
    bootstrap_ci,
    exact_match,
    number_consistency,
    paired_bootstrap,
    recall_at_k,
    reciprocal_rank,
    rouge,
    rouge_l,
    rouge_n,
    token_f1,
)
from paperdigest.text import chunk_text, normalize_answer, sentences, words

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 3} with some extra words." for i in range(40))


def test_sentences_and_words():
    assert sentences("First one. Second one! Third?") == ["First one.", "Second one!", "Third?"]
    assert words("GraphNet-3 reaches 87.3.") == ["GraphNet-3", "reaches", "87.3"]


def test_chunks_respect_the_token_budget_and_never_cut_words():
    chunks = chunk_text(TEXT, max_tokens=40, overlap_tokens=12)
    assert len(chunks) > 3
    all_words = set(words(TEXT))
    for c in chunks:
        assert c.n_tokens <= 40
        assert set(words(c.text)) <= all_words  # no broken word
        assert c.text.endswith(".")             # whole sentences
    assert chunks[1].first_sentence < chunks[0].first_sentence + len(sentences(chunks[0].text))  # overlap


def test_long_sentence_is_split_on_word_boundaries():
    long = " ".join(f"w{i}" for i in range(95)) + "."
    chunks = chunk_text(long, max_tokens=30, overlap_tokens=0)
    assert all(c.n_tokens <= 30 for c in chunks)
    assert sum(c.n_tokens for c in chunks) == 95


@pytest.mark.parametrize("bad", [(0, 0), (10, 10), (10, -1)])
def test_chunk_parameters_are_checked(bad):
    with pytest.raises(ValueError):
        chunk_text("A b c.", *bad)


def test_rouge_known_values():
    assert rouge_n("the cat sat", "the cat sat", 1) == 1.0
    assert rouge_n("the cat", "a dog", 1) == 0.0
    # "cat sat on mat" vs "cat on the mat": unigram overlap after normalization (articles removed).
    assert rouge_n("cat sat on mat", "cat on mat", 1) == pytest.approx(2 * (3 / 4) * 1 / (3 / 4 + 1))
    assert rouge_l("w b c d", "w c d") == pytest.approx(2 * (3 / 4) * 1 / (3 / 4 + 1))
    r = rouge("x y z", "x y z")
    assert r == {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}


def test_rouge_against_reference_implementation():
    rs = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    pred, ref = "graph models reach high accuracy", "graph models reach a high test accuracy"
    ours = rouge(pred, ref)
    theirs = scorer.score(ref, pred)
    assert ours["rouge1"] == pytest.approx(theirs["rouge1"].fmeasure, abs=0.05)


def test_squad_metrics():
    assert normalize_answer("The  Cat!") == "cat"
    assert exact_match("the 87.3", ["87.3"]) == 1.0
    assert exact_match("87", ["87.3"]) == 0.0
    assert token_f1("memory cost", ["the memory cost on long inputs"]) == pytest.approx(2 * 1 * 0.4 / 1.4)
    assert token_f1("x", ["y", "x"]) == 1.0


def test_retrieval_metrics():
    assert recall_at_k([3, 1, 2], {1}, 1) == 0.0
    assert recall_at_k([3, 1, 2], {1}, 2) == 1.0
    assert reciprocal_rank([3, 1, 2], {1}) == 0.5
    assert reciprocal_rank([3], {1}) == 0.0
    assert np.isnan(recall_at_k([1], set(), 1))


def test_number_consistency():
    assert number_consistency("reaches 87.3 and 12", "We report 87.3 here.") == 0.5
    assert number_consistency("no numbers", "1 2 3") == 1.0


def test_bootstrap_helpers():
    lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4], n_boot=200)
    assert 0.1 <= lo <= 0.25 <= hi <= 0.4
    r = paired_bootstrap([1, 1, 1, 1], [0, 0, 0, 0], n_boot=50)
    assert r["delta"] == 1 and r["p_value"] == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])
