"""Synthetic research papers with reference abstracts and gold QA, for the offline demo and the tests.

Each paper has seven paragraphs (introduction, related work, method, experiments, comparison,
limitations, conclusion) built from templates with random facts. The reference summary is an
abstract of at least 20 words. Each question has gold answers and the index of its evidence
paragraph. The text is random. It is not from real papers.
"""
from __future__ import annotations

import random

from .data import QAItem, Question, SummItem

TASKS = ["graph classification", "speech recognition", "protein folding", "image segmentation",
         "machine translation", "time series forecasting", "document retrieval", "molecule generation"]
ARCHS = ["transformer", "graph neural network", "convolutional network", "state space model", "diffusion model"]
COMPONENTS = ["sparse attention", "a learned positional code", "a contrastive loss", "a mixture of experts",
              "a memory bank", "low-rank adapters"]
DATASETS = ["OGB-Mol", "LibriLite", "CASP-Mini", "CityScenes", "WMT-Lite", "ETT-Hour", "MS-Docs", "ZINC-Small"]
METRICS = ["accuracy", "F1 score", "BLEU score", "mean IoU", "recall"]
LIMITS = ["the memory cost on long inputs", "the need for labelled data", "slow inference on CPUs",
          "sensitivity to the learning rate", "poor transfer to new domains"]
FILLER = [
    "Many earlier studies focus on small benchmarks.", "Prior work often reports results without variance.",
    "Some methods trade accuracy for speed.", "Recent surveys list open problems in this area.",
    "Reproducibility is still a concern for many methods.", "Several authors note a gap between benchmarks and practice.",
    "Large models dominate recent leaderboards.", "Data quality has a strong effect on the results.",
]


def _name(rng: random.Random) -> str:
    return rng.choice(["Graph", "Flux", "Sparse", "Echo", "Prism", "Nova", "Delta", "Quill"]) + \
        rng.choice(["Former", "Net", "Mixer", "Flow", "Gate", "Lens"]) + f"-{rng.randint(2, 9)}"


def make_paper(k: int, rng: random.Random) -> tuple[SummItem, QAItem]:
    m, task, arch, comp = _name(rng), rng.choice(TASKS), rng.choice(ARCHS), rng.choice(COMPONENTS)
    data, metric, lim = rng.choice(DATASETS), rng.choice(METRICS), rng.choice(LIMITS)
    base = _name(rng)
    while base == m:
        base = _name(rng)
    v = round(rng.uniform(60, 95), 1)
    v2 = round(v - rng.uniform(0.5, 6), 1)
    epochs, batch = rng.choice([10, 20, 30, 50, 100]), rng.choice([16, 32, 64, 128])
    fill = lambda n: " ".join(rng.sample(FILLER, n))  # noqa: E731
    paragraphs = [
        f"This paper studies {task}. {fill(2)} We introduce {m}, a new approach for {task}.",
        f"{fill(3)} Classical methods for {task} use hand-made features.",
        f"We propose {m}, a {arch} that uses {comp}. The design keeps the number of parameters small. {fill(1)}",
        f"We train {m} on {data} for {epochs} epochs with a batch size of {batch}. "
        f"{m} reaches an {metric} of {v} on {data}. {fill(1)}",
        f"We compare {m} with five baselines. The strongest baseline, {base}, reaches {v2} on {data}. {fill(1)}",
        f"The main limitation of {m} is {lim}. {fill(1)}",
        f"In summary, {m} improves {task} on {data}. Future work will study other datasets.",
    ]
    summary = (f"This paper proposes {m}, a {arch} for {task} that uses {comp}. On {data}, {m} reaches an "
               f"{metric} of {v}, compared with {v2} for the strongest baseline {base}. "
               f"The main limitation is {lim}.")
    questions = (
        Question(f"What {metric} does {m} reach on {data}?", (str(v), f"{metric} of {v}"), (3,)),
        Question("Which baseline is the strongest?", (base,), (4,)),
        Question(f"How many epochs is {m} trained for?", (str(epochs), f"{epochs} epochs"), (3,)),
        Question(f"What is the main limitation of {m}?", (lim,), (5,)),
        Question(f"Which component does {m} use?", (comp,), (2,)),
    )
    pid = f"paper-{k:04d}"
    return SummItem(pid, "\n\n".join(paragraphs), summary), QAItem(pid, tuple(paragraphs), questions)


def generate(n: int = 100, seed: int = 0) -> tuple[list[SummItem], list[QAItem]]:
    rng = random.Random(seed)
    pairs = [make_paper(k, rng) for k in range(n)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def to_rows(summ: list[SummItem], qa: list[QAItem]) -> tuple[list[dict], list[dict]]:
    s_rows = [{"id": s.id, "document": s.document, "summary": s.summary} for s in summ]
    q_rows = [{"id": q.id, "paragraphs": list(q.paragraphs), "questions": [
        {"question": x.question, "answers": list(x.answers), "evidence": list(x.evidence)} for x in q.questions]}
        for q in qa]
    return s_rows, q_rows
