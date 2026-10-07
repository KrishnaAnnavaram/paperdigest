# Data

paperdigest does not include datasets, papers or predictions. Git ignores every file in this folder
except this README.

## Recommended datasets

| Use | Dataset | Source | Licence / terms |
|---|---|---|---|
| Summarization (paper body → abstract) | `ccdv/arxiv-summarization` | Hugging Face Hub | Check the dataset card. arXiv papers keep the licence that each author chose |
| Summarization (paper body → abstract) | `ccdv/pubmed-summarization` | Hugging Face Hub | Check the dataset card (PubMed Open Access subset) |
| Short summaries | SciTLDR (`allenai/scitldr`) | Hugging Face Hub | Apache-2.0 (check the card) |
| QA with gold answers and evidence | Qasper (`allenai/qasper`) | Hugging Face Hub | CC BY 4.0 (check the card) |

A paper title is not a reference summary. The loader refuses a reference summary with fewer than 20 words.

## File formats

`summarization.jsonl`, one JSON object for each line:

```json
{"id": "paper-0001", "document": "Full text of the paper ...", "summary": "The reference abstract ..."}
```

`qa.jsonl`, one JSON object for each line:

```json
{"id": "paper-0001", "paragraphs": ["Paragraph 0 ...", "Paragraph 1 ..."],
 "questions": [{"question": "Which baseline is the strongest?", "answers": ["FluxNet-4"], "evidence": [1]}]}
```

`evidence` lists the indices of the paragraphs that contain the answer. Retrieval metrics use it.

## Procedure

1. Install the `hf` extra: `pip install -e ".[hf]"`.
2. Load a dataset with `paperdigest.data.load_hf_summarization` (seeded random sample), or convert Qasper to the QA format above.
3. Write the files with `paperdigest.data.write_jsonl` into `data/`.
4. Run `paperdigest summarize-bench --data data/summarization.jsonl` or `paperdigest qa-bench --data data/qa.jsonl`.

## Synthetic data (no download)

`paperdigest synth --out data/synthetic` writes both files for 100 synthetic papers. The text is
random. It is not from real papers, and results on it say nothing about real models.
