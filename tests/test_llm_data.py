"""Problem 1 (hard-coded key), problem 9 (swallowed LLM errors), problems 2 and 3 (title as reference,
first-rows sample) and problem 6 (non-deterministic decoding)."""
import io
import json
import urllib.error

import pytest

from paperdigest.config import Settings
from paperdigest.data import DataError, load_qa, load_summarization, parse_qa, parse_summ, sample, write_jsonl
from paperdigest.llm import PROMPT_VERSION, FakeLLM, LLMError, OpenAICompatibleClient, make_client


def ok_response(text="A summary."):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": text}}]}).encode())


def test_key_comes_from_environment_and_is_not_shown(monkeypatch):
    monkeypatch.setenv("PAPERDIGEST_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-value-123")
    s = Settings.from_env()
    assert s.api_key == "test-value-123"
    assert "test-value-123" not in repr(s)
    client = make_client(s)
    assert "test-value-123" not in repr(client)


def test_missing_key_is_an_error():
    with pytest.raises(LLMError, match="no API key"):
        OpenAICompatibleClient("https://x", "m", "")


def test_request_is_deterministic():
    seen = {}

    def transport(request, timeout):
        seen["body"] = json.loads(request.data)
        return ok_response()

    client = OpenAICompatibleClient("https://x/v1", "m", "k", transport=transport, seed=7)
    assert client.complete("hi", 10) == "A summary."
    assert seen["body"]["temperature"] == 0 and seen["body"]["seed"] == 7


def test_retries_then_success_and_backoff():
    calls, sleeps = [], []

    def transport(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError("u", 429, "rate limit", {}, None)
        return ok_response("done")

    client = OpenAICompatibleClient("https://x", "m", "k", max_retries=3, transport=transport, sleep=sleeps.append)
    assert client.complete("p") == "done"
    assert sleeps == [1.0, 2.0]


def test_failure_raises_and_never_returns_error_text():
    def transport(request, timeout):
        raise urllib.error.URLError("down")

    client = OpenAICompatibleClient("https://x", "m", "k", max_retries=2, transport=transport, sleep=lambda s: None)
    with pytest.raises(LLMError, match="after 3 attempts"):
        client.complete("p")


def test_non_retry_status_and_bad_payload():
    def forbidden(request, timeout):
        raise urllib.error.HTTPError("u", 401, "no", {}, None)

    with pytest.raises(LLMError, match="HTTP 401"):
        OpenAICompatibleClient("https://x", "m", "k", transport=forbidden).complete("p")
    with pytest.raises(LLMError, match="format"):
        OpenAICompatibleClient("https://x", "m", "k", transport=lambda r, timeout: io.BytesIO(b"{}")).complete("p")
    with pytest.raises(LLMError, match="empty"):
        OpenAICompatibleClient("https://x", "m", "k", max_retries=0,
                               transport=lambda r, timeout: ok_response("  ")).complete("p")


def test_fake_llm_is_deterministic():
    fake = FakeLLM()
    prompt = "Summarize this part in at most 5 words.\n\nOne two three. Four five six. Seven."
    assert fake.complete(prompt) == fake.complete(prompt) == "One two three."
    assert PROMPT_VERSION


def test_title_is_rejected_as_a_reference_summary():
    with pytest.raises(DataError, match="title"):
        parse_summ({"id": "1", "document": "body text", "summary": "A Short Paper Title"})


def test_qa_schema():
    item = parse_qa({"id": "p", "paragraphs": ["a", "b"], "questions": [
        {"question": "q?", "answers": ["x"], "evidence": [1]}]})
    assert item.questions[0].evidence == (1,)
    with pytest.raises(DataError, match="out of range"):
        parse_qa({"id": "p", "paragraphs": ["a"], "questions": [{"question": "q", "answers": ["x"], "evidence": [3]}]})
    with pytest.raises(DataError):
        parse_qa({"id": "p", "paragraphs": []})


def test_sample_is_seeded_and_not_the_first_rows():
    items = list(range(1000))
    a = sample(items, 50, seed=1)
    assert a == sample(items, 50, seed=1)
    assert a != items[:50] and len(set(a)) == 50
    assert sample(items, None) == items


def test_jsonl_round_trip(tmp_path):
    rows = [{"id": "1", "document": "Body. " * 10, "summary": "word " * 25}]
    path = write_jsonl(tmp_path / "s.jsonl", rows)
    assert load_summarization(path)[0].id == "1"
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DataError, match="not valid JSON"):
        load_qa(bad)


def test_settings_checks(monkeypatch):
    monkeypatch.setenv("PAPERDIGEST_LLM_PROVIDER", "anthropic-direct")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("PAPERDIGEST_LLM_PROVIDER", "fake")
    monkeypatch.setenv("PAPERDIGEST_CHUNK_TOKENS", "50")
    monkeypatch.setenv("PAPERDIGEST_CHUNK_OVERLAP", "50")
    with pytest.raises(ValueError):
        Settings.from_env()
