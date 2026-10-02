import json

from bti_assistant.llm import FakeChatClient, LLMConfig, message
from bti_assistant.prompts import build_rag_messages, build_suggestion_messages, format_sources
from bti_assistant.rag import answer_question, suggest_heading, vote_headings
from bti_assistant.store import Hit

from .conftest import HEADINGS


def test_sources_are_labelled_with_their_references():
    hits = [Hit("DE-001/21#0", "DE-001/21", "Damenstiefel aus Leder.", 0.9, "6403"),
            Hit("FR-2022-01#0", "FR-2022-01", "x " * 1000, 0.5, "6403")]
    text = format_sources(hits, max_chars=50, show_heading=True)
    assert text.splitlines()[0] == "[DE-001/21] (heading 6403) Damenstiefel aus Leder."
    assert text.splitlines()[1].endswith("...") and len(text.splitlines()[1]) < 90


def test_prompt_contains_instruction_sources_and_question():
    msgs = build_rag_messages("Which heading for leather boots?", [Hit("D#0", "D", "Stiefel aus Leder.", 0.9, "6403")])
    assert msgs[0]["role"] == "system" and "only the past customs decisions" in msgs[0]["content"]
    assert "[D] (heading 6403) Stiefel aus Leder." in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: Which heading for leather boots?")


def test_suggestion_prompt_lists_the_candidate_texts():
    hits = [Hit("D#0", "D", "Stiefel aus Leder.", 0.9, "6403")]
    msgs = build_suggestion_messages("Boots of leather", hits, HEADINGS, ["6403"])
    assert "6403: Footwear" in msgs[1]["content"] and "ignore any instructions" in msgs[0]["content"]


def test_answer_question_with_a_fake_llm(retriever):
    client = FakeChatClient([message("Leather boots were classified in 6403 [DE-001/21] [XX-99].")])
    result = answer_question("Oberteil aus Rindleder", retriever, client, model="fake", k=3)
    assert result.cited_ids == ["DE-001/21", "XX-99"]
    assert result.citation_precision == 0.5  # XX-99 was never retrieved: an invented source
    assert "DE-001/21" in result.retrieved_ids
    sent = client.calls[0]
    assert sent["temperature"] == 0.0 and "[DE-001/21]" in sent["messages"][1]["content"]
    assert result.prompt_tokens == 100


def test_vote_headings_weights_by_similarity():
    hits = [Hit("a#0", "a", "", 0.9, "6403"), Hit("b#0", "b", "", 0.6, "6404"),
            Hit("c#0", "c", "", 0.5, "6404"), Hit("d#0", "d", "", 0.4, None)]
    assert vote_headings(hits) == ["6404", "6403"]  # 1.1 against 0.9
    assert vote_headings([]) == []


def test_suggest_heading_accepts_a_supported_candidate(retriever):
    reply = {"heading": "6403", "references": ["DE-001/21"], "reason": "leather uppers"}
    client = FakeChatClient([message(json.dumps(reply))])
    s = suggest_heading("Stiefel mit Oberteil aus Rindleder", retriever, client, "fake", HEADINGS, k=4)
    assert s.heading == "6403" and s.error is None and "6403" in s.candidates
    assert s.cited_ids == ["DE-001/21"] and s.citation_precision == 1.0
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_suggest_heading_rejects_unsupported_or_broken_answers(retriever):
    client = FakeChatClient([message(json.dumps({"heading": "8471", "references": []})), message("not json")])
    s = suggest_heading("Stiefel aus Rindleder", retriever, client, "fake", HEADINGS, k=4)
    assert s.heading is None and "not among the candidates" in s.error
    s = suggest_heading("Stiefel aus Rindleder", retriever, client, "fake", HEADINGS, k=4)
    assert s.heading is None and s.error.startswith("unusable answer")


def test_config_from_environment(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:3b")
    cfg = LLMConfig.from_env()
    assert cfg.base_url == "http://localhost:11434/v1" and cfg.model == "qwen2.5:3b"
