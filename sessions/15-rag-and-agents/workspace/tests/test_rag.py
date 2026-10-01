from review_assistant.llm import FakeChatClient, LLMConfig, message
from review_assistant.prompts import build_rag_messages, format_sources
from review_assistant.rag import answer_question
from review_assistant.store import Hit


def test_sources_are_labelled_with_their_ids():
    hits = [Hit("r1#0", "r1", "Will not connect.", 0.9), Hit("r3#0", "r3", "x " * 1000, 0.5)]
    text = format_sources(hits, max_chars=50)
    assert text.splitlines()[0] == "[r1] Will not connect."
    assert text.splitlines()[1].endswith("...") and len(text.splitlines()[1]) < 70


def test_prompt_contains_instruction_sources_and_question():
    msgs = build_rag_messages("Does it connect?", [Hit("r1#0", "r1", "Will not connect.", 0.9)])
    assert msgs[0]["role"] == "system" and "only the customer reviews" in msgs[0]["content"]
    assert "[r1] Will not connect." in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: Does it connect?")


def test_answer_question_with_a_fake_llm(retriever):
    client = FakeChatClient([message("Several owners could not connect the scale to WiFi [r1] [r99].")])
    result = answer_question("Does the scale connect to wifi?", retriever, client, model="fake", k=3)
    assert result.cited_ids == ["r1", "r99"]
    assert result.citation_precision == 0.5  # r99 was never retrieved: an invented source
    assert "r1" in result.retrieved_ids
    sent = client.calls[0]
    assert sent["temperature"] == 0.0 and "[r1]" in sent["messages"][1]["content"]
    assert result.prompt_tokens == 100


def test_config_from_environment(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:3b")
    cfg = LLMConfig.from_env()
    assert cfg.base_url == "http://localhost:11434/v1" and cfg.model == "qwen2.5:3b"
