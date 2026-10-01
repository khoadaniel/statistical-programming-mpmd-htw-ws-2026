"""Tests for the exercises in README.md. They are expected to fail (xfail) until you solve them."""

import pytest

from review_assistant.agent import run_agent
from review_assistant.chunking import chunk_sentences
from review_assistant.llm import FakeChatClient, message, tool_call
from review_assistant.tools import ToolBox, make_search_tool

exercise = pytest.mark.xfail(raises=NotImplementedError, strict=False, reason="exercise not solved yet")


@exercise
def test_exercise2_chunk_sentences():
    text = "One two three. Four five six seven! Eight nine? Ten."
    assert chunk_sentences(text, max_words=7) == ["One two three. Four five six seven!", "Eight nine? Ten."]
    assert chunk_sentences("", max_words=5) == []
    long = "a b c d e f g h."
    assert chunk_sentences(long + " Short one.", max_words=3) == [long, "Short one."]


@exercise
def test_exercise3_token_budget(retriever):
    client = FakeChatClient(lambda m, t: message(tool_calls=[tool_call("search_reviews", {"query": "x"})]),
                            tokens_per_call=(400, 100))
    result = run_agent("q", client, "fake", ToolBox([make_search_tool(retriever)]), max_steps=10,
                       max_total_tokens=1200)
    assert result.stop_reason == "budget" and result.answer is None
    assert result.llm_calls == 3  # 500, 1000, 1500 > 1200: stop after the third call
