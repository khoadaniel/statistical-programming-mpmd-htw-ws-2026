"""LLM access through any OpenAI-compatible endpoint, configured by environment variables.

    LLM_BASE_URL  default http://localhost:11434/v1   (Ollama on your own machine)
    LLM_MODEL     default llama3.2                    (ollama pull llama3.2)
    LLM_API_KEY   default "ollama"                    (Ollama ignores it; real providers need one)

Other providers that speak the same protocol: OpenAI (https://api.openai.com/v1), a university
server with vLLM, LM Studio, Groq, OpenRouter and others. Never write a key into code or notebooks.

`FakeChatClient` imitates `client.chat.completions.create` so that the tests run offline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("LLM_API_KEY", "ollama"),
        )


def make_client(config: LLMConfig | None = None):
    """An `openai.OpenAI` client for the configured endpoint, and the model name to use."""
    from openai import OpenAI

    config = config or LLMConfig.from_env()
    return OpenAI(base_url=config.base_url, api_key=config.api_key), config.model


# ---------------------------------------------------------------- fake client for tests


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> SimpleNamespace:
    """A tool call in the shape the OpenAI SDK returns."""
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def message(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or None)


def _response(msg: SimpleNamespace, prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


@dataclass
class FakeChatClient:
    """Returns scripted replies. `script` is a list of messages, or a function (messages, tools) -> message.

    Every call is recorded in `calls`, so a test can check what the model was sent.
    """

    script: list[SimpleNamespace] | Callable[[list[dict], list[dict] | None], SimpleNamespace]
    tokens_per_call: tuple[int, int] = (100, 20)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict], tools: list[dict] | None = None, **kwargs):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages], "tools": tools, **kwargs})
        if callable(self.script):
            msg = self.script(messages, tools)
        else:
            if not self.script:
                raise RuntimeError("the fake client has no scripted reply left")
            msg = self.script.pop(0)
        return _response(msg, *self.tokens_per_call)
