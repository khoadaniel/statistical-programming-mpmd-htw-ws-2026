"""The RAG pipeline: retrieve, assemble, generate, cite.

- `answer_question`: a free question answered from past decisions, with cited BTI references.
- `vote_headings`: the retrieval-only baseline: rank headings by the similarity of their decisions.
- `suggest_heading`: the classification assistant: the LLM chooses one of the retrieved headings and
  cites the decisions that support it; the answer is parsed and checked.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .metrics import citation_precision, extract_citations
from .prompts import build_rag_messages, build_suggestion_messages
from .store import Hit


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[Hit]
    cited_ids: list[str]
    citation_precision: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def retrieved_ids(self) -> list[str]:
        return [h.doc_id for h in self.hits]


def _usage(resp) -> tuple[int | None, int | None]:
    usage = getattr(resp, "usage", None)
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


def answer_question(question: str, retriever, client, model: str, k: int = 5,
                    chapter: str | None = None, temperature: float = 0.0) -> RagAnswer:
    hits = retriever.search(question, k=k, chapter=chapter)  # 1 retrieve
    messages = build_rag_messages(question, hits)  # 2 assemble
    resp = client.chat.completions.create(model=model, messages=messages,  # 3 generate
                                          temperature=temperature)
    text = resp.choices[0].message.content or ""
    cited = extract_citations(text)  # 4 cite, and check the citations
    return RagAnswer(question, text, hits, cited, citation_precision(cited, [h.doc_id for h in hits]), *_usage(resp))


def vote_headings(hits: Sequence[Hit]) -> list[str]:
    """Headings ranked by the summed similarity of their retrieved decisions (a weighted k-NN vote).

    >>> vote_headings([Hit("a#0", "a", "", 0.9, "6403"), Hit("b#0", "b", "", 0.8, "6404"),
    ...                Hit("c#0", "c", "", 0.7, "6404")])
    ['6404', '6403']
    """
    totals: dict[str, float] = {}
    for h in hits:
        if h.heading:
            totals[h.heading] = totals.get(h.heading, 0.0) + max(h.score, 0.0)
    return sorted(totals, key=lambda x: -totals[x])


@dataclass
class HeadingSuggestion:
    description: str
    heading: str | None  # None if the answer could not be used
    candidates: list[str]
    hits: list[Hit]
    cited_ids: list[str] = field(default_factory=list)
    citation_precision: float = 0.0
    reason: str = ""
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def suggest_heading(description: str, retriever, client, model: str, heading_texts: Mapping[str, str],
                    k: int = 10, temperature: float = 0.0) -> HeadingSuggestion:
    """Retrieve k similar decisions, let the LLM choose one of their headings, check the answer."""
    hits = retriever.search(description, k=k)
    candidates = vote_headings(hits)
    messages = build_suggestion_messages(description, hits, heading_texts, candidates)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                          response_format={"type": "json_object"})
    out = HeadingSuggestion(description, None, candidates, hits)
    out.prompt_tokens, out.completion_tokens = _usage(resp)
    try:
        answer = json.loads(resp.choices[0].message.content or "")
        heading = str(answer["heading"]).strip()
        refs = [str(r) for r in answer.get("references", [])]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
        out.error = f"unusable answer: {type(e).__name__}"
        return out
    out.cited_ids = list(dict.fromkeys(refs))
    out.citation_precision = citation_precision(out.cited_ids, [h.doc_id for h in hits])
    out.reason = str(answer.get("reason", ""))
    if heading not in candidates:  # the model proposed a heading that no retrieved decision supports
        out.error = f"heading {heading!r} is not among the candidates"
        return out
    out.heading = heading
    return out
