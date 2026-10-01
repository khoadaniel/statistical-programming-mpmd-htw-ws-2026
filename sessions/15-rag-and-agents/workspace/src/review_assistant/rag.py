"""The RAG pipeline: retrieve, assemble, generate, cite."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import citation_precision, extract_citations
from .prompts import build_rag_messages
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


def answer_question(question: str, retriever, client, model: str, k: int = 5,
                    parent_asin: str | None = None, temperature: float = 0.0) -> RagAnswer:
    hits = retriever.search(question, k=k, parent_asin=parent_asin)  # 1 retrieve
    messages = build_rag_messages(question, hits)  # 2 assemble
    resp = client.chat.completions.create(model=model, messages=messages,  # 3 generate
                                          temperature=temperature)
    text = resp.choices[0].message.content or ""
    cited = extract_citations(text)  # 4 cite, and check the citations
    usage = getattr(resp, "usage", None)
    return RagAnswer(
        question=question, answer=text, hits=hits, cited_ids=cited,
        citation_precision=citation_precision(cited, [h.doc_id for h in hits]),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )
