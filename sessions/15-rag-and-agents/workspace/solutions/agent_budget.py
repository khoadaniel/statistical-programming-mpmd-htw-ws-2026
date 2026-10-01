"""Exercise 3: a token budget guardrail (replace run_agent in src/review_assistant/agent.py)."""

from review_assistant.agent import AGENT_SYSTEM, AgentResult, Step, _assistant_turn
from review_assistant.tools import ToolBox


def run_agent(question: str, client, model: str, toolbox: ToolBox, max_steps: int = 6,
              system: str = AGENT_SYSTEM, max_total_tokens: int | None = None,
              temperature: float = 0.0) -> AgentResult:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    result = AgentResult(question=question, answer=None, stop_reason="max_steps")
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=messages, tools=toolbox.schemas(),
                                              temperature=temperature)
        result.llm_calls += 1
        usage = getattr(resp, "usage", None)
        result.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        result.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        if max_total_tokens is not None and result.total_tokens > max_total_tokens:
            result.stop_reason = "budget"
            return result
        msg = resp.choices[0].message
        if not msg.tool_calls:
            result.answer, result.stop_reason = msg.content, "answer"
            return result
        messages.append(_assistant_turn(msg))
        for tc in msg.tool_calls:
            observation, ok = toolbox.call(tc.function.name, tc.function.arguments)
            result.steps.append(Step(tc.function.name, tc.function.arguments, observation, ok))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})
    return result
