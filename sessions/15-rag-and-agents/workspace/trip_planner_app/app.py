"""A small web server that runs the trip planner graph and streams every step to the browser.

Run in this folder:
    uv run app.py
then open http://127.0.0.1:8000 (another port: uv run app.py --port 8765)
"""
import argparse
import json
from pathlib import Path

import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# The key: a .env in this folder, or else the .env of the notebooks one folder up.
load_dotenv(find_dotenv())

from langgraph.types import Command  # noqa: E402

from trip_graph import build_graph  # noqa: E402

HERE = Path(__file__).parent
graph = build_graph()
app = FastAPI(title="Trip planner: a team of agents")


class ChatMessage(BaseModel):
    thread_id: str
    message: str


def to_jsonable(value):
    """Pydantic objects and messages as plain JSON values."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def short(text: str, limit: int = 900) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def message_event(parent_node: str, parent_id: str, node: str, update: dict):
    """One step inside an agent (model reply or tool result), as an event for the browser."""
    for message in update.get("messages", []):
        if message.type == "ai":
            calls = [{"tool": call["name"], "args": call["args"]} for call in message.tool_calls]
            yield {"type": "inner", "parent": parent_node, "task_id": parent_id, "kind": "model",
                   "text": short(message.text), "tool_calls": calls}
        elif message.type == "tool":
            yield {"type": "inner", "parent": parent_node, "task_id": parent_id, "kind": "tool_result",
                   "tool": message.name, "content": short(str(message.content))}


def run_events(graph_input, config):
    """Translate the LangGraph stream into small JSON events, one per line."""
    stream = graph.stream(graph_input, config, stream_mode=["tasks", "updates"], subgraphs=True)
    for namespace, mode, data in stream:
        if namespace:
            # Inside an agent: the namespace names the node that runs it, as "node:task_id".
            parent_node, _, parent_id = namespace[0].partition(":")
            if mode == "tasks" and "input" in data and data["name"] == "model":
                yield {"type": "inner", "parent": parent_node, "task_id": parent_id, "kind": "model_call"}
            elif mode == "updates":
                for node, update in data.items():
                    if isinstance(update, dict):
                        yield from message_event(parent_node, parent_id, node, update)
            continue
        if mode == "tasks" and "input" in data:
            node_input = data["input"]
            event = {"type": "task_start", "node": data["name"], "task_id": data["id"]}
            if hasattr(node_input, "city"):
                # A Send payload: the input of one city agent.
                event["city"] = node_input.city
            event["input"] = json.loads(json.dumps(node_input, default=to_jsonable))
            yield event
        elif mode == "tasks":
            yield {"type": "task_end", "node": data["name"], "task_id": data["id"],
                   "error": data["error"] and str(data["error"]),
                   "result": json.loads(json.dumps(data["result"], default=to_jsonable)),
                   "paused": bool(data.get("interrupts"))}
        elif mode == "updates" and "__interrupt__" in data:
            yield {"type": "interrupt", **data["__interrupt__"][0].value}
        elif mode == "updates":
            for node, update in data.items():
                yield {"type": "update", "node": node,
                       "update": json.loads(json.dumps(update, default=to_jsonable))}


@app.post("/api/chat")
def chat(chat_message: ChatMessage):
    config = {"configurable": {"thread_id": chat_message.thread_id}}
    snapshot = graph.get_state(config)
    # A paused graph continues with the answer; otherwise the message starts a new run.
    if snapshot.interrupts:
        graph_input = Command(resume=chat_message.message)
    else:
        graph_input = {"request": chat_message.message}

    def lines():
        try:
            for event in run_events(graph_input, config):
                yield json.dumps(event, ensure_ascii=False) + "\n"
            final = graph.get_state(config)
            yield json.dumps({"type": "done", "paused": bool(final.interrupts),
                              "state": json.loads(json.dumps(final.values, default=to_jsonable))},
                             ensure_ascii=False) + "\n"
        except Exception as error:  # show the error in the page instead of a broken stream
            yield json.dumps({"type": "error", "message": f"{type(error).__name__}: {error}"}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/api/graph")
def graph_structure():
    """The graph as LangGraph sees it, as Mermaid text."""
    return {"mermaid": graph.get_graph().draw_mermaid()}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/recorded_run.json")
def recorded_run():
    """A recorded run, for the replay button: works without an API key."""
    return FileResponse(HERE / "recorded_run.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="127.0.0.1", port=parser.parse_args().port)
