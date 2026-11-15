"""A city agent whose tools come from the MCP server, not from Python functions in the agent's file.

    uv run mcp_demo/agent_with_mcp.py "Can 3 of us do Prague this weekend with 450 EUR in total?"

Compare with city_agent in trip_graph.py: there the four tools are decorated Python functions in the same
file. Here the agent only knows the server. The tools, their descriptions and their input schemas arrive
over MCP, and so do the skills: the system prompt holds only the skill headers, and the agent calls
load_skill when a header fits the task (progressive disclosure).

Needs ANTHROPIC_API_KEY (a .env file in this folder or in ../../workbooks/) and an internet connection for the weather.
"""

import asyncio
import sys
import warnings
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from fastmcp import Client
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_anthropic import ChatAnthropic
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

load_dotenv(find_dotenv())
load_dotenv(Path(__file__).resolve().parents[3] / "workbooks" / ".env")

SERVER = Path(__file__).with_name("trip_server.py")
DEFAULT_QUESTION = "Can 3 of us do Prague this coming weekend with 450 EUR in total? Should we pack for rain?"


async def main(question: str) -> None:
    # 1. Skill discovery: ask the server for the skill headers and put only those into the system prompt.
    async with Client(SERVER) as client:
        headers = (await client.call_tool("list_skills", {})).data
    skill_lines = "\n".join(f"- {header['name']}: {header['description']}" for header in headers)
    system_prompt = (
        "You research one city for a short, cheap trip from Berlin. Use the tools for every fact; never guess a number.\n"
        "Skills are written instructions. When a skill fits the task, call load_skill with its name and follow it.\n"
        f"Skills available:\n{skill_lines}"
    )

    # 2. Tool discovery: the adapter starts the server, lists its tools and wraps each one as a LangChain tool.
    async with MCPAdapter(SERVER) as adapter:
        tools = await adapter.list_tools()
        print("Tools from the server:", [tool.name for tool in tools], file=sys.stderr)

        model = ChatAnthropic(model="claude-haiku-5-5", max_tokens=2048, timeout=60, max_retries=2)
        agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    # 3. Print every message: you can see the load_skill call before the tools that the skill asks for.
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_QUESTION))
