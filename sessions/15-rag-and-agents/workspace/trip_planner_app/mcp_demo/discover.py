"""Discovery: connect to the trip server and ask it what it offers. No API key, no model.

    uv run mcp_demo/discover.py

A client starts the server itself (stdio), so you do not start it by hand. The script then
1. lists the tools, resources and prompts (tool discovery),
2. lists the skills by name and description only, then loads one body (skill discovery),
3. calls one tool, to show that the server really works.
"""

import asyncio
import sys
from pathlib import Path

from fastmcp import Client

SERVER = Path(__file__).with_name("trip_server.py")


async def main() -> None:
    # Client(path) starts "python trip_server.py" as a child process and talks to it over stdio.
    async with Client(SERVER) as client:
        # 1. Tool discovery: the server answers tools/list, resources/list and prompts/list.
        print("== Tools ==")
        for tool in await client.list_tools():
            print(f"  {tool.name:18} {tool.description.splitlines()[0]}")
            print(f"  {'':18} inputs: {list(tool.input_schema['properties'])}")

        print("\n== Resources and resource templates ==")
        for resource in await client.list_resources():
            print(f"  {resource.uri}")
        for template in await client.list_resource_templates():
            print(f"  {template.uri_template}   (a template: fill in the name)")

        print("\n== Prompts ==")
        for prompt in await client.list_prompts():
            print(f"  {prompt.name}({', '.join(argument.name for argument in prompt.arguments)})")

        # 2. Skill discovery: first only the headers (cheap), then the body of the one that fits.
        print("\n== Skill discovery: headers only ==")
        skill_headers = (await client.call_tool("list_skills", {})).data
        for header in skill_headers:
            print(f"  {header['name']:14} {header['description']}")

        chosen = "budget-check"
        print(f"\n== Skill loaded on demand: {chosen} ==")
        print((await client.call_tool("load_skill", {"name": chosen})).data)
        same_text = (await client.read_resource(f"skill://{chosen}"))[0].text
        print(f"\n(The resource skill://{chosen} gives the same text: {len(same_text)} characters.)")

        # 3. One real call, without a model.
        print("\n== One tool call ==")
        call = await client.call_tool("budget_per_person", {
            "total_budget_eur": 450, "people": 3, "nights": 1, "return_fare_eur": 40, "bed_per_night_eur": 25})
        print("  ", call.data)


if __name__ == "__main__":
    # stderr is for logs; the script's own output goes to stdout.
    print(f"Python {sys.version.split()[0]}, server file: {SERVER.name}", file=sys.stderr)
    asyncio.run(main())
