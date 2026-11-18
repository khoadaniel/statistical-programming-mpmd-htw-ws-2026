"""MCP client. It knows only the URL of the server, not its code. Run it: uv run client.py"""

import asyncio

from fastmcp import Client

MCP_URL = "http://127.0.0.1:8765/mcp"


async def main() -> None:
    # async with opens the connection and closes it at the end.
    async with Client(MCP_URL) as client:
        # 1. Discover: tools/list returns each tool's name, description and input schema.
        for tool in await client.list_tools():
            print(f"{tool.name}: {tool.description} | schema={tool.input_schema}")

        # 2. Call: tools/call runs the tool on the server; .data is the result as a Python value.
        print((await client.call_tool("add", {"a": 2, "b": 3})))
        print((await client.call_tool("minus", {"a": 10, "b": 4})))


asyncio.run(main())

