"""MCP server with two calculator tools. Run it as its own process: uv run server.py"""

from fastmcp import FastMCP

# The name is what clients see when they connect.
mcp = FastMCP("Calculator")


# @mcp.tool turns a function into an MCP tool:
# name from the function name, description from the docstring, input schema from the type hints.
@mcp.tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool
def minus(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


if __name__ == "__main__":
    # Streamable HTTP on this computer only. The endpoint is http://127.0.0.1:8765/mcp
    mcp.run(transport="http", host="127.0.0.1", port=8765)
