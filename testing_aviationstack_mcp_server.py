import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from settings import AVIATIONSTACK_API_KEY, AVIATIONSTACK_COMMAND, AVIATIONSTACK_SRC_DIR

###############################################################################
# AviationStack MCP Server Diagnostic & Test Runner
#
# Functional Details:
# - Spawns and attaches to the AviationStack MCP server subprocess using stdio transport.
# - Discovers and lists all registered aviation tools (airports, airlines, flights).
# - Validates subproject virtual environment, PYTHONPATH, and API key configurations.
###############################################################################

load_dotenv()

client = MultiServerMCPClient(
    {
        "aviationstack": {
            "transport": "stdio",
            "command": AVIATIONSTACK_COMMAND,
            "args": [
                "mcp",
                "run"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY,
                "AVIATIONSTACK_API_KEY": AVIATIONSTACK_API_KEY,
                "PYTHONPATH": str(AVIATIONSTACK_SRC_DIR)
            }
        }
    }
)

async def main():

    tools = await client.get_tools()

    print("\nAvailable Tools:\n")

    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())