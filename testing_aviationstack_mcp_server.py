import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

from settings import AVIATIONSTACK_API_KEY, AVIATIONSTACK_COMMAND, AVIATIONSTACK_SRC_DIR

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


import asyncio

async def main():

    tools = await client.get_tools()

    print("\nAvailable Tools:\n")

    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())