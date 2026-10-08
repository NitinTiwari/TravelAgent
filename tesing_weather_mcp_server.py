import os
from dotenv import load_dotenv
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
#load_dotenv()
load_dotenv(override=True)

from settings import PYTHON_EXECUTABLE, WEATHER_MCP_SERVER_SCRIPT, OPENWEATHER_API_KEY

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "stdio",
            "command": PYTHON_EXECUTABLE,
            "args": [
                str(WEATHER_MCP_SERVER_SCRIPT)
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
        }
    }
)

async def main():

    print("Loading tools...")

    tools = await client.get_tools()

    print("Tools loaded!")

    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())