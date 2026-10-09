import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from settings import PYTHON_EXECUTABLE, WEATHER_MCP_SERVER_SCRIPT, OPENWEATHER_API_KEY

###############################################################################
# FastMCP Weather Server Diagnostic & Test Runner
#
# Functional Details:
# - Connects to the custom FastMCP weather server via stdio transport.
# - Loads and verifies the availability of weather tools (get_current_weather, get_forecast).
# - Validates that OpenWeather credentials and script paths are properly configured.
###############################################################################

#load_dotenv()
load_dotenv(override=True)

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