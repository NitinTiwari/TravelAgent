import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

###############################################################################
# Multi-Server MCP Integration Test Suite
#
# Functional Details:
# - Tests multi-server MCP client connectivity and tool registration.
# - Verifies live tool calls across Tavily HTTP, AviationStack Stdio, and Weather Stdio servers.
# - Validates tool invocation responses for airports, airlines, flight search, and weather forecasts.
###############################################################################

#load_dotenv()
load_dotenv(override=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },

        "aviationstack": {
            "transport": "stdio",
            "command": r"C:\Users\hp\AI\Multi-agent-system-Travel_MCP\aviationstack-mcp\.venv\Scripts\python.exe",
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
            }
        },

        "weather": {
            "transport": "stdio",
            "command": r"C:\Users\hp\AI\Multi-agent-system-Travel_MCP\langraph_env3\Scripts\python.exe",
            "args": [
                r"C:\Users\hp\AI\Multi-agent-system-Travel_MCP\custom_weather_mcp_server.py"
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
        }
    }
)



# tools discovery
async def main():

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)

asyncio.run(main()) 
