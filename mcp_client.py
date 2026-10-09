import os
import sys
import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from settings import get_mcp_server_config, GROQ_DESTINATION_MODEL

###############################################################################
# MCP Client & Multi-Tool Adapter Integration
#
# Functional Details:
# - Connects and manages communication across multiple Model Context Protocol (MCP) servers
#   (Tavily Search, AviationStack Flights & Airlines, OpenWeather Custom Server).
# - Discovers, initializes, and maps available MCP tools dynamically for agent use.
# - Provides async wrapper functions for:
#   * tavily_mcp_search: Performs live web intelligence and hotel/flight search.
#   * aviation_mcp_call / get_airports / get_airlines: Fetches aviation data.
#   * weather_mcp_search / forecast_mcp_search: Retrieves real-time weather & forecasts.
# - Implements LLM-based destination entity extraction (extract_destination) from user travel prompts.
###############################################################################

client = MultiServerMCPClient(get_mcp_server_config())

search_tool = None
aviation_tools = {}

async def initialize_mcp():

    global search_tool
    global aviation_tools

    if search_tool is not None and aviation_tools:
        return

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)

    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    aviation_tools = {
        tool.name: tool
        for tool in tools
        if tool.name != "tavily_search"
    }

async def tavily_mcp_search(query: str):
    await initialize_mcp()
    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )
    return result


# we will pass here one tool name(retrieved from aviation mcp server) and some args
async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):

    tools = await client.get_tools()

    tool = next(
        t for t in tools
        if t.name == tool_name
    )

    result = await tool.ainvoke(
        tool_args or {}
    )

    return result



async def get_airports():

    await initialize_mcp()

    tool = aviation_tools.get("list_airports")

    if not tool:
        return "Airport tool unavailable"

    result = await tool.ainvoke({})

    return result


async def get_airlines():

    await initialize_mcp()

    tool = aviation_tools.get("list_airlines")

    if not tool:
        return "Airline tool unavailable"

    result = await tool.ainvoke({})

    return result

weather_tool = None
forecast_tool = None

async def initialize_weather_tools():

    global weather_tool, forecast_tool

    if weather_tool is not None:
        return

    tools = await client.get_tools()

    weather_tool = next(
        t for t in tools
        if t.name == "get_current_weather"
    )

    forecast_tool = next(
        t for t in tools
        if t.name == "get_forecast"
    )


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    return await weather_tool.ainvoke(
        {
            "city": city
        }
    )


async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    return await forecast_tool.ainvoke(
        {
            "city": city
        }
    )


# LLM
llm = ChatGroq(
    model=GROQ_DESTINATION_MODEL
)

###################################
# Destination Extractor
###################################

def extract_destination(query: str):

    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


async def _self_test():
    print("Testing MCP Client...")
    await initialize_mcp()
    print("MCP Client Initialized successfully!")


if __name__ == "__main__":
    asyncio.run(_self_test())