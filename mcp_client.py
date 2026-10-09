import os
import sys
import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from settings import (
    get_mcp_server_config,
    GROQ_DESTINATION_MODEL,
    TTL_WEATHER_CURRENT,
    TTL_WEATHER_FORECAST,
    TTL_AVIATION_METADATA,
    TTL_HOTEL_SEARCH,
)
from logger import logger
import time

###############################################################################
# Semantic Tool Output Caching (TTL-Based)
###############################################################################

_mcp_cache: dict[str, tuple[any, float]] = {}


def get_cached_mcp(cache_key: str):
    """Retrieves cached tool data if within valid TTL."""
    if cache_key in _mcp_cache:
        data, expiry = _mcp_cache[cache_key]
        now = time.time()
        if now < expiry:
            remaining_min = int((expiry - now) / 60)
            logger.info(f"⚡ [FETCHED_FROM_CACHE] Reusing cached MCP data for: '{cache_key}' (TTL remaining: {remaining_min}m)")
            return data
        else:
            logger.debug(f"⌛ [CACHE_EXPIRED] Cache expired for: '{cache_key}'")
            del _mcp_cache[cache_key]
    return None


def set_cached_mcp(cache_key: str, data: any, ttl_seconds: int = 3600):
    """Caches tool data with a specific Time-To-Live."""
    if data is not None:
        _mcp_cache[cache_key] = (data, time.time() + ttl_seconds)
        logger.debug(f"💾 [CACHE_SAVED] Saved '{cache_key}' to TTL cache (TTL: {ttl_seconds}s)")


def clear_mcp_cache():
    """Clears all cached MCP outputs."""
    _mcp_cache.clear()
    logger.info("🧹 [CACHE_CLEARED] MCP output cache cleared.")


def get_mcp_cache_stats() -> dict:
    """Returns active cache count and keys."""
    now = time.time()
    active_entries = {k: int((exp - now) / 60) for k, (_, exp) in _mcp_cache.items() if now < exp}
    return {
        "total_active": len(active_entries),
        "entries": active_entries
    }


client = MultiServerMCPClient(get_mcp_server_config())

search_tool = None
aviation_tools = {}

async def initialize_mcp():
    global search_tool
    global aviation_tools

    if search_tool is not None and aviation_tools:
        return

    tools = await client.get_tools()
    logger.debug(f"Discovered MCP Tools: {[t.name for t in tools]}")

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
    cache_key = f"tavily:{query.lower().strip()}"
    cached = get_cached_mcp(cache_key)
    if cached is not None:
        return cached

    await initialize_mcp()
    logger.info(f"🌐 [FETCHED_FROM_MCP_LIVE] Cache miss. Executing Tavily Search MCP for: '{query}'")
    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )
    set_cached_mcp(cache_key, result, ttl_seconds=TTL_HOTEL_SEARCH)
    return result


async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):
    args_repr = str(sorted((tool_args or {}).items()))
    cache_key = f"aviation:{tool_name}:{args_repr}"
    cached = get_cached_mcp(cache_key)
    if cached is not None:
        return cached

    tools = await client.get_tools()
    tool = next(
        t for t in tools
        if t.name == tool_name
    )

    logger.info(f"🌐 [FETCHED_FROM_MCP_LIVE] Cache miss. Executing Aviation MCP: '{tool_name}'")
    result = await tool.ainvoke(
        tool_args or {}
    )
    set_cached_mcp(cache_key, result, ttl_seconds=TTL_AVIATION_METADATA)
    return result


async def get_airports():
    cache_key = "aviation:list_airports"
    cached = get_cached_mcp(cache_key)
    if cached is not None:
        return cached

    await initialize_mcp()
    tool = aviation_tools.get("list_airports")
    if not tool:
        return "Airport tool unavailable"

    logger.info("🌐 [FETCHED_FROM_MCP_LIVE] Cache miss. Executing Aviation MCP: 'list_airports'")
    result = await tool.ainvoke({})
    set_cached_mcp(cache_key, result, ttl_seconds=TTL_AVIATION_METADATA)
    return result


async def get_airlines():
    cache_key = "aviation:list_airlines"
    cached = get_cached_mcp(cache_key)
    if cached is not None:
        return cached

    await initialize_mcp()
    tool = aviation_tools.get("list_airlines")
    if not tool:
        return "Airline tool unavailable"

    logger.info("🌐 [FETCHED_FROM_MCP_LIVE] Cache miss. Executing Aviation MCP: 'list_airlines'")
    result = await tool.ainvoke({})
    set_cached_mcp(cache_key, result, ttl_seconds=TTL_AVIATION_METADATA)
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
    cache_key = f"weather_current:{city.lower().strip()}"
    cached = get_cached_mcp(cache_key)
    if cached is not None:
        return cached

    await initialize_weather_tools()
    logger.info(f"🌐 [FETCHED_FROM_MCP_LIVE] Cache miss. Executing OpenWeather MCP: 'get_current_weather' for '{city}'")
    result = await weather_tool.ainvoke(
        {
            "city": city
        }
    )
    set_cached_mcp(cache_key, result, ttl_seconds=TTL_WEATHER_CURRENT)
    return result


async def forecast_mcp_search(city: str):
    cache_key = f"weather_forecast:{city.lower().strip()}"
    cached = get_cached_mcp(cache_key)
    if cached is not None:
        return cached

    await initialize_weather_tools()
    logger.info(f"🌐 [FETCHED_FROM_MCP_LIVE] Cache miss. Executing OpenWeather MCP: 'get_forecast' for '{city}'")
    result = await forecast_tool.ainvoke(
        {
            "city": city
        }
    )
    set_cached_mcp(cache_key, result, ttl_seconds=TTL_WEATHER_FORECAST)
    return result


# LLM
llm = ChatGroq(
    model=GROQ_DESTINATION_MODEL
)

###################################
# Destination Extractor (Hybrid: Fast Lexicon/Regex + LLM Fallback)
###################################

COMMON_DESTINATIONS = {
    # India
    "varanasi": "Varanasi", "kashi": "Varanasi", "delhi": "Delhi", "mumbai": "Mumbai", "goa": "Goa",
    "jaipur": "Jaipur", "agra": "Agra", "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "hyderabad": "Hyderabad", "chennai": "Chennai", "kolkata": "Kolkata", "manali": "Manali",
    "shimla": "Shimla", "kerala": "Kerala", "rishikesh": "Rishikesh", "amritsar": "Amritsar",
    "udaipur": "Udaipur", "ayodhya": "Ayodhya", "haridwar": "Haridwar", "ladakh": "Ladakh",
    "srinagar": "Srinagar", "darjeeling": "Darjeeling", "ooty": "Ooty", "coorg": "Coorg",
    "pune": "Pune", "ahmedabad": "Ahmedabad", "mysore": "Mysore", "pondicherry": "Puducherry",
    # Global
    "japan": "Japan", "tokyo": "Tokyo", "kyoto": "Kyoto", "osaka": "Osaka", "paris": "Paris",
    "france": "France", "london": "London", "uk": "London", "rome": "Rome", "italy": "Italy",
    "dubai": "Dubai", "uae": "Dubai", "bangkok": "Bangkok", "thailand": "Thailand", "singapore": "Singapore",
    "bali": "Bali", "indonesia": "Bali", "new york": "New York", "san francisco": "San Francisco",
    "barcelona": "Barcelona", "spain": "Spain", "amsterdam": "Amsterdam", "zurich": "Zurich",
    "switzerland": "Switzerland", "sydney": "Sydney", "australia": "Australia", "vietnam": "Vietnam",
    "maldives": "Maldives", "mauritius": "Mauritius", "egypt": "Egypt", "cairo": "Cairo"
}

def extract_destination(query: str) -> str:
    """
    Extracts destination city/country using a zero-token fast heuristic match.
    Prioritizes 'from Origin to Destination' syntax, followed by directional prepositions and lexicon matching.
    Falls back seamlessly to Groq LLM if the pattern is ambiguous or unknown.
    """
    import re

    # 1. Explicit 'from <Origin> to <Destination>' pattern (highest accuracy for multi-city queries)
    from_to_match = re.search(
        r'\bfrom\s+[A-Za-z\s]+?\s+to\s+([A-Za-z\s]+?)(?:,|\.|\s+under|\s+for|\s+including|\s+with|\s+on|\s+stay|\s+budget|\s*$)',
        query,
        re.IGNORECASE
    )
    if from_to_match:
        candidate = from_to_match.group(1).strip()
        clean = re.sub(r'^(?:a|an|the|complete|luxury|budget)\s+', '', candidate, flags=re.IGNORECASE).strip()
        if clean and len(clean.split()) <= 3:
            dest = COMMON_DESTINATIONS.get(clean.lower(), clean.title())
            logger.info(f"⚡ [FETCHED_FROM_LOCAL_RULES] Extracted destination '{dest}' from 'from-to' pattern (0 LLM tokens, 0ms)")
            return dest

    # 2. Directional patterns (e.g. 'trip to XYZ', 'visit XYZ', 'to XYZ', 'in XYZ')
    dir_match = re.search(
        r'(?:trip to|tour of|vacation in|travel to|going to|holiday in|visit to|visit|explore|to|in)\s+([A-Za-z\s]+?)(?:,|\.|\s+under|\s+for|\s+including|\s+with|\s+from|\s+on|\s+stay|\s+budget|\s*$)',
        query,
        re.IGNORECASE
    )
    if dir_match:
        candidate = dir_match.group(1).strip()
        clean = re.sub(r'^(?:a|an|the|complete|luxury|budget)\s+', '', candidate, flags=re.IGNORECASE).strip()
        if clean and len(clean.split()) <= 3:
            dest = COMMON_DESTINATIONS.get(clean.lower(), clean.title())
            logger.info(f"⚡ [FETCHED_FROM_LOCAL_RULES] Extracted destination '{dest}' via directional pattern (0 LLM tokens, 0ms)")
            return dest

    # 3. Check known destination lexicon (0 tokens, 0ms latency)
    q_lower = query.lower()
    for key in sorted(COMMON_DESTINATIONS.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(key) + r'\b', q_lower):
            dest = COMMON_DESTINATIONS[key]
            logger.info(f"⚡ [FETCHED_FROM_LOCAL_RULES] Extracted destination '{dest}' from lexicon (0 LLM tokens, 0ms)")
            return dest

    # 4. LLM Fallback only if heuristic fails
    logger.info("🧠 [FETCHED_FROM_LLM_CALL] Heuristic destination match missed. Invoking Groq LLM for entity extraction...")
    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """
    response = llm.invoke(prompt)
    dest = response.content.strip()
    logger.info(f"🧠 [FETCHED_FROM_LLM_CALL] Groq LLM returned destination: '{dest}'")
    return dest


async def _self_test():
    print("Testing MCP Client...")
    await initialize_mcp()
    print("MCP Client Initialized successfully!")


if __name__ == "__main__":
    asyncio.run(_self_test())