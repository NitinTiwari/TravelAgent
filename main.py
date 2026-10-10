# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

import os
from typing import TypedDict, Annotated
import operator
import asyncio
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

# from tools.tavily_tool import tavily_search

# from mcp_client import tavily_mcp_search

from mcp_client import (
    tavily_mcp_search,
    get_airports,
    get_airlines,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search
)
from logger import logger, setup_session_logger
from settings import DATABASE_URL, GROQ_PLANNER_MODEL, TAVILY_API_KEY
from guardrails import validate_input_guardrail, sanitize_output

###############################################################################
# LangGraph Multi-Agent Travel Planning Workflow & PostgreSQL Memory Checkpointer
#
# Functional Details:
# - Orchestrates a sequential multi-agent travel concierge pipeline:
#     0. input_guardrail: Enforces PII masking, Prompt Injection defense, and Domain safety.
#     1. flight_agent: Queries airport & flight data via AviationStack MCP and Tavily fallback.
#     2. hotel_agent: Gathers accommodation recommendations and pricing via Tavily MCP.
#     3. weather_agent: Retrieves current weather & multi-day forecasts via OpenWeather MCP.
#     4. itinerary_agent: Synthesizes all gathered data into a comprehensive travel plan.
# - Maintains persistent conversation state (TravelState) and message history across turns.
# - Integrates PostgreSQL ConnectionPool checkpointer (PostgresSaver) with automated fallback
#   to in-memory checkpointing (MemorySaver) for resilient state persistence.
# - Comprehensive event and session logging via logger.py.
###############################################################################

# LLM
llm = ChatGroq(
    model=GROQ_PLANNER_MODEL
)

# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str
    is_blocked: bool
    guardrail_status: str


# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def prune_aviation_data(raw_data, query: str, max_items: int = 5) -> str:
    """
    Lightweight deterministic filter that extracts only relevant airports/airlines
    matching query keywords, eliminating thousands of irrelevant context tokens.
    """
    if not raw_data:
        return "No flight metadata available."

    import json
    import re
    query_words = set(w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 3)

    items = []
    if isinstance(raw_data, list):
        items = raw_data
    elif isinstance(raw_data, dict) and "data" in raw_data and isinstance(raw_data["data"], list):
        items = raw_data["data"]
    elif isinstance(raw_data, dict):
        items = [raw_data]

    if items and isinstance(items[0], dict):
        matched = []
        for item in items:
            item_text = " ".join(str(v).lower() for v in item.values())
            if any(w in item_text for w in query_words):
                matched.append(item)

        if matched:
            return json.dumps(matched[:max_items], indent=2)
        # Fallback to compact sample
        return json.dumps(items[:max_items], indent=2)

    return str(raw_data)[:1200]


# Flight Agent
def flight_agent(state: TravelState):
    logger.info("✈️ [flight_agent] Node started")
    query = state["user_query"]
    logger.debug(f"[flight_agent] Processing query: {query}")

    try:
        logger.debug("[flight_agent] Invoking AviationStack MCP: list_airports & list_airlines")
        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        # Deterministically prune irrelevant global airports/airlines before LLM invocation
        pruned_airports = prune_aviation_data(airports, query)
        pruned_airlines = prune_aviation_data(airlines, query)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=pruned_airports,
            airline_data=pruned_airlines
        )

        logger.info(f"🧠 [FETCHED_FROM_LLM_CALL] Invoking Groq LLM ({GROQ_PLANNER_MODEL}) for flight estimation & route analysis")
        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content
        logger.info("✅ [flight_agent] Flight guidance generated successfully")

    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"
        logger.error(f"❌ [flight_agent] Error: {e}", exc_info=True)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# Hotel Agent Prompt
HOTEL_AGENT_PROMPT = """
You are an expert travel accommodation and hotel advisor.

User Travel Request:
{query}

Web Search Results & Hotel Data:
{hotel_search_data}

Please generate a well-structured, easy-to-read hotel guide for the user:
1. 🏨 Top Recommended Hotels (categorize by Budget, Mid-range, and Luxury/Heritage options if available)
2. 📍 Location & Proximity (distance to main attractions, temples, Ghats, station, or city center mentioned in user request)
3. 💰 Estimated Price Range & Budget Fit (keeping the user's budget and number of guests in mind)
4. ✨ Key Amenities & Highlights (family-friendly, AC, Wi-Fi, pure veg dining, cleanliness)
5. 💡 Booking & Stay Advice (recommended area to stay, peak rush warnings, booking tips)
6. 🔗 Reference Links (if available in the search results)

Return a clean, beautifully formatted Markdown guide.
"""

def extract_hotel_search_summary(raw_data) -> str:
    """Parses raw Tavily MCP or HTTP responses into clean text snippets."""
    if not raw_data:
        return "No hotel search results found."

    text_content = ""
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict) and "text" in item:
                text_content += item["text"] + "\n"
            elif hasattr(item, "text"):
                text_content += getattr(item, "text") + "\n"
            elif isinstance(item, str):
                text_content += item + "\n"
    elif isinstance(raw_data, dict):
        if "results" in raw_data and isinstance(raw_data["results"], list):
            formatted_items = []
            for r in raw_data["results"]:
                title = r.get("title", "Hotel Option")
                url = r.get("url", "")
                content = r.get("content", "")
                formatted_items.append(f"- **{title}** ({url})\n  {content}")
            return "\n\n".join(formatted_items)
        text_content = str(raw_data)
    else:
        text_content = str(raw_data)

    # If text_content contains serialized JSON from Tavily, unpack it
    try:
        import json
        parsed = json.loads(text_content.strip())
        if isinstance(parsed, dict) and "results" in parsed:
            formatted_items = []
            for r in parsed["results"]:
                title = r.get("title", "Hotel Option")
                url = r.get("url", "")
                content = r.get("content", "")
                formatted_items.append(f"- **{title}** ({url})\n  {content}")
            return "\n\n".join(formatted_items)
    except Exception:
        pass

    return text_content[:4000]

# Hotel Agent
def hotel_agent(state: TravelState):
    logger.info("🏨 [hotel_agent] Node started")
    query = f"Best hotels for {state['user_query']}"
    raw_hotel_data = ""
    logger.debug(f"[hotel_agent] Query: {query}")

    try:
        logger.debug("[hotel_agent] Querying Tavily MCP Search")
        raw_hotel_data = asyncio.run(tavily_mcp_search(query))
    except FileNotFoundError as fnf_err:
        import httpx
        logger.warning(f"[hotel_agent] MCP transport unavailable ({fnf_err}), using Tavily HTTP fallback")
        if TAVILY_API_KEY:
            try:
                response = httpx.post(
                    "https://api.tavily.com/search",
                    json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "advanced", "max_results": 6},
                    timeout=30,
                )
                raw_hotel_data = response.json()
            except Exception as e:
                raw_hotel_data = f"Hotel search fallback error: {e}"
                logger.error(f"[hotel_agent] HTTP fallback error: {e}")
        else:
            raw_hotel_data = "Tavily API key not set."
            logger.error("[hotel_agent] Tavily API key not configured")
    except Exception as exc:
        raw_hotel_data = f"Hotel search encountered an error: {exc}"
        logger.error(f"[hotel_agent] Search error: {exc}", exc_info=True)

    # Extract clean search summary from raw tool output
    search_summary = extract_hotel_search_summary(raw_hotel_data)

    # Pass through LLM to generate structured, human-readable hotel advice
    try:
        logger.info(f"🧠 [FETCHED_FROM_LLM_CALL] Invoking Groq LLM ({GROQ_PLANNER_MODEL}) for hotel categorization & tier structuring")
        prompt = HOTEL_AGENT_PROMPT.format(
            query=state["user_query"],
            hotel_search_data=search_summary
        )
        response = llm.invoke([
            SystemMessage(content="You are an expert travel accommodation advisor."),
            HumanMessage(content=prompt)
        ])
        hotel_data = response.content
        logger.info("✅ [hotel_agent] Hotel recommendations generated successfully")
    except Exception as e:
        hotel_data = search_summary if search_summary else f"Hotel information unavailable: {e}"
        logger.error(f"❌ [hotel_agent] LLM synthesis error: {e}", exc_info=True)

    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(content="Hotel recommendations generated")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def weather_agent(state: TravelState):
    logger.info("🌤️ [weather_agent] Node started")
    city = extract_destination(state["user_query"])
    logger.info(f"📍 [weather_agent] Working with destination: {city}")

    try:
        weather_data = asyncio.run(weather_mcp_search(city))
    except Exception as e:
        weather_data = f"Current weather unavailable: {e}"
        logger.error(f"[weather_agent] Current weather error: {e}", exc_info=True)

    try:
        forecast_data = asyncio.run(forecast_mcp_search(city))
    except Exception as e:
        forecast_data = f"Forecast unavailable: {e}"
        logger.error(f"[weather_agent] Forecast error: {e}", exc_info=True)

    formatted_weather = f"""### 🌤️ Weather Report for {city.title()}
- **Current Weather Conditions**:
{weather_data}

- **5-Day Weather Forecast**:
{forecast_data}
"""
    logger.info("✅ [weather_agent] Weather data ready")

    return {
        "weather_results": formatted_weather,
        "messages": [
            AIMessage(content="Weather information fetched")
        ]
    }


# Input Guardrail Node
def input_guardrail_node(state: TravelState):
    logger.info("🛡️ [input_guardrail] Node started - Validating input safety, PII, and topic alignment")
    query = state.get("user_query", "")
    guard_res = validate_input_guardrail(query)

    if not guard_res["is_valid"]:
        logger.warning(f"🚫 [input_guardrail] Blocked query: {guard_res['block_reason']}")
        block_msg = f"🛡️ **Travel Security & Domain Guardrail Notice**:\n\n{guard_res['block_reason']}"
        return {
            "is_blocked": True,
            "guardrail_status": f"Blocked: {guard_res['block_reason']}",
            "itinerary": block_msg,
            "messages": [AIMessage(content=block_msg)],
        }

    pii_note = f" (Redacted PII: {', '.join(guard_res['pii_redacted'])})" if guard_res["pii_redacted"] else ""
    status_msg = f"Passed security and domain verification{pii_note}"
    logger.info(f"✅ [input_guardrail] Input approved{pii_note}")

    return {
        "user_query": guard_res["sanitized_query"],
        "is_blocked": False,
        "guardrail_status": status_msg,
        "messages": [AIMessage(content=f"Guardrail Check: {status_msg}")],
    }


def guardrail_router(state: TravelState) -> str:
    """Conditional edge router: short-circuit to END if blocked, else proceed to flight_agent."""
    if state.get("is_blocked", False):
        logger.info("🛑 [guardrail_router] Routing directly to END (Query blocked by guardrails)")
        return END
    return "flight_agent"


# Itinerary Agent
def itinerary_agent(state: TravelState):
    logger.info("🗓️ [itinerary_agent] Node started - Final Master Plan Synthesis")

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}

    Weather Information:
    {state['weather_results']}
    """

    try:
        logger.info(f"🧠 [FETCHED_FROM_LLM_CALL] Invoking Groq LLM ({GROQ_PLANNER_MODEL}) for master itinerary synthesis")
        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel planner"
            ),
            HumanMessage(content=prompt)
        ])
        itinerary_data = sanitize_output(response.content)
        logger.info("✅ [itinerary_agent] Master itinerary generated and sanitized successfully")
    except Exception as e:
        itinerary_data = f"Itinerary generation encountered an error: {e}"
        logger.error(f"❌ [itinerary_agent] Error: {e}", exc_info=True)
        response = AIMessage(content=itinerary_data)

    return {
        "itinerary": itinerary_data,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


graph = StateGraph(TravelState)

# Nodes
graph.add_node("input_guardrail", input_guardrail_node)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)

# Edges with Conditional Guardrail Check
graph.add_edge(START, "input_guardrail")
graph.add_conditional_edges(
    "input_guardrail",
    guardrail_router,
    {
        "flight_agent": "flight_agent",
        END: END,
    }
)
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)


def get_checkpointer():
    """Initializes PostgresSaver with a robust ConnectionPool, falling back to MemorySaver if unavailable."""
    if DATABASE_URL:
        try:
            from psycopg_pool import ConnectionPool
            pool = ConnectionPool(
                DATABASE_URL,
                min_size=1,
                max_size=10,
                max_idle=60.0,
                max_lifetime=600.0,
                timeout=30.0,
                check=ConnectionPool.check_connection,
                reconnect_timeout=30.0,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                },
            )
            pool.open()
            saver = PostgresSaver(pool)
            saver.setup()
            logger.info("[OK] Connected to PostgreSQL ConnectionPool checkpointer (Neon / Remote)")
            return saver
        except Exception as e:
            logger.warning(f"[WARN] PostgreSQL connection pool failed ({e}). Falling back to MemorySaver.")
    
    logger.info("[INFO] Using in-memory checkpoint saver.")
    return MemorySaver()

checkpointer = get_checkpointer()
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import uuid
    thread_id = str(uuid.uuid4())
    session_log, log_file = setup_session_logger(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    user_input = input("Enter travel request: ")
    session_log.info(f"User Request: {user_input}")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0,
            "is_blocked": False,
            "guardrail_status": "",
        },
        config=config
    )

    session_log.info("🎉 Multi-agent pipeline execution completed successfully!")
    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)
    
    print(f"\n📁 Session log saved to: {log_file}")