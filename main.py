
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

#from mcp_client import tavily_mcp_search

from mcp_client import (
    tavily_mcp_search,
    get_airports,
    get_airlines,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search
)
from settings import DATABASE_URL, GROQ_PLANNER_MODEL, TAVILY_API_KEY

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



# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

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

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

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
    print("\nINSIDE HOTEL AGENT\n")
    query = f"Best hotels for {state['user_query']}"
    raw_hotel_data = ""

    try:
        raw_hotel_data = asyncio.run(tavily_mcp_search(query))
    except FileNotFoundError as fnf_err:
        import httpx
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
        else:
            raw_hotel_data = "Tavily API key not set."
    except Exception as exc:
        raw_hotel_data = f"Hotel search encountered an error: {exc}"

    # Extract clean search summary from raw tool output
    search_summary = extract_hotel_search_summary(raw_hotel_data)

    # Pass through LLM to generate structured, human-readable hotel advice
    try:
        prompt = HOTEL_AGENT_PROMPT.format(
            query=state["user_query"],
            hotel_search_data=search_summary
        )
        response = llm.invoke([
            SystemMessage(content="You are an expert travel accommodation advisor."),
            HumanMessage(content=prompt)
        ])
        hotel_data = response.content
    except Exception as e:
        hotel_data = search_summary if search_summary else f"Hotel information unavailable: {e}"

    print("✅ hotel_agent completed")
    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(content="Hotel recommendations generated")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def weather_agent(state: TravelState):
    print("\nINSIDE WEATHER AGENT\n")
    city = extract_destination(state["user_query"])

    try:
        weather_data = asyncio.run(weather_mcp_search(city))
    except Exception as e:
        weather_data = f"Current weather unavailable: {e}"

    try:
        forecast_data = asyncio.run(forecast_mcp_search(city))
    except Exception as e:
        forecast_data = f"Forecast unavailable: {e}"

    formatted_weather = f"""### 🌤️ Weather Report for {city.title()}
- **Current Weather Conditions**:
{weather_data}

- **5-Day Weather Forecast**:
{forecast_data}
"""

    return {
        "weather_results": formatted_weather,
        "messages": [
            AIMessage(content="Weather information fetched")
        ]
    }





# Itinerary Agent
def itinerary_agent(state: TravelState):

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

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }







graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)


graph.add_edge(START, "flight_agent")
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
                max_idle=30,
                timeout=30,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            pool.open()
            saver = PostgresSaver(pool)
            saver.setup()
            print("[OK] Connected to PostgreSQL ConnectionPool checkpointer (Neon / Remote)")
            return saver
        except Exception as e:
            print(f"[WARN] PostgreSQL connection pool failed ({e}). Falling back to MemorySaver.")
    
    print("[INFO] Using in-memory checkpoint saver.")
    return MemorySaver()

checkpointer = get_checkpointer()
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":

    # every run starts fresh.
    import uuid
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }


    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)