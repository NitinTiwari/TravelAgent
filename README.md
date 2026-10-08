# ✈️ AI Multi-Agent Travel Booking & Planning System

An intelligent, production-ready multi-agent travel concierge powered by **LangGraph**, **Model Context Protocol (MCP)**, **Groq Cloud LLMs**, **PostgreSQL Long-Term Memory Checkpointing**, and a modern **Streamlit UI**.

---

## 🌟 Overview & Architecture

The application coordinates multiple specialized AI agents in a stateful directed acyclic graph (DAG) to generate complete, personalized travel itineraries. Each agent has a specific role and accesses data through specialized MCP (Model Context Protocol) servers.

```mermaid
graph LR
    START([Start / User Prompt]) --> FlightAgent[✈️ Flight Agent\n(Aviationstack MCP)]
    FlightAgent --> HotelAgent[🏨 Hotel Agent\n(Tavily Search MCP)]
    HotelAgent --> WeatherAgent[☀️ Weather Agent\n(OpenWeather FastMCP)]
    WeatherAgent --> ItineraryAgent[📋 Itinerary Planner\n(Groq LLM)]
    ItineraryAgent --> END([Complete Travel Itinerary])
```

### 🤖 Specialized Agents & Workflow

1. **Flight Agent (`flight_agent`)**:
   - Acts as the first step in the pipeline.
   - Queries real-time airline and airport metadata through the **Aviationstack MCP Server**.
   - Identifies likely departure/arrival airports, active routes, flight durations, airfare estimates, and provides peak season pricing advice.
2. **Hotel Agent (`hotel_agent`)**:
   - Takes the user query and searches for the best accommodations.
   - Performs web searches via the **Tavily MCP Server** (streamable HTTP transport) with automatic direct HTTP fallback if the MCP server is unreachable.
   - Finds top-rated accommodations, pricing tiers, and neighborhood recommendations.
3. **Weather Agent (`weather_agent`)**:
   - Uses a lightweight Groq LLM extraction step to pinpoint the exact destination city from the user's query.
   - Calls the custom **OpenWeather FastMCP Server** (`stdio` transport) to fetch current conditions and a 5-period weather forecast for the destination.
4. **Itinerary Agent (`itinerary_agent`)**:
   - The final synthesis step.
   - Synthesizes flight options, lodging, and live weather conditions into a structured, day-by-day travel itinerary complete with booking advice and travel tips.

---

## 🛠️ Technology Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph` with PostgreSQL checkpoints for memory)
- **Protocol**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) via `langchain-mcp-adapters`
- **LLM Provider**: [Groq Cloud](https://groq.com/) (using models like `openai/gpt-oss-20b` for planning and extraction)
- **MCP Servers**:
  - **Aviationstack MCP Server**: Stdio MCP server for global airport and flight data.
  - **Custom Weather MCP Server**: FastMCP server integrated with the OpenWeatherMap API.
  - **Tavily Search MCP**: Streamable HTTP MCP server for curated, real-time web research.
- **Persistence / Memory**: PostgreSQL via `PostgresSaver` (allows conversational memory and resuming paused graphs).
- **Frontend**: Streamlit with custom CSS glassmorphism UI & live step-by-step progress indicators.

---

## 📁 Project Structure

```text
Multi-agent-system-Travel_MCP/
├── .env                              # Environment variables & API keys (Not in VC)
├── .env.example                      # Template for environment variables
├── settings.py                       # Centralized global settings & MCP configuration
├── main.py                           # LangGraph multi-agent graph definition & checkpointer
├── mcp_client.py                     # MultiServerMCPClient manager & helper tools
├── frontend.py                       # Streamlit web user interface
├── custom_weather_mcp_server.py      # FastMCP OpenWeather stdio server
├── tesing_weather_mcp_server.py      # Standalone test script for Weather MCP
├── testing_aviationstack_mcp_server.py # Standalone test script for Aviationstack MCP
├── aviationstack-mcp/                # Aviationstack MCP source & package (Sub-module)
└── travel_plans/                     # Generated travel exports & plans (Markdown)
```

---

## ⚙️ Configuration & Environment Variables

All settings and credentials are centrally managed by `settings.py` and loaded from the `.env` file. This centralizes configuration and handles path resolution dynamically.

Create your `.env` file in the project root (you can copy `.env.example`):

```env
# Groq Cloud LLM API Key
GROQ_API_KEY=your_groq_api_key_here

# Tavily Search API Key
TAVILY_API_KEY=your_tavily_api_key_here

# Aviationstack Flight API Key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key_here

# OpenWeatherMap API Key
OPENWEATHER_API_KEY=your_openweather_api_key_here

# PostgreSQL Database Connection URL (for LangGraph persistence)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/langgraph_memory_demo

# Optional: LangSmith Tracing
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=false
```

### Validate Settings
You can test your environment configuration anytime by running the settings validator:
```powershell
python settings.py
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- PostgreSQL database instance running locally or in the cloud (update `DATABASE_URL` accordingly).

### 2. Activate Virtual Environment
```powershell
.\langraph_env3\Scripts\activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r aviationstack-mcp/requirements.txt
pip install langgraph langchain-groq langchain-mcp-adapters psycopg psycopg-binary streamlit requests python-dotenv httpx mcp
```

### 4. Run the Streamlit Web Application
To launch the interactive UI:
```powershell
python -m streamlit run frontend.py
```
Open your browser at `http://localhost:8501`. The UI provides a chat-like interface to enter travel queries and watch the agents work in real-time.

### 5. Run via Command-Line Interface (CLI)
You can also run the system directly from the terminal:
```powershell
python main.py
```
Enter your travel request when prompted, and the agents will print their progress and the final itinerary to the console.

---

## 💡 How Settings Resolution Works (`settings.py`)

`settings.py` standardizes all project paths and credentials to prevent errors:
- Automatically resolves `PROJECT_ROOT` without fragile relative paths.
- Handles case-insensitive naming differences (e.g., `Tavily_API_Key` vs `TAVILY_API_KEY`).
- Constructs the multi-server MCP dictionary (`get_mcp_server_config()`) dynamically so that tests, the frontend, and the CLI agents all use consistent endpoints and process arguments.