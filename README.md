# ✈️ AI Multi-Agent Travel Booking & Planning System

An intelligent, production-ready multi-agent travel concierge powered by **LangGraph**, **Model Context Protocol (MCP)**, **Groq Cloud LLMs**, **PostgreSQL Long-Term Memory Checkpointing**, **TTL-Based Semantic Caching**, and a modern **Streamlit UI**.

---

## 🌟 Overview & Architecture

The application coordinates multiple specialized AI agents in a stateful directed acyclic graph (DAG) powered by **LangGraph** to generate complete, personalized travel itineraries. Each agent has a specialized role, dynamically queries external tools through the **Model Context Protocol (MCP)** standard, and uses a multi-tier token optimization architecture to maximize performance while minimizing LLM inference costs.

---

## 📊 Graphical Data Flow

The following graphical diagrams illustrate the complete end-to-end data flow, multi-agent pipeline, MCP tool integration, caching layer, and state persistence lifecycle.

### 1. End-to-End System & Data Flow Architecture

```mermaid
flowchart TD
    %% Styling
    classDef client fill:#0e1a2b,stroke:#3a7bd5,stroke-width:2px,color:#e0edf8;
    classDef memory fill:#142236,stroke:#f59e0b,stroke-width:2px,color:#fef3c7;
    classDef agent fill:#0f2744,stroke:#38bdf8,stroke-width:2px,color:#ffffff;
    classDef mcp fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#e0e7ff;
    classDef llm fill:#2e1065,stroke:#c084fc,stroke-width:2px,color:#f3e8ff;
    classDef cache fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5;

    subgraph UI ["🖥️ User Interface & Entry Points"]
        User(["👤 User"]):::client
        StreamlitApp["🌐 Streamlit Web UI (frontend.py)\n• Live agent streaming\n• Thread ID session tracking\n• Export Markdown & Session Log (.log)"]:::client
        CLIApp["💻 CLI Runner (main.py)\n• Terminal execution & logs"]:::client
        User -->|Submit travel prompt & budget| StreamlitApp
        User -->|Run travel query| CLIApp
    end

    subgraph Engine ["⚡ LangGraph State Engine & Persistence"]
        GraphRunner["🔄 LangGraph StateGraph (app.invoke / app.stream)\nState Schema: TravelState"]:::memory
        Checkpointer[("🐘 PostgreSQL Checkpointer\n(PostgresSaver ConnectionPool / Neon)\n↳ Fallback: In-Memory MemorySaver")]:::memory
        StreamlitApp -->|Invoke with thread_id| GraphRunner
        CLIApp -->|Invoke with thread_id| GraphRunner
        GraphRunner <-->|Read / Write checkpoint state| Checkpointer
    end

    subgraph CacheLayer ["⚡ Semantic TTL Caching & Zero-Token Fast Paths"]
        LocalCache[("💾 Thread-Safe In-Memory Cache\n• Weather: 1h TTL\n• Forecast: 2h TTL\n• Hotel Search: 6h TTL\n• Aviation Meta: 24h TTL")]:::cache
        ZeroTokenRules["⚡ Zero-Token Heuristic Matcher\n• 'From X to Y' Regex\n• Directional & Lexicon Matcher\n(Bypasses LLM for City Extraction)"]:::cache
    end

    subgraph Pipeline ["🤖 Sequential Multi-Agent Data Pipeline"]
        direction TB

        %% Flight Agent Node
        subgraph Step1 ["1️⃣ Flight Agent (flight_agent)"]
            direction TB
            FA["✈️ Flight Agent Node"]:::agent
            MCPClient1["🔌 MultiServerMCPClient"]:::mcp
            AvMCP["Aviationstack MCP Server\n(stdio transport)\n• list_airports\n• list_airlines"]:::mcp
            GroqFA["🧠 Groq LLM\n(Flight Reasoning)"]:::llm

            FA -->|1. Check TTL Cache| LocalCache
            FA -->|2. Cache Miss: Query MCP| MCPClient1
            MCPClient1 --> AvMCP
            FA -->|3. Prune metadata & Prompt| GroqFA
            GroqFA -.->|4. Return flight guidance| FA
        end

        %% Hotel Agent Node
        subgraph Step2 ["2️⃣ Hotel Agent (hotel_agent)"]
            direction TB
            HA["🏨 Hotel Agent Node"]:::agent
            MCPClient2["🔌 MultiServerMCPClient / HTTP"]:::mcp
            TavMCP["Tavily Search MCP\n(Streamable HTTP / Fallback)\n• tavily_search"]:::mcp
            GroqHA["🧠 Groq LLM\n(Accommodation Synthesis)"]:::llm

            HA -->|1. Check TTL Cache| LocalCache
            HA -->|2. Cache Miss: Search Web| MCPClient2
            MCPClient2 --> TavMCP
            HA -->|3. Structure hotel recommendations| GroqHA
            GroqHA -.->|4. Return categorized hotel guide| HA
        end

        %% Weather Agent Node
        subgraph Step3 ["3️⃣ Weather Agent (weather_agent)"]
            direction TB
            WA["🌤️ Weather Agent Node"]:::agent
            FastMCP["OpenWeather FastMCP Server\n(stdio transport)\n• get_current_weather\n• get_forecast"]:::mcp

            WA -->|1. Extract City| ZeroTokenRules
            ZeroTokenRules -.->|Hit: 0 LLM Tokens| WA
            WA -->|2. Check TTL Cache| LocalCache
            WA -->|3. Cache Miss: Call MCP| FastMCP
        end

        %% Itinerary Agent Node
        subgraph Step4 ["4️⃣ Itinerary Agent (itinerary_agent)"]
            direction TB
            IA["🗓️ Itinerary Planner Node"]:::agent
            GroqIA["🧠 Groq LLM\n(Synthesis & Master Itinerary)"]:::llm

            IA -->|Synthesize flight + hotel + weather + user query| GroqIA
            GroqIA -.->|Return complete day-by-day itinerary| IA
        end

        %% Flow between steps
        Step1 ==>|Update TravelState:\nflight_results| Step2
        Step2 ==>|Update TravelState:\nhotel_results| Step3
        Step3 ==>|Update TravelState:\nweather_results| Step4
    end

    GraphRunner --> Step1
    Step4 --> OutputDelivery

    subgraph OutputDelivery ["📦 Outputs, Exports & Logging"]
        direction TB
        StreamlitRender["🖥️ Real-time UI Cards & Metrics\n(Agents Run, LLM Calls, Cost Status)"]:::client
        LocalMD["📁 Local Markdown File\n(travel_plans/travel_plan_<timestamp>.md)"]:::cache
        SessionLog["📄 Per-Session Log File\n(logs/session_<timestamp>_<thread>.log)"]:::cache
        CloudR2["☁️ Cloudflare R2 Object Storage\n(Optional Backup)"]:::cache

        Step4 --> StreamlitRender
        StreamlitRender --> LocalMD
        StreamlitRender --> SessionLog
        StreamlitRender --> CloudR2
    end
```

---

### 2. Request & Execution Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 User
    participant UI as 🖥️ Streamlit UI / CLI
    participant LG as ⚡ LangGraph Engine
    participant Cache as 💾 TTL Cache & Rules
    participant DB as 🐘 PostgreSQL Checkpointer
    participant FA as ✈️ Flight Agent
    participant HA as 🏨 Hotel Agent
    participant WA as 🌤️ Weather Agent
    participant IA as 🗓️ Itinerary Agent
    participant MCP as 🔌 MCP Servers
    participant LLM as 🧠 Groq Cloud LLM
    participant Log as 📁 Logger (logs/)

    User->>UI: Enter travel request & thread_id
    UI->>LG: app.stream(initial TravelState, config={"thread_id": id})
    LG->>DB: Save initial checkpoint
    UI->>Log: Initialize session log (logs/session_<timestamp>.log)

    %% Flight Agent
    rect rgb(14, 26, 43)
        Note over LG,FA: Step 1: Flight Discovery & Estimation
        LG->>FA: Invoke flight_agent(state)
        FA->>Cache: Check aviation metadata cache (24h TTL)
        alt Cache Hit
            Cache-->>FA: ⚡ [FETCHED_FROM_CACHE] Return cached airport metadata
        else Cache Miss
            FA->>MCP: 🌐 [FETCHED_FROM_MCP_LIVE] list_airports() & list_airlines()
            MCP-->>FA: Return raw metadata
            FA->>Cache: Store in TTL cache
        end
        FA->>LLM: 🧠 [FETCHED_FROM_LLM_CALL] Flight reasoning with pruned metadata
        LLM-->>FA: Flight options & fare estimates
        FA-->>LG: Return {flight_results, messages, llm_calls: +1}
        LG->>DB: Save checkpoint
    end

    %% Hotel Agent
    rect rgb(20, 34, 54)
        Note over LG,HA: Step 2: Accommodation Search
        LG->>HA: Invoke hotel_agent(state)
        HA->>Cache: Check hotel search cache (6h TTL)
        alt Cache Hit
            Cache-->>HA: ⚡ [FETCHED_FROM_CACHE] Return cached search results
        else Cache Miss
            HA->>MCP: 🌐 [FETCHED_FROM_MCP_LIVE] tavily_search("Best hotels in <destination>")
            MCP-->>HA: Return web search hotel candidates
            HA->>Cache: Store in TTL cache
        end
        HA->>LLM: 🧠 [FETCHED_FROM_LLM_CALL] Structure hotel tiers & recommendations
        LLM-->>HA: Curated hotel tiers (Budget / Mid / Luxury)
        HA-->>LG: Return {hotel_results, messages, llm_calls: +1}
        LG->>DB: Save checkpoint
    end

    %% Weather Agent
    rect rgb(15, 39, 68)
        Note over LG,WA: Step 3: Destination Extraction & Weather
        LG->>WA: Invoke weather_agent(state)
        WA->>Cache: ⚡ [FETCHED_FROM_LOCAL_RULES] Zero-token regex & lexicon match
        opt Fallback if Heuristic Misses
            WA->>LLM: 🧠 [FETCHED_FROM_LLM_CALL] Groq extraction fallback
        end
        WA->>Cache: Check weather & forecast cache (1h / 2h TTL)
        alt Cache Hit
            Cache-->>WA: ⚡ [FETCHED_FROM_CACHE] Return cached weather
        else Cache Miss
            WA->>MCP: 🌐 [FETCHED_FROM_MCP_LIVE] get_current_weather() & get_forecast()
            MCP-->>WA: Return current temp & 5-day forecast
            WA->>Cache: Store in TTL cache
        end
        WA-->>LG: Return {weather_results, messages}
        LG->>DB: Save checkpoint
    end

    %% Itinerary Agent
    rect rgb(46, 16, 101)
        Note over LG,IA: Step 4: Master Itinerary Synthesis
        LG->>IA: Invoke itinerary_agent(state)
        IA->>LLM: 🧠 [FETCHED_FROM_LLM_CALL] Prompt with full aggregated state
        LLM-->>IA: Synthesized day-by-day travel plan
        IA-->>LG: Return {itinerary, messages, llm_calls: +1}
        LG->>DB: Save final graph checkpoint
    end

    %% Output & Storage
    rect rgb(6, 78, 59)
        Note over UI,Log: Step 5: Rendering & Export
        UI->>UI: Render metrics, tabs & download cards
        UI->>Log: Flush and close session log
        UI-->>User: Display complete interactive itinerary + log download
    end
```

---

## ⚡ Production Token & Latency Optimizations

To minimize inference costs and latency in high-throughput environments, the system employs three layers of optimization:

### 1. Zero-Token Destination Extraction Heuristics
Rather than spending an LLM inference call on every query to extract destination cities, `mcp_client.py` uses prioritized rule-based extraction:
- **Directional Regex**: Matches `"from <Origin> to <Destination>"` patterns (e.g., `"trip from Delhi to Chennai"` accurately resolves destination as `Chennai`).
- **Preposition Regex**: Matches `"to <City>"`, `"in <City>"`, `"visit <City>"`.
- **Lexicon Matcher**: Searches a comprehensive dictionary of major global and Indian travel hubs.
- **LLM Fallback**: Invoked only when heuristic checks fail, saving **1 LLM call per run** for >90% of user queries.

### 2. Aviation Metadata Pruning
Raw aviation API responses contain hundreds of unnecessary fields. `prune_aviation_data()` strips payloads down to essential IATA codes, airport names, and airline titles before passing them to LLM prompts, reducing prompt token bloat by up to **75%**.

### 3. Semantic TTL-Based Tool Output Caching
All MCP tool outputs are stored in a thread-safe, in-memory cache with configurable TTLs defined in [`settings.py`](file:///c:/Users/hp/AI/Multi-agent-system-Travel_MCP/settings.py):

| Cache Domain | Default TTL | Environment Variable | Purpose |
| :--- | :--- | :--- | :--- |
| **Current Weather** | `3600s` (1 Hour) | `TTL_WEATHER_CURRENT` | Prevents redundant weather API calls for the same city |
| **Weather Forecast** | `7200s` (2 Hours) | `TTL_WEATHER_FORECAST` | Reuses 5-day forecast across repeat queries |
| **Hotel Search** | `21600s` (6 Hours) | `TTL_HOTEL_SEARCH` | Caches Tavily accommodation search results |
| **Aviation Metadata** | `86400s` (24 Hours) | `TTL_AVIATION_METADATA` | Caches static airline and airport lists |

---

## 📝 Centralized Logging & Origin Tracking

Every event is recorded with clear origin tags in both the centralized log (`logs/travel_app.log`) and per-session log files (`logs/session_<timestamp>_<thread_id>.log`):

| Log Tag | Event Description |
| :--- | :--- |
| `⚡ [FETCHED_FROM_CACHE]` | Tool output retrieved from the TTL cache (displays remaining TTL in minutes). |
| `🌐 [FETCHED_FROM_MCP_LIVE]` | Cache miss; live MCP server execution over stdio or HTTP. |
| `⚡ [FETCHED_FROM_LOCAL_RULES]` | Destination city identified via zero-token heuristic regex / lexicon (0ms, 0 tokens). |
| `🧠 [FETCHED_FROM_LLM_CALL]` | Live Groq Cloud LLM inference invocation for reasoning or synthesis. |

---

## 🛠️ Technology Stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph` with PostgreSQL ConnectionPool checkpointer)
- **Protocol**: [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) via `langchain-mcp-adapters`
- **LLM Provider**: [Groq Cloud](https://groq.com/) (`openai/gpt-oss-20b` for planning, synthesis, and reasoning)
- **MCP Servers**:
  - **Aviationstack MCP Server**: Stdio MCP server for global airport and flight data.
  - **Custom Weather FastMCP Server**: FastMCP stdio server integrated with OpenWeatherMap API.
  - **Tavily Search MCP**: Streamable HTTP MCP server with direct HTTP fallback for accommodation research.
- **Persistence**: PostgreSQL via `PostgresSaver` with TCP keepalives and connection recycling.
- **Frontend**: Streamlit with custom CSS glassmorphism styling, real-time stream status, and one-click `.md` and `.log` exports.

---

## 📁 Project Structure

```text
Multi-agent-system-Travel_MCP/
├── .env                              # Environment variables & API keys (Not in VC)
├── .env.example                      # Template for environment variables
├── settings.py                       # Centralized global settings, TTLs & MCP configuration
├── logger.py                         # Centralized logging & per-session event tracking
├── main.py                           # LangGraph multi-agent graph definition & checkpointer
├── mcp_client.py                     # MultiServerMCPClient manager, TTL cache & helpers
├── frontend.py                       # Streamlit web user interface
├── custom_weather_mcp_server.py      # FastMCP OpenWeather stdio server
├── testing_weather_mcp_server.py     # Standalone test script for Weather MCP
├── testing_aviationstack_mcp_server.py # Standalone test script for Aviationstack MCP
├── aviationstack-mcp/                # Aviationstack MCP source & package (Sub-module)
├── travel_plans/                     # Generated travel exports & plans (Markdown)
└── logs/                             # Chronological event & per-session execution logs
```

---

## ⚙️ Configuration & Environment Variables

All settings and credentials are centrally managed by [`settings.py`](file:///c:/Users/hp/AI/Multi-agent-system-Travel_MCP/settings.py) and loaded from `.env`:

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

# Optional: TTL Caching Durations (Seconds)
TTL_WEATHER_CURRENT=3600
TTL_WEATHER_FORECAST=7200
TTL_AVIATION_METADATA=86400
TTL_HOTEL_SEARCH=21600

# Optional: Cloudflare R2 Cloud Storage Backup
R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=travel-plans
```

### Validate Settings
Test your configuration anytime by running the settings validator:
```powershell
python settings.py
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- PostgreSQL database instance running locally or in the cloud (Neon, Supabase, AWS RDS).

### 2. Activate Virtual Environment
```powershell
.\langraph_env3\Scripts\activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r aviationstack-mcp/requirements.txt
pip install langgraph langchain-groq langchain-mcp-adapters psycopg psycopg-binary psycopg-pool streamlit requests python-dotenv httpx mcp
```

### 4. Run the Streamlit Web Application
```powershell
python -m streamlit run frontend.py
```
Open your browser at `http://localhost:8501`.

### 5. Run via Command-Line Interface (CLI)
```powershell
python main.py
```