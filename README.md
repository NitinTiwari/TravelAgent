# ✈️ AI Multi-Agent Travel Booking & Planning System

An intelligent, production-ready multi-agent travel concierge powered by **LangGraph**, **Model Context Protocol (MCP)**, **Groq Cloud LLMs**, **PostgreSQL Long-Term Memory Checkpointing**, and a modern **Streamlit UI**.

---

## 🌟 Overview & Architecture

The application coordinates multiple specialized AI agents in a stateful directed acyclic graph (DAG) powered by **LangGraph** to generate complete, personalized travel itineraries. Each agent has a specialized role and dynamically queries external tools through the **Model Context Protocol (MCP)** standard.

---

## 📊 Graphical Data Flow

The following graphical diagrams illustrate the complete end-to-end data flow, multi-agent pipeline, MCP tool integration, and state persistence lifecycle.

### 1. End-to-End System & Data Flow Architecture

```mermaid
flowchart TD
    %% Styling
    classDef client fill:#0e1a2b,stroke:#3a7bd5,stroke-width:2px,color:#e0edf8;
    classDef memory fill:#142236,stroke:#f59e0b,stroke-width:2px,color:#fef3c7;
    classDef agent fill:#0f2744,stroke:#38bdf8,stroke-width:2px,color:#ffffff;
    classDef mcp fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#e0e7ff;
    classDef llm fill:#2e1065,stroke:#c084fc,stroke-width:2px,color:#f3e8ff;
    classDef storage fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#ecfdf5;

    subgraph UI ["🖥️ User Interface & Entry Points"]
        User(["👤 User"]):::client
        StreamlitApp["🌐 Streamlit Web UI (frontend.py)\n• Live agent streaming\n• Thread ID session tracking\n• Export & download cards"]:::client
        CLIApp["💻 CLI Runner (main.py)\n• Terminal execution"]:::client
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

    subgraph Pipeline ["🤖 Sequential Multi-Agent Data Pipeline"]
        direction TB

        %% Flight Agent Node
        subgraph Step1 ["1️⃣ Flight Agent (flight_agent)"]
            direction TB
            FA["✈️ Flight Agent Node"]:::agent
            MCPClient1["🔌 MultiServerMCPClient"]:::mcp
            AvMCP["Aviationstack MCP Server\n(stdio transport)\n• list_airports\n• list_airlines"]:::mcp
            AvAPI["🌐 Aviationstack REST API"]:::mcp
            GroqFA["🧠 Groq LLM\n(Flight Reasoning)"]:::llm

            FA -->|1. Request flight tools| MCPClient1
            MCPClient1 -->|2. Execute stdio MCP tool| AvMCP
            AvMCP -->|3. Fetch live data| AvAPI
            AvAPI -.->|4. Airport/Airline metadata| FA
            FA -->|5. Format prompt + metadata| GroqFA
            GroqFA -.->|6. Return flight guidance| FA
        end

        %% Hotel Agent Node
        subgraph Step2 ["2️⃣ Hotel Agent (hotel_agent)"]
            direction TB
            HA["🏨 Hotel Agent Node"]:::agent
            MCPClient2["🔌 MultiServerMCPClient / HTTP"]:::mcp
            TavMCP["Tavily Search MCP\n(Streamable HTTP / Fallback)\n• tavily_search"]:::mcp
            GroqHA["🧠 Groq LLM\n(Accommodation Synthesis)"]:::llm

            HA -->|1. Search query: Best hotels| MCPClient2
            MCPClient2 -->|2. HTTP POST query| TavMCP
            TavMCP -.->|3. Hotel listings & URLs| HA
            HA -->|4. Structure hotel recommendations| GroqHA
            GroqHA -.->|5. Return categorized hotel guide| HA
        end

        %% Weather Agent Node
        subgraph Step3 ["3️⃣ Weather Agent (weather_agent)"]
            direction TB
            WA["🌤️ Weather Agent Node"]:::agent
            GroqWA["🧠 Groq LLM\n(Destination Extractor)"]:::llm
            FastMCP["OpenWeather FastMCP Server\n(stdio transport)\n• get_current_weather\n• get_forecast"]:::mcp
            OWAPI["🌐 OpenWeatherMap API"]:::mcp

            WA -->|1. Prompt: Extract destination| GroqWA
            GroqWA -.->|2. Extracted city name| WA
            WA -->|3. Call MCP weather & forecast| FastMCP
            FastMCP -->|4. HTTP REST query| OWAPI
            OWAPI -.->|5. Current temp & 5-day forecast| WA
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

    subgraph OutputDelivery ["📦 Outputs, Exports & Storage"]
        direction TB
        StreamlitRender["🖥️ Real-time UI Cards & Metrics\n(Agents Run, LLM Calls, Cost Status)"]:::client
        LocalMD["📁 Local Markdown File\n(travel_plans/travel_plan_<timestamp>.md)"]:::storage
        CloudR2["☁️ Cloudflare R2 Object Storage\n(upload_plan_to_r2)"]:::storage

        OutputDeliveryNode["Complete Itinerary Ready"] --> StreamlitRender
        StreamlitRender --> LocalMD
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
    participant DB as 🐘 PostgreSQL Checkpointer
    participant FA as ✈️ Flight Agent
    participant HA as 🏨 Hotel Agent
    participant WA as 🌤️ Weather Agent
    participant IA as 🗓️ Itinerary Agent
    participant MCP as 🔌 MCP Servers (Aviation / Tavily / Weather)
    participant LLM as 🧠 Groq Cloud LLM
    participant Storage as ☁️ Storage (Local / Cloudflare R2)

    User->>UI: Enter travel request & thread_id
    UI->>LG: app.stream(initial TravelState, config={"thread_id": id})
    LG->>DB: Save initial checkpoint

    %% Flight Agent
    rect rgb(14, 26, 43)
        Note over LG,FA: Step 1: Flight Discovery & Estimation
        LG->>FA: Invoke flight_agent(state)
        FA->>MCP: list_airports() & list_airlines() (Aviationstack MCP stdio)
        MCP-->>FA: Return airports & airlines JSON
        FA->>LLM: Prompt with flight metadata + user request
        LLM-->>FA: Flight options, estimated fares & route advice
        FA-->>LG: Return {flight_results, messages, llm_calls: +1}
        LG->>DB: Save checkpoint (flight_agent completed)
        LG-->>UI: Stream flight_results to status card
    end

    %% Hotel Agent
    rect rgb(20, 34, 54)
        Note over LG,HA: Step 2: Accommodation & Hotel Search
        LG->>HA: Invoke hotel_agent(state)
        HA->>MCP: tavily_search("Best hotels for <destination>")
        MCP-->>HA: Return web search hotel candidates
        HA->>LLM: Prompt with hotel search data + user budget
        LLM-->>HA: Curated hotel tiers (Budget/Mid/Luxury) & locations
        HA-->>LG: Return {hotel_results, messages, llm_calls: +1}
        LG->>DB: Save checkpoint (hotel_agent completed)
        LG-->>UI: Stream hotel_results to status card
    end

    %% Weather Agent
    rect rgb(15, 39, 68)
        Note over LG,WA: Step 3: Destination Extraction & Live Weather
        LG->>WA: Invoke weather_agent(state)
        WA->>LLM: extract_destination(user_query)
        LLM-->>WA: Return extracted destination city (e.g. "Varanasi")
        WA->>MCP: get_current_weather(city) & get_forecast(city)
        MCP-->>WA: Return current temperature, conditions & 5-day forecast
        WA-->>LG: Return {weather_results, messages}
        LG->>DB: Save checkpoint (weather_agent completed)
        LG-->>UI: Stream weather_results to status card
    end

    %% Itinerary Agent
    rect rgb(46, 16, 101)
        Note over LG,IA: Step 4: Master Itinerary Synthesis
        LG->>IA: Invoke itinerary_agent(state)
        IA->>LLM: Prompt with full aggregated state (flights + hotels + weather + query)
        LLM-->>IA: Synthesized day-by-day travel plan & booking tips
        IA-->>LG: Return {itinerary, messages, llm_calls: +1}
        LG->>DB: Save final graph checkpoint
        LG-->>UI: Stream final itinerary
    end

    %% Output & Storage
    rect rgb(6, 78, 59)
        Note over UI,Storage: Step 5: Rendering, Export & Cloud Persistence
        UI->>UI: Render metrics, itinerary markdown card & download button
        UI->>Storage: Save travel_plan_<timestamp>.md to travel_plans/
        opt Cloudflare R2 Enabled
            UI->>Storage: upload_plan_to_r2(filename, content)
        end
        UI-->>User: Display complete interactive itinerary
    end
```

---

### 3. State Schema (`TravelState`) Data Lifecycle

The graph shares a centralized `TravelState` TypedDict that accumulates agent outputs incrementally:

| State Key | Type | Description | Producer Node |
| :--- | :--- | :--- | :--- |
| `user_query` | `str` | Original user travel prompt and constraints | Entry / User Input |
| `messages` | `Annotated[list, operator.add]` | Chronological chat and status message history | All Agents |
| `flight_results` | `str` | Airport routes, flight durations, airfares, booking tips | `flight_agent` |
| `hotel_results` | `str` | Categorized hotel options (Budget/Mid/Luxury) & areas | `hotel_agent` |
| `weather_results` | `str` | Current weather conditions and 5-day forecast | `weather_agent` |
| `itinerary` | `str` | Complete day-by-day synthesized travel plan | `itinerary_agent` |
| `llm_calls` | `int` | Counter tracking total LLM inference calls | All Agent Nodes |

---

### 🤖 Specialized Agents & Workflow

1. **Flight Agent (`flight_agent`)**:
   - Acts as the first step in the pipeline.
   - Queries real-time airline and airport metadata through the **Aviationstack MCP Server** (`stdio` transport).
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