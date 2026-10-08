import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Define Project Base Directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables (.env file)
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

# ==========================================
# 🔑 API Keys & Credentials
# ==========================================
# Tavily API Key (supports case-insensitive key name variations in .env)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("Tavily_API_Key") or ""

# Aviationstack API Key
AVIATIONSTACK_API_KEY = (
    os.getenv("AVIATIONSTACK_API_KEY") 
    or os.getenv("AVIATION_STACK_API_KEY") 
    or ""
)

# OpenWeather API Key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or ""

# Groq Cloud API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""

# LangChain / LangSmith API Key & Tracing
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") or ""
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "Multi-Agent-Travel-MCP")

# PostgreSQL Database Connection URL for LangGraph Checkpointer
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/langgraph_memory_demo"
)

# ==========================================
# 🤖 LLM Models Configuration
# ==========================================
GROQ_PLANNER_MODEL = os.getenv("GROQ_PLANNER_MODEL", "openai/gpt-oss-20b")
GROQ_DESTINATION_MODEL = os.getenv("GROQ_DESTINATION_MODEL", "openai/gpt-oss-20b")

# ==========================================
# 🔌 MCP (Model Context Protocol) Server Paths
# ==========================================
AVIATIONSTACK_DIR = PROJECT_ROOT / "aviationstack-mcp"
AVIATIONSTACK_SRC_DIR = AVIATIONSTACK_DIR / "src"

# Path to aviationstack executable in venv (falls back to generic sys.executable if not present)
AVIATIONSTACK_EXE_PATH = AVIATIONSTACK_DIR / ".venv" / "Scripts" / "aviationstack-mcp.exe"
AVIATIONSTACK_COMMAND = (
    str(AVIATIONSTACK_EXE_PATH)
    if AVIATIONSTACK_EXE_PATH.exists()
    else "aviationstack-mcp"
)

WEATHER_MCP_SERVER_SCRIPT = PROJECT_ROOT / "custom_weather_mcp_server.py"
PYTHON_EXECUTABLE = sys.executable

# ==========================================
# 🛠️ MCP Client Configuration Dictionary
# ==========================================
def get_mcp_server_config():
    """Generates the multi-server MCP client configuration dict."""
    return {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },
        "aviationstack": {
            "transport": "stdio",
            "command": AVIATIONSTACK_COMMAND,
            "args": ["mcp", "run"],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY,
                "AVIATIONSTACK_API_KEY": AVIATIONSTACK_API_KEY,
                "PYTHONPATH": str(AVIATIONSTACK_SRC_DIR),
            },
        },
        "weather": {
            "transport": "stdio",
            "command": PYTHON_EXECUTABLE,
            "args": [str(WEATHER_MCP_SERVER_SCRIPT)],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
            },
        },
    }

# ==========================================
# 🩺 Health Check & Validation
# ==========================================
def validate_settings():
    """Validates configuration and returns status report dictionary."""
    status = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "TAVILY_API_KEY": "[OK] Set" if TAVILY_API_KEY else "[MISSING]",
        "AVIATIONSTACK_API_KEY": "[OK] Set" if AVIATIONSTACK_API_KEY else "[MISSING]",
        "OPENWEATHER_API_KEY": "[OK] Set" if OPENWEATHER_API_KEY else "[MISSING]",
        "GROQ_API_KEY": "[OK] Set" if GROQ_API_KEY else "[MISSING]",
        "DATABASE_URL": "[OK] Set" if DATABASE_URL else "[MISSING]",
        "Aviationstack Exe Exists": "[OK] Yes" if AVIATIONSTACK_EXE_PATH.exists() else "[WARN] No (using fallback)",
        "Weather Script Exists": "[OK] Yes" if WEATHER_MCP_SERVER_SCRIPT.exists() else "[MISSING]",
    }
    return status

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print("\n--- System Settings Validation ---")
    for k, v in validate_settings().items():
        print(f"{k:26}: {v}")
