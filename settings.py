import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

###############################################################################
# Central Configuration & Environment Manager
#
# Functional Details:
# - Loads and validates environment variables (.env file and Streamlit secrets).
# - Resolves project directory paths, virtual environment binaries, and CLI commands.
# - Manages API credentials (Tavily, AviationStack, OpenWeather, Groq, PostgreSQL).
# - Configures LLM model names for planner and destination extractor agents.
# - Generates dynamic server configurations for MultiServerMCPClient (Stdio and HTTP transports).
###############################################################################

# Define Project Base Directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables (.env file)
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

def get_config_val(key: str, default: str = "") -> str:
    """Retrieve configuration from environment variable or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

# ==========================================
# 🔑 API Keys & Credentials
# ==========================================
# Tavily API Key (supports case-insensitive key name variations)
TAVILY_API_KEY = (
    get_config_val("TAVILY_API_KEY") 
    or get_config_val("Tavily_API_Key") 
    or ""
)

# Aviationstack API Key
AVIATIONSTACK_API_KEY = (
    get_config_val("AVIATIONSTACK_API_KEY") 
    or get_config_val("AVIATION_STACK_API_KEY") 
    or ""
)

# OpenWeather API Key
OPENWEATHER_API_KEY = get_config_val("OPENWEATHER_API_KEY") or ""

# Groq Cloud API Key
GROQ_API_KEY = get_config_val("GROQ_API_KEY") or ""

# LangChain / LangSmith API Key & Tracing
LANGCHAIN_API_KEY = get_config_val("LANGCHAIN_API_KEY") or ""
LANGCHAIN_TRACING_V2 = get_config_val("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = get_config_val("LANGCHAIN_PROJECT", "Multi-Agent-Travel-MCP")

# PostgreSQL Database Connection URL for LangGraph Checkpointer
DATABASE_URL = get_config_val("DATABASE_URL", "")

# Cloudflare R2 Object Storage Configuration
R2_ACCOUNT_ID = get_config_val("R2_ACCOUNT_ID") or get_config_val("CLOUDFLARE_R2_ACCOUNT_ID") or ""
R2_ACCESS_KEY_ID = get_config_val("R2_ACCESS_KEY_ID") or get_config_val("CLOUDFLARE_R2_ACCESS_KEY_ID") or ""
R2_SECRET_ACCESS_KEY = get_config_val("R2_SECRET_ACCESS_KEY") or get_config_val("CLOUDFLARE_R2_SECRET_ACCESS_KEY") or ""
R2_BUCKET_NAME = get_config_val("R2_BUCKET_NAME") or get_config_val("CLOUDFLARE_R2_BUCKET_NAME") or "travel-plans"

# ==========================================
# 🤖 LLM Models Configuration
# ==========================================
GROQ_PLANNER_MODEL = get_config_val("GROQ_PLANNER_MODEL", "openai/gpt-oss-20b")
GROQ_DESTINATION_MODEL = get_config_val("GROQ_DESTINATION_MODEL", "openai/gpt-oss-20b")
GROQ_PROMPT_GUARD_MODEL = get_config_val("GROQ_PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")

# ==========================================
# 🛡️ Production Guardrail Configuration
# ==========================================
ENABLE_GUARDRAILS = get_config_val("ENABLE_GUARDRAILS", "true").lower() in ("true", "1", "yes")
PROMPT_GUARD_THRESHOLD = float(get_config_val("PROMPT_GUARD_THRESHOLD", "0.7"))

# ==========================================
# ⚡ Semantic Cache TTL Settings (Seconds)
# ==========================================
TTL_WEATHER_CURRENT = int(get_config_val("TTL_WEATHER_CURRENT", "3600"))       # 1 Hour
TTL_WEATHER_FORECAST = int(get_config_val("TTL_WEATHER_FORECAST", "7200"))     # 2 Hours
TTL_AVIATION_METADATA = int(get_config_val("TTL_AVIATION_METADATA", "86400")) # 24 Hours
TTL_HOTEL_SEARCH = int(get_config_val("TTL_HOTEL_SEARCH", "21600"))           # 6 Hours

# ==========================================
# 🔌 MCP (Model Context Protocol) Server Paths
# ==========================================
AVIATIONSTACK_DIR = PROJECT_ROOT / "aviationstack-mcp"
AVIATIONSTACK_SRC_DIR = AVIATIONSTACK_DIR / "src"

# Path to aviationstack executable in venv (if present locally)
AVIATIONSTACK_EXE_PATH = AVIATIONSTACK_DIR / ".venv" / "Scripts" / "aviationstack-mcp.exe"

WEATHER_MCP_SERVER_SCRIPT = PROJECT_ROOT / "custom_weather_mcp_server.py"
PYTHON_EXECUTABLE = sys.executable

# ==========================================
# 🛠️ MCP Client Configuration Dictionary
# ==========================================
def get_mcp_server_config():
    """Generates the multi-server MCP client configuration dict."""
    # Determine cross-platform command & args for aviationstack
    if AVIATIONSTACK_EXE_PATH.exists():
        aviation_cmd = str(AVIATIONSTACK_EXE_PATH)
        aviation_args = ["mcp", "run"]
    elif shutil.which("aviationstack-mcp"):
        aviation_cmd = shutil.which("aviationstack-mcp")
        aviation_args = ["mcp", "run"]
    else:
        aviation_cmd = PYTHON_EXECUTABLE
        aviation_args = [str(AVIATIONSTACK_SRC_DIR / "aviationstack_mcp" / "__main__.py")]

    return {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },
        "aviationstack": {
            "transport": "stdio",
            "command": aviation_cmd,
            "args": aviation_args,
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
        "Cloudflare R2": "[OK] Configured" if (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY) else "[NOT CONFIGURED]",
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
