import os
import sys
import logging
from datetime import datetime
from settings import PROJECT_ROOT

# Ensure stdout handles UTF-8 smoothly on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Default root application log file
APP_LOG_FILE = os.path.join(LOGS_DIR, "travel_app.log")

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

# Global base logger
logger = logging.getLogger("TravelSystem")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # General file handler
    fh = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def get_logger(name: str = "TravelSystem") -> logging.Logger:
    """Returns a named logger inheriting global settings."""
    return logging.getLogger(name)


def setup_session_logger(thread_id: str = "default") -> tuple[logging.Logger, str]:
    """
    Sets up a dedicated per-session log file for an individual travel planning run.
    Attaches the session file handler to the global TravelSystem logger so all MCP client
    caching logs, agent node logs, and LLM call logs are saved in the per-session log file.
    Returns: (session_logger, log_filepath)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_thread = "".join(c for c in str(thread_id) if c.isalnum() or c in ("-", "_")).strip() or "default"
    log_filename = f"session_{timestamp}_{sanitized_thread}.log"
    log_filepath = os.path.join(LOGS_DIR, log_filename)

    session_logger = logging.getLogger(f"Session_{sanitized_thread}_{timestamp}")
    session_logger.setLevel(logging.DEBUG)
    session_logger.handlers.clear()

    # Session file handler
    sfh = logging.FileHandler(log_filepath, encoding="utf-8")
    sfh.setLevel(logging.DEBUG)
    sfh.setFormatter(formatter)
    session_logger.addHandler(sfh)

    # Console handler for live tracking
    sch = logging.StreamHandler()
    sch.setLevel(logging.INFO)
    sch.setFormatter(formatter)
    session_logger.addHandler(sch)

    # Also attach the session file handler to TravelSystem so all MCP & Agent logs are captured in this file
    travel_system_logger = logging.getLogger("TravelSystem")
    for h in list(travel_system_logger.handlers):
        if getattr(h, "_is_session_handler", False):
            travel_system_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    sfh._is_session_handler = True
    travel_system_logger.addHandler(sfh)

    session_logger.info("=" * 70)
    session_logger.info(f"🚀 TRAVEL PLANNING SESSION STARTED | Thread ID: {thread_id}")
    session_logger.info(f"📁 Session Log File: {log_filepath}")
    session_logger.info("=" * 70)

    return session_logger, log_filepath

