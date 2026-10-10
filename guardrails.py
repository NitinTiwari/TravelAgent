import re
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from settings import (
    GROQ_API_KEY,
    GROQ_DESTINATION_MODEL,
    GROQ_PROMPT_GUARD_MODEL,
    PROMPT_GUARD_THRESHOLD,
    ENABLE_GUARDRAILS,
)
from logger import logger

###############################################################################
# Two-Tier Production Hybrid Guardrail System with Location & Actionability Gate
#
# Tier 1 (Fast-Path Negative Filter - 0ms):
# - PII Redaction & Masking
# - DoS / Query Length Filtering
# - Adversarial Injection Defense (Regex + Meta Llama Prompt Guard 86M)
# - Non-Actionable Task Detection (Blocks stories, poems, jokes, essays, Q&A)
# - Destination / City Requirement Gate (Enforces destination or from/to route)
#
# Tier 2 (Intent-Aware LLM Classification):
# - Evaluates ACTIONABLE_TRIP_PLAN vs. NON_ACTIONABLE intent via Groq LLM
#
# Output Sanitization:
# - Strips dangerous HTML/Script tags to prevent XSS in Streamlit UI
###############################################################################

# Regex patterns for PII redaction
PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", "[EMAIL_REDACTED]"),
    (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD_REDACTED]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    (r"\b\d{4}\s\d{4}\s\d{4}\b", "[AADHAAR_REDACTED]"),
]

# Fast regex signatures for prominent adversarial injection attempts
KNOWN_INJECTION_SIGNATURES = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|system)\s+prompts?",
    r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions|developer\s+mode)",
    r"you\s+are\s+now\s+(in\s+)?(dan|jailbreak|unfiltered)\s+mode",
    r"act\s+as\s+(an?\s+)?unrestricted\s+ai",
]

# Tier 1: Fast Regex Filter for Non-Actionable Creative Writing / Casual Chat Requests
NON_ACTIONABLE_PATTERNS = [
    r"\b(tell|write|narrate|create|generate)\s+(me\s+)?(a\s+|one\s+)?(short\s+|funny\s+|bedtime\s+)?(story|poem|song|essay|joke|tale|novel|script|code|riddle)\b",
    r"\b(what\s+is|explain|define|history\s+of)\s+(time\s+travel|travel\s+insurance\s+law|aviation|airplanes?)\b",
    r"\bhow\s+to\s+become\s+a\s+(travel\s+agent|flight\s+attendant|pilot|blogger)\b",
    r"\b(solve|calculate|write\s+code|debug|program)\b",
]


class GuardrailResult(TypedDict):
    is_valid: bool
    sanitized_query: str
    block_reason: str
    injection_score: float
    is_travel_topic: bool
    pii_redacted: list[str]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Redacts sensitive personal information (emails, phone numbers, payment cards, SSN/Aadhaar).
    Returns the redacted text and a list of redaction categories detected.
    """
    redacted_text = text
    detected = []

    for pattern, replacement in PII_PATTERNS:
        matches = re.findall(pattern, redacted_text)
        if matches:
            detected.append(replacement.strip("[]"))
            redacted_text = re.sub(pattern, replacement, redacted_text)

    return redacted_text, detected


def check_prompt_injection(text: str) -> tuple[bool, float, str]:
    """
    Evaluates adversarial jailbreak / prompt injection probability.
    Uses regex signatures first, then Meta Llama Prompt Guard on Groq.
    """
    # 1. Fast regex heuristic check
    for sig in KNOWN_INJECTION_SIGNATURES:
        if re.search(sig, text, re.IGNORECASE):
            logger.warning(f"🛡️ [GUARDRAIL_ALERT] Prompt injection pattern matched signature: '{sig}'")
            return True, 1.0, "Prompt injection attempt detected via security heuristics."

    # 2. Invoke Groq Llama Prompt Guard
    try:
        guard_client = ChatGroq(
            model=GROQ_PROMPT_GUARD_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,
        )
        resp = guard_client.invoke([HumanMessage(content=text)])
        score = float(resp.content.strip())
        logger.debug(f"🛡️ [GUARDRAIL] Llama Prompt Guard score: {score:.4f} (threshold: {PROMPT_GUARD_THRESHOLD})")

        if score >= PROMPT_GUARD_THRESHOLD:
            logger.warning(f"🛡️ [GUARDRAIL_BLOCKED] Prompt Guard detected injection with score {score:.4f}")
            return True, score, f"Adversarial or prompt injection attempt detected (Confidence: {score:.1%})."

        return False, score, ""
    except Exception as exc:
        logger.warning(f"🛡️ [GUARDRAIL_WARN] Prompt Guard API evaluation error: {exc}. Permitting query through fallback.")
        return False, 0.0, ""


def check_destination_present(text: str) -> tuple[bool, str]:
    """
    Verifies that the user query specifies a destination city/country or a 'from Origin to Destination' route.
    Queries without explicit location context (e.g. 'plan a trip for 3 days') are rejected to prevent downstream hallucinations.
    """
    # 1. Check explicit 'from X to Y' pattern
    from_to_match = re.search(r'\bfrom\s+[A-Za-z\s]+\s+to\s+[A-Za-z\s]+', text, re.IGNORECASE)
    if from_to_match:
        logger.debug("🛡️ [GUARDRAIL] Location check passed via 'from-to' route pattern.")
        return True, ""

    # 2. Check directional preposition patterns (e.g. 'trip to Paris', 'hotels in Goa', 'visit Tokyo')
    dir_match = re.search(
        r'\b(?:trip to|tour of|vacation in|travel to|going to|holiday in|visit to|visit|explore|hotels in|flight to|stay in|in)\s+([A-Za-z\s]+)',
        text,
        re.IGNORECASE
    )
    if dir_match:
        candidate = dir_match.group(1).strip()
        if candidate and len(candidate.split()) <= 4:
            logger.debug(f"🛡️ [GUARDRAIL] Location check passed via directional pattern: '{candidate}'")
            return True, ""

    # 3. Check known destination lexicon from mcp_client
    try:
        from mcp_client import COMMON_DESTINATIONS
        q_lower = text.lower()
        for key in COMMON_DESTINATIONS.keys():
            if re.search(r'\b' + re.escape(key) + r'\b', q_lower):
                logger.debug(f"🛡️ [GUARDRAIL] Location check passed via destination lexicon match: '{key}'")
                return True, ""
    except Exception:
        pass

    # 4. Fallback LLM verification for ambiguous location mentions
    try:
        classifier_client = ChatGroq(
            model=GROQ_DESTINATION_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,
        )
        prompt = f"""Determine if the following user query specifies a target destination city, country, or route (e.g., 'Paris', 'Goa', 'from Delhi to Mumbai').

User Query: "{text}"

Answer ONLY "HAS_DESTINATION" if a specific geographic city, country, or route is present, or "NO_DESTINATION" if no target location is specified."""
        resp = classifier_client.invoke([
            SystemMessage(content="You are a geographic entity classifier."),
            HumanMessage(content=prompt)
        ])
        if "HAS_DESTINATION" in resp.content.strip().upper():
            return True, ""
    except Exception:
        pass

    logger.warning("🛡️ [GUARDRAIL_BLOCKED] User query lacks a destination city/country or route.")
    return False, "Destination Missing: Please specify your target destination city or route (e.g., 'Flight from Delhi to Mumbai' or 'Plan a 4-day trip to Paris')."


def check_intent_and_topic(text: str) -> tuple[bool, str]:
    """
    Tier 1 (Fast-Path Negative Filter) + Tier 2 (Intent-Aware Classifier):
    1. Tier 1: Check fast regex patterns for creative writing / non-actionable prompts (0ms, 0 tokens)
    2. Check Destination & City Requirement
    3. Tier 2: Groq LLM Intent Classification (ACTIONABLE_TRIP_PLAN vs. NON_ACTIONABLE)
    """
    # Tier 1: Fast Regex Filter for Creative Writing / Casual Chat
    for pattern in NON_ACTIONABLE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"🛡️ [GUARDRAIL_BLOCKED] Tier 1 matched non-actionable creative writing pattern: '{pattern}'")
            return False, "This assistant is a travel concierge for planning trips, flights, hotels, and itineraries. Creative writing (stories, poems, jokes) and general chatter are not supported."

    # Location / Destination Requirement Gate
    has_dest, dest_reason = check_destination_present(text)
    if not has_dest:
        return False, dest_reason

    # Tier 2: Intent-Aware LLM Classifier
    logger.info("🛡️ [GUARDRAIL] Running Tier 2 Intent-Aware Classifier...")
    classifier_prompt = f"""You are a strict task-intent classifier for an AI Travel Booking & Itinerary Assistant.
Evaluate whether the user query represents an ACTIONABLE_TRIP_PLAN.

Definition:
- ACTIONABLE_TRIP_PLAN: The user is explicitly asking to plan a trip, search flights, recommend hotels, check weather, or request an itinerary for a destination.
- NON_ACTIONABLE: The user is asking for creative writing (stories, poems, jokes), general history, coding, theoretical Q&A, or general chatter.

User Query:
"{text}"

Answer ONLY "ACTIONABLE_TRIP_PLAN" or "NON_ACTIONABLE"."""

    try:
        classifier_client = ChatGroq(
            model=GROQ_DESTINATION_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,
        )
        resp = classifier_client.invoke([
            SystemMessage(content="You are a strict binary task-intent classifier."),
            HumanMessage(content=classifier_prompt)
        ])
        decision = resp.content.strip().upper()

        if "ACTIONABLE_TRIP_PLAN" not in decision:
            logger.warning("🛡️ [GUARDRAIL_BLOCKED] Tier 2 classified query as NON_ACTIONABLE.")
            return False, "This request is non-actionable. The Travel Assistant only processes concrete trip planning, flight search, hotel recommendation, and itinerary generation requests."

        return True, ""
    except Exception as err:
        logger.warning(f"🛡️ [GUARDRAIL_WARN] Intent classifier error: {err}. Permitting query through fallback.")
        return True, ""


def validate_input_guardrail(query: str) -> GuardrailResult:
    """
    Master Two-Tier Input Guardrail Pipeline:
    1. Length & Emptiness Sanity Filter
    2. PII Masking & Redaction
    3. Adversarial / Jailbreak Prompt Guard
    4. Two-Tier Hybrid Intent & Destination Gate
    """
    if not ENABLE_GUARDRAILS:
        return {
            "is_valid": True,
            "sanitized_query": query,
            "block_reason": "",
            "injection_score": 0.0,
            "is_travel_topic": True,
            "pii_redacted": [],
        }

    raw_query = (query or "").strip()

    # 1. Length & Empty check
    if not raw_query or len(raw_query) < 5:
        return {
            "is_valid": False,
            "sanitized_query": raw_query,
            "block_reason": "Query is too short. Please specify your trip details or destination (minimum 5 characters).",
            "injection_score": 0.0,
            "is_travel_topic": False,
            "pii_redacted": [],
        }

    if len(raw_query) > 2000:
        return {
            "is_valid": False,
            "sanitized_query": raw_query[:2000],
            "block_reason": "Query exceeds the maximum allowable length of 2,000 characters.",
            "injection_score": 0.0,
            "is_travel_topic": True,
            "pii_redacted": [],
        }

    # 2. PII Masking
    sanitized_text, pii_detected = mask_pii(raw_query)
    if pii_detected:
        logger.info(f"🛡️ [GUARDRAIL_PII] Redacted sensitive PII types: {pii_detected}")

    # 3. Prompt Injection Guard
    is_injection, score, injection_reason = check_prompt_injection(sanitized_text)
    if is_injection:
        return {
            "is_valid": False,
            "sanitized_query": sanitized_text,
            "block_reason": f"Security Guardrail: {injection_reason}",
            "injection_score": score,
            "is_travel_topic": False,
            "pii_redacted": pii_detected,
        }

    # 4. Two-Tier Hybrid Intent & Destination Gate
    is_valid_intent, intent_reason = check_intent_and_topic(sanitized_text)
    if not is_valid_intent:
        return {
            "is_valid": False,
            "sanitized_query": sanitized_text,
            "block_reason": intent_reason,
            "injection_score": score,
            "is_travel_topic": False,
            "pii_redacted": pii_detected,
        }

    logger.info("✅ [GUARDRAIL_PASSED] Input passed all security, intent, and destination validation checks.")
    return {
        "is_valid": True,
        "sanitized_query": sanitized_text,
        "block_reason": "",
        "injection_score": score,
        "is_travel_topic": True,
        "pii_redacted": pii_detected,
    }


def sanitize_output(text: str) -> str:
    """
    Sanitizes LLM outputs to prevent raw script execution or malicious markdown injections.
    """
    if not text:
        return ""

    clean = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"javascript\s*:", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"onerror\s*=", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"onload\s*=", "", clean, flags=re.IGNORECASE)

    return clean
