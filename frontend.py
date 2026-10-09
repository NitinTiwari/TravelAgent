import os
import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage
from main import app
from r2_storage import upload_plan_to_r2, is_r2_configured
from logger import setup_session_logger

###############################################################################
# Streamlit Interactive Web Application & Travel Concierge UI
#
# Functional Details:
# - Provides an interactive, modern user interface for the AI Travel Booking System.
# - Manages chat sessions, unique conversation thread IDs, and persistent state.
# - Streams and renders multi-agent execution results (Flights, Hotels, Weather, Itinerary).
# - Features real-time state inspection, node execution tracking, and markdown export/download.
# - Connects directly to the compiled LangGraph execution graph for end-to-end trip planning.
###############################################################################

st.set_page_config(
    page_title="AI-Travel Booking System-NItin",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #080d14;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}

/* ── Remove unnecessary top gaps ── */
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

section[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0.1rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 0.9rem !important;
    padding-right: 0.9rem !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.3rem !important;
}

/* ── Header & Sidebar Toggle Controls (<< Button) ── */
header[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebarHeader"] {
    padding-top: 0.1rem !important;
    padding-bottom: 0rem !important;
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
    min-height: unset !important;
    height: auto !important;
    margin-bottom: 0rem !important;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    display: flex !important;
    color: #7ab8f5 !important;
    z-index: 999999 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding: 0 !important;
}

[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    background: #0e1a2b !important;
    color: #7ab8f5 !important;
    border: 1px solid #1e2e44 !important;
    border-radius: 6px !important;
    padding: 0.15rem 0.4rem !important;
    height: auto !important;
    min-height: unset !important;
    margin: 0 !important;
}

/* ── Hero ── */
.hero-wrapper {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 0 !important;
    margin-bottom: 1.5rem;
    height: 270px;
    border: 1px solid rgba(58,123,213,0.35);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}
.hero-bg {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    filter: brightness(0.42) saturate(1.25);
    position: absolute;
    top: 0; left: 0;
}
.hero-content {
    position: relative;
    z-index: 2;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
    background: radial-gradient(circle, rgba(8,13,20,0.2) 0%, rgba(8,13,20,0.7) 100%);
}
.hero-badge {
    background: rgba(58,123,213,0.25);
    border: 1px solid rgba(58,123,213,0.5);
    color: #7ab8f5 !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 0.9rem;
    display: inline-block;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.6rem;
    line-height: 1.2;
}
.hero-sub {
    color: #94adc8;
    font-size: 1rem;
    max-width: 560px;
}

/* ── Input card ── */
.input-card {
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
}
.input-label {
    color: #7ab8f5;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Quick destinations ── */
.dest-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.2rem;
}
.dest-chip {
    background: #111b2b;
    border: 1px solid #1e3050;
    color: #f7fdf4;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
}
.dest-chip:hover { background: #1a2e44; border-color: #3a7bd5; color: #fff; }

/* ── Generate button ── */
div[data-testid="stButton"] {
    display: flex;
    justify-content: flex-start;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a6bbf 0%, #0d4a8a 50%, #0a3d75 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.6rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    box-shadow: 0 0 24px rgba(26,107,191,0.35), 0 4px 15px rgba(0,0,0,0.4) !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 0 40px rgba(26,107,191,0.6), 0 6px 20px rgba(0,0,0,0.5) !important;
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #2278d4 0%, #1057a0 50%, #0d4a8a 100%) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Agent status cards ── */
[data-testid="stStatusWidget"] {
    background: #0e1a2e !important;
    border: 1px solid #1e3050 !important;
    border-radius: 12px !important;
}
[data-testid="stStatusWidget"] > div:first-child {
    background: #0e1a2e !important;
    border-radius: 12px 12px 0 0 !important;
}
[data-testid="stStatusWidget"] details,
[data-testid="stStatusWidget"] details > div,
[data-testid="stStatusWidget"] [data-testid="stVerticalBlock"] {
    background: #0a1520 !important;
    color: #ffffff !important;
    padding: 0.25rem 0.5rem !important;
}
[data-testid="stStatusWidget"] * { color: #ffffff !important; }
[data-testid="stStatusWidget"] a { color: #4ea8f0 !important; }
[data-testid="stStatusWidget"] hr { border-color: #1e3050 !important; }

/* ── Section headers ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2e44;
}
.sec-head span { font-size: 1.15rem; font-weight: 600; color: #e0edf8; }

/* ── Metric bar ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-box {
    flex: 1;
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #4ea8f0; }
.metric-lbl { font-size: 0.78rem; color: #5a7a96; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Final plan ── */
.final-card {
    background: linear-gradient(160deg, #0c1a2e 0%, #0a1520 100%);
    border: 1px solid #1e3a5c;
    border-left: 4px solid #3a7bd5;
    border-radius: 14px;
    padding: 1.8rem;
    line-height: 1.8;
    color: #cce0f5;
    font-size: 0.95rem;
}

/* ── Save bar ── */
.save-bar {
    background: #0e1623;
    border: 1px solid #1e2e44;
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    color: #5a8ab0;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #090e18 !important;
    border-right: 1px solid #141f30 !important;
}
.sidebar-chip {
    background: #0e1a2b;
    border: 1px solid #1a2e44;
    border-radius: 8px;
    padding: 0.45rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.83rem;
    color: #7aa8cc;
}
.sidebar-header {
    color: #e0edf8;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0 !important;
    padding: 0 !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.sidebar-title { color: #e0edf8; font-size: 1.05rem; font-weight: 600; margin: 0.8rem 0 0.4rem !important; }

/* Hide branding */
#MainMenu, footer { visibility: hidden !important; }

/* Textarea */
.stTextArea textarea {
    background: #0a1520 !important;
    border: 1px solid #1e2e44 !important;
    border-radius: 10px !important;
    color: #e8f4ff !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #3a7bd5 !important;
    box-shadow: 0 0 0 2px rgba(58,123,213,0.2) !important;
}
.stTextArea textarea::placeholder { color: #4a6a85 !important; }

/* Text input (sidebar User ID field) */
input[type="text"], .stTextInput input {
    background: #0e1a2b !important;
    border: 1px solid #1a2e44 !important;
    border-radius: 8px !important;
    color: #e0edf8 !important;
}
input[type="text"]:focus, .stTextInput input:focus {
    border-color: #3a7bd5 !important;
    box-shadow: 0 0 0 2px rgba(58,123,213,0.2) !important;
}
input[type="text"]::placeholder { color: #3a5570 !important; }

/* All Streamlit labels — dark bg → light text */
.stTextInput label, .stTextArea label,
.stSelectbox label, .stNumberInput label {
    color: #7ab8f5 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
}

/* General markdown / paragraph text */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    color: #cce0f5 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e8f4ff !important; }
.stMarkdown code {
    background: #0e1a2b !important;
    color: #7ab8f5 !important;
    padding: 0.15em 0.4em;
    border-radius: 4px;
}

/* Metric labels — was #5a7a96 (too dim on dark bg) */
.metric-lbl { color: #7aa8cc !important; }

/* Save bar — was #5a8ab0 (slightly dim) */
.save-bar { color: #8ab8d8 !important; }
.save-bar code { color: #7ab8f5 !important; background: #0a1520 !important; }

/* Streamlit warning / info / success on dark bg */
.stAlert { background: #0e1a2b !important; border-radius: 10px !important; }
.stAlert p, .stAlert div { color: #e0edf8 !important; }

/* Sidebar text & dividers */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown { color: #a0c4e0 !important; }
section[data-testid="stSidebar"] hr {
    border-color: #1a2e44 !important;
    margin: 0.35rem 0 0.65rem 0 !important;
}

/* Download button — light bg → dark text  */
div[data-testid="stDownloadButton"] > button {
    background: #1a3a5c !important;
    color: #e8f4ff !important;
    border: 1px solid #2a5080 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-header'>🌍 AI Travel Planner</div>", unsafe_allow_html=True)
    st.markdown("---")

    thread_id = st.text_input("👤 User ID", value="NitinTiwati_user",
                              help="Your session ID — keeps travel history across queries")

    st.markdown("<div class='sidebar-title'>Powered by</div>", unsafe_allow_html=True)
    for tech in ["🔗 LangGraph", "🧠 Groq · LLaMA 3.3 70B", "🐘 PostgreSQL", "☁️ Cloudflare R2", "🔍 Tavily Search", "✈️ AviationStack"]:
        st.markdown(f"<div class='sidebar-chip'>{tech}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>Agent Pipeline</div>", unsafe_allow_html=True)
    for step in ["① ✈️ Flight Agent", "② 🏨 Hotel Agent", "③ 🌤️ Weather Agent", "④ 🗓️ Itinerary Agent"]:
        st.markdown(f"<div class='sidebar-chip'>{step}</div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <img class="hero-bg"
         src="https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1600&q=85"
         alt="Multi-Agent AI Global Network"/>
    <div class="hero-content">
        <div class="hero-badge">✦ Multi-Agent AI System</div>
        <div class="hero-title">✈️ AI Travel Booking System</div>
        <div class="hero-sub">Four specialized agents work together — searching flights, hotels, building an itinerary, and delivering your perfect trip plan.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Destination image strip ───────────────────────────────────────────────────
# DESTINATIONS = [
#     ("🇯🇵 Tokyo",     "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=300&q=70"),
#     ("🇫🇷 Paris",     "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=300&q=70"),
#     ("🇹🇭 Bangkok",   "https://images.unsplash.com/photo-1508009603885-50cf7c579365?w=300&q=70"),
#     ("🇮🇹 Rome",      "https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=300&q=70"),
#     ("🇦🇪 Dubai",     "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=300&q=70"),
# ]

# cols = st.columns(5)
# for col, (name, img_url) in zip(cols, DESTINATIONS):
#     with col:
#         st.markdown(f"""
#         <div style="border-radius:10px;overflow:hidden;position:relative;height:90px;cursor:pointer;">
#             <img src="{img_url}" style="width:100%;height:100%;object-fit:cover;filter:brightness(0.55);" />
#             <div style="position:absolute;bottom:8px;left:0;right:0;text-align:center;
#                         color:#fff;font-size:0.8rem;font-weight:600;">{name}</div>
#         </div>
#         """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='input-label'>🗺️ Describe your trip</div>", unsafe_allow_html=True)

# QUICK = ["7-day Japan under ₹2L", "Paris trip for 5 days", "Dubai weekend trip", "Bali backpacking 10 days"]
# qcols = st.columns(len(QUICK))
# quick_fill = ""
# for qc, label in zip(qcols, QUICK):
#     with qc:
#         if st.button(label, key=f"q_{label}"):
#             quick_fill = label

user_query = st.text_area(
    "Describe your trip",
    # value=quick_fill,
    placeholder="e.g. Plan a complete 4-day Kashi Vishwanath, Varanasi trip including flights, hotels and sightseeing under ₹1.5 lakhs",
    height=100,
    label_visibility="collapsed",
)

btn_col, _ = st.columns([1, 2])
with btn_col:
    generate = st.button("🚀  Generate My Travel Plan", use_container_width=True)

# ── Agent pipeline ────────────────────────────────────────────────────────────
AGENT_META = {
    "flight_agent":    ("✈️", "Flight Agent"),
    "hotel_agent":     ("🏨", "Hotel Agent"),
    "weather_agent":   ("🌤️", "Weather Agent"),
    "itinerary_agent": ("🗓️", "Itinerary & Final Planner"),
}

if generate:
    if not user_query.strip():
        st.warning("Please describe your trip first.")
    else:
        # Initialize session-specific logger file
        session_logger, session_log_path = setup_session_logger(thread_id)
        session_logger.info(f"User Query: {user_query}")
        session_logger.info(f"Thread ID: {thread_id}")

        config = {"configurable": {"thread_id": thread_id}}
        collected = {
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        }
        agents_run = 0

        st.markdown("---")
        st.markdown("<div class='sec-head'><span>🤖 Agent Pipeline — Live</span></div>",
                    unsafe_allow_html=True)

        try:
            for chunk in app.stream(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "user_query": user_query,
                    "flight_results": "",
                    "hotel_results": "",
                    "weather_results": "",
                    "itinerary": "",
                    "llm_calls": 0,
                },
                config=config,
                stream_mode="updates",
            ):
                for node_name, state_update in chunk.items():
                    agents_run += 1
                    icon, label = AGENT_META.get(node_name, ("🔧", node_name))
                    session_logger.info(f"Node execution completed: {node_name}")

                    with st.status(f"{icon}  {label}", state="complete", expanded=True):
                        if node_name == "flight_agent":
                            text = state_update.get("flight_results", "")
                            collected["flight_results"] = text
                            st.markdown(text or "_No flight data returned._")

                        elif node_name == "hotel_agent":
                            text = state_update.get("hotel_results", "")
                            collected["hotel_results"] = text
                            st.markdown(text or "_No hotel data returned._")

                        elif node_name == "weather_agent":
                            text = state_update.get("weather_results", "")
                            collected["weather_results"] = text
                            st.markdown(text or "_No weather data returned._")

                        elif node_name == "itinerary_agent":
                            text = state_update.get("itinerary", "")
                            collected["itinerary"] = text
                            collected["final_response"] = text
                            st.markdown(text or "_No itinerary generated._")

                        collected["llm_calls"] = state_update.get("llm_calls", collected["llm_calls"])
            session_logger.info("Graph streaming completed successfully.")
        except Exception as stream_err:
            session_logger.error(f"Error during graph execution: {stream_err}", exc_info=True)
            st.error(f"⚠️ An error occurred during plan generation: {stream_err}")

        # Metrics
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">{agents_run}</div><div class="metric-lbl">Agents Run</div></div>
            <div class="metric-box"><div class="metric-val">{collected['llm_calls']}</div><div class="metric-lbl">LLM Calls</div></div>
            <div class="metric-box"><div class="metric-val">✅</div><div class="metric-lbl">Status</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Final plan card
        final_text = collected["final_response"] or collected["itinerary"]
        if final_text:
            st.markdown("<div class='sec-head'><span>🧠 Final Travel Plan</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='final-card'>{final_text}</div>",
                        unsafe_allow_html=True)

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"travel_plan_{timestamp}.md"
        save_dir = os.path.join(os.path.dirname(__file__), "travel_plans")
        os.makedirs(save_dir, exist_ok=True)

        file_content = f"""# Travel Plan
**Query:** {user_query}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**User ID:** {thread_id}

---

## ✈️ Flight Information
{collected['flight_results'] or 'N/A'}

---

## 🏨 Hotel Information
{collected['hotel_results'] or 'N/A'}

---

## 🌤️ Weather Forecast & Information
{collected['weather_results'] or 'N/A'}

---

## 🗓️ Final Itinerary & Travel Plan
{collected['itinerary'] or 'N/A'}

---
*LLM Calls: {collected['llm_calls']}*
"""
        # Save locally
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
            f.write(file_content)
        session_logger.info(f"Saved travel plan to {os.path.join(save_dir, filename)}")

        # Upload to Cloudflare R2 Object Storage
        r2_status = None
        if is_r2_configured():
            r2_status = upload_plan_to_r2(filename, file_content)
            session_logger.info(f"Cloudflare R2 upload result: {r2_status}")

        session_logger.info("Session complete.")

        # Read session log content for download
        log_content = ""
        if os.path.exists(session_log_path):
            with open(session_log_path, "r", encoding="utf-8") as lf:
                log_content = lf.read()

        dl_col1, dl_col2, info_col = st.columns([1.2, 1.2, 2.6])
        with dl_col1:
            st.download_button("⬇️ Download Plan (.md)", data=file_content,
                               file_name=filename, mime="text/markdown",
                               use_container_width=True)
        with dl_col2:
            st.download_button("📜 Download Log (.log)", data=log_content,
                               file_name=os.path.basename(session_log_path),
                               mime="text/plain", use_container_width=True)
        with info_col:
            if r2_status and r2_status.get("success"):
                st.markdown(
                    f"<div class='save-bar'>☁️ Saved to R2 → <code>{r2_status['key']}</code><br>📄 Log → <code>{os.path.basename(session_log_path)}</code></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='save-bar'>📁 Saved → <code>travel_plans/{filename}</code><br>📄 Log → <code>logs/{os.path.basename(session_log_path)}</code></div>",
                    unsafe_allow_html=True
                )