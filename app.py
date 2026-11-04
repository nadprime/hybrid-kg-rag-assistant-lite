"""
app.py
------
Streamlit conversational application for the Hybrid Knowledge Graph RAG Assistant.

Architecture Highlights:
  - Orchestration : CrewAI multi-agent sequential pipeline
  - Knowledge Base: Local NetworkX entity-relationship graph + ChromaDB vector store
  - Intelligence  : Google Gemini (configurable via MODEL)
  - Dataset       : MOSDAC satellite & oceanographic data (ISRO)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import Generator, List

from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup & Environment Initialisation
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Suppress noisy deprecation messages from downstream dependencies
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Keep CrewAI operational artifacts localized within the project directory
storage_dir = src_dir / "hybrid_kg_rag_assistant" / "crewai_storage"
storage_dir.mkdir(parents=True, exist_ok=True)
os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)

# Load environment configuration (.env)
load_dotenv()
_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
if _api_key:
    # Synchronize keys expected by Google Generative AI, LiteLLM, and ChromaDB
    os.environ["GEMINI_API_KEY"] = _api_key
    os.environ["EMBEDDINGS_GOOGLE_API_KEY"] = _api_key

MODEL_NAME = os.getenv("MODEL", "gemini/gemini-3.8-flash").strip()
MAX_QUERY_LENGTH = 2_000

# ---------------------------------------------------------------------------
# Streamlit Page Setup & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid KG RAG Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0e7490 100%);
        padding: 2rem 2.25rem;
        border: 1px solid rgba(125, 211, 252, 0.25);
        border-radius: 18px;
        margin-bottom: 1.25rem;
        text-align: left;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
    }
    .main-header h1 { color: #f8fafc; margin: 0; font-size: 2.25rem; font-weight: 750; letter-spacing: -0.02em; }
    .main-header h3 { color: #bae6fd; margin: 0.55rem 0 0 0; font-weight: 450; font-size: 1.08rem; }
    .main-header p  { color: #cbd5e1; margin: 1rem 0 0 0; font-size: 0.86rem; }
    .model-pill {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(14, 116, 144, 0.35);
        color: #cffafe;
        font-size: 0.78rem;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.35rem 0.8rem;
        margin-bottom: 0.65rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #6366f1;
        background-color: #ffffff;
        color: #4f46e5;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #4f46e5;
        color: #ffffff;
        border-color: #4f46e5;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.25);
    }
    
    .debug-container {
        background-color: #111827;
        color: #f3f4f6;
        padding: 0.85rem;
        border-radius: 8px;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.78rem;
        line-height: 1.45;
        max-height: 380px;
        overflow-y: auto;
        border: 1px solid #374151;
    }
    .debug-title {
        color: #10b981;
        font-weight: 600;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
    }
    .agent-step {
        border-left: 3px solid #3b82f6;
        padding: 0.4rem 0.6rem;
        margin: 0.35rem 0;
        background-color: #1f2937;
        border-radius: 0 4px 4px 0;
    }
    .success-step {
        border-left: 3px solid #10b981;
        background-color: #064e3b;
    }
    .agent-thinking {
        border-left: 3px solid #f59e0b;
        background-color: #451a03;
    }
    .processing-step {
        border-left: 3px solid #8b5cf6;
        background-color: #2e1065;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "debug_output" not in st.session_state:
    st.session_state.debug_output = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ---------------------------------------------------------------------------
# Knowledge Graph Resource Loader
# ---------------------------------------------------------------------------
@st.cache_resource
def get_local_kg_tool():
    """Cache the LocalKGTool instance so the NetworkX graph loads only once."""
    from hybrid_kg_rag_assistant.tools.custom_tool import LocalKGTool
    return LocalKGTool()


# ---------------------------------------------------------------------------
# Execution Pipeline
# ---------------------------------------------------------------------------
def run_agent_pipeline(query: str, chat_history: List[dict]) -> str:
    """
    Executes the multi-agent workflow to retrieve context and synthesize an answer.

    Parameters
    ----------
    query : str
        The user's current question.
    chat_history : list of dict
        Previous conversation exchanges.

    Returns
    -------
    str
        The synthesized response text from the crew.
    """
    query = query.strip()
    if not query:
        raise ValueError("Please enter a question before submitting.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Please keep questions under {MAX_QUERY_LENGTH:,} characters.")

    # Verify API credentials without exposing the secret in the UI or logs.
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_api_key or google_api_key == "your_google_api_key_here":
        raise ValueError(
            "Google API Key is missing. Please set `GOOGLE_API_KEY` in your `.env` file."
        )

    # Lazily import crew to keep startup instant
    from hybrid_kg_rag_assistant.crew import HybridKgRagAssistant

    # Format previous dialogue context (excluding the current turn)
    context_lines = []
    if chat_history and len(chat_history) > 1:
        # Use previous 2-3 exchanges to avoid context flooding
        recent = chat_history[:-1][-4:]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            snippet = msg["content"][:180] + ("..." if len(msg["content"]) > 180 else "")
            context_lines.append(f"{role}: {snippet}")

    inputs = {"query": query}
    if context_lines:
        conversation_context = "\n".join(context_lines)
        inputs["query"] = (
            f"Current Question: {query}\n\n"
            f"Recent Context:\n{conversation_context}\n\n"
            "Please provide a contextually aware, conversational response."
        )

    # Record activity checkpoints for sidebar console
    query_lower = query.lower()
    st.session_state.debug_output.append("🚀 [SYSTEM] Initializing Hybrid RAG + KG pipeline")
    st.session_state.debug_output.append("🔍 [RETRIEVAL] Accessing local NetworkX entity graph")
    
    if any(k in query_lower for k in ["kalpana", "insat", "satellite", "oceansat"]):
        st.session_state.debug_output.append("🛰️ [GRAPH] Satellite domain entities prioritized")
    elif any(k in query_lower for k in ["isro", "mission", "irs"]):
        st.session_state.debug_output.append("🚀 [GRAPH] ISRO mission hierarchy activated")
    elif any(k in query_lower for k in ["weather", "ocean", "cyclone"]):
        st.session_state.debug_output.append("🌊 [DOMAIN] Meteorological/Oceanographic focus")
    else:
        st.session_state.debug_output.append("📦 [DATA] Multi-hop relationship search active")

    st.session_state.debug_output.append("🤖 [AGENT] DataRetrieval Expert: Processing")
    st.session_state.debug_output.append("✍️ [AGENT] ResponseSynthesizer: Formulating answer")

    # Run CrewAI sequential execution
    crew_instance = HybridKgRagAssistant().crew()
    result = crew_instance.kickoff(inputs=inputs)

    st.session_state.debug_output.append("✅ [STATUS] Execution completed successfully")
    return str(result)


def stream_words(text: str) -> Generator[str, None, None]:
    """
    Generator that streams a full text string word-by-word with natural cadence.
    """
    words = text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(0.015)


# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="main-header">
    <h1>🛰️ Hybrid Knowledge Graph RAG Assistant</h1>
    <h3>Conversational AI for ISRO / MOSDAC Satellite & Oceanographic Data</h3>
    <p>Powered by CrewAI • NetworkX Knowledge Graph • ChromaDB • grounded retrieval</p>
    <span class="model-pill">Gemini model: {MODEL_NAME}</span>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Controls & Activity Console
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 Conversation Controls")
    if _api_key and _api_key != "your_google_api_key_here":
        st.success("Gemini API configured", icon="✅")
    else:
        st.warning("Add GOOGLE_API_KEY to .env", icon="🔐")
    st.caption(f"Active model: `{MODEL_NAME}`")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.debug_output = []
        st.session_state.pending_query = None
        st.rerun()

    if st.session_state.chat_history:
        user_msgs = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
        st.metric("Questions Asked", user_msgs)

    st.markdown("---")
    st.markdown("### 🧠 Agent Activity Console")

    # Render dynamic activity log
    if st.session_state.debug_output:
        st.markdown('<div class="debug-container"><div class="debug-title">Console Events</div>', unsafe_allow_html=True)
        # Display the most recent 6 events
        recent_events = st.session_state.debug_output[-6:]
        for entry in recent_events:
            if "✅" in entry or "[STATUS]" in entry:
                cls = "success-step"
            elif "[AGENT]" in entry:
                cls = "agent-thinking"
            elif "[RETRIEVAL]" in entry or "[GRAPH]" in entry:
                cls = "processing-step"
            else:
                cls = "agent-step"
            st.markdown(f'<div class="{cls} agent-step"><code>{entry}</code></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
        <div class="debug-container">
            <div class="debug-title">Console Status: Idle</div>
            <span style="color:#9ca3af;">
            • Retrieval Agent: Standby<br>
            • Synthesis Agent: Standby<br>
            • NetworkX KG: Ready<br>
            • ChromaDB Vector Store: Ready
            </span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Complete activity log expander
    if len(st.session_state.debug_output) > 6:
        with st.expander("📋 View Complete Activity Log"):
            for idx, item in enumerate(st.session_state.debug_output, 1):
                st.text(f"{idx}. {item}")

    st.markdown("---")
    with st.expander("ℹ️ Architecture Overview", expanded=False):
        st.markdown(
            """
        - **Local Knowledge Graph**: In-memory NetworkX co-occurrence graph parsed from MOSDAC data.
        - **Vector RAG**: ChromaDB semantic passage store.
        - **Agents**: Sequential multi-agent workflow via CrewAI.
        - **Model**: Configurable Gemini model via `MODEL`.
        """
        )

# ---------------------------------------------------------------------------
# Navigation Tabs: Chat Assistant & Knowledge Graph Explorer
# ---------------------------------------------------------------------------
tab_chat, tab_graph = st.tabs(["💬 Conversational Assistant", "🕸️ Interactive Knowledge Graph Explorer"])

# ===========================================================================
# TAB 1: Conversational Assistant
# ===========================================================================
with tab_chat:
    # Replay Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Suggested Prompts (Visible on fresh session)
    example_prompts = [
        "Tell me about MOSDAC mission and its objectives.",
        "What is Kalpana-1 satellite and its significance?",
        "Explain ISRO's earth observation satellites.",
        "What are the applications of remote sensing?",
        "Describe INSAT satellite series and its uses.",
        "What is the role of meteorological satellites in weather forecasting?",
    ]

    if not st.session_state.chat_history:
        st.markdown("#### 💡 Quick Questions to Try:")
        row1_cols = st.columns(3)
        row2_cols = st.columns(3)

        for i in range(3):
            if row1_cols[i].button(example_prompts[i], key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = example_prompts[i]
                st.rerun()

        for i in range(3, 6):
            if row2_cols[i - 3].button(example_prompts[i], key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = example_prompts[i]
                st.rerun()

    # User Input & Execution Handling
    user_text = st.chat_input(
        "Ask about satellites, ISRO missions, or weather data...",
        max_chars=MAX_QUERY_LENGTH,
    )

    # Resolve active query (typed input takes priority over clicked prompt)
    active_query = user_text or st.session_state.pending_query
    if active_query:
        st.session_state.pending_query = None

        # Render User Turn
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Render Assistant Turn with interactive status indicator
        with st.chat_message("assistant"):
            try:
                with st.status("Analyzing question across Knowledge Graph & RAG...", expanded=True) as status:
                    st.write("🔍 Extracting entities & querying NetworkX graph...")
                    response_text = run_agent_pipeline(active_query, st.session_state.chat_history)
                    st.write(f"✨ Synthesizing final answer with {MODEL_NAME}...")
                    status.update(label="Response generated!", state="complete", expanded=False)

                # Stream the synthesized answer cleanly
                st.write_stream(stream_words(response_text))
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except ValueError as val_err:
                st.error(f"⚠️ Configuration Notice: {val_err}")
            except Exception:
                LOGGER.exception("Agent pipeline failed")
                st.error("❌ The assistant could not complete that request.")
                st.info(
                    "Check your network connection and Gemini configuration, then try again."
                )


# ===========================================================================
# TAB 2: Interactive Knowledge Graph Explorer
# ===========================================================================
with tab_graph:
    st.markdown("### 🕸️ MOSDAC Entity & Relationship Graph")
    st.caption(
        "Interactive visualization of the in-memory NetworkX knowledge graph. "
        "Drag nodes, zoom in/out, and hover over entities to inspect connectivity."
    )

    try:
        kg_tool = get_local_kg_tool()
        g = kg_tool.graph

        # Graph Metrics Overview
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        total_nodes = g.number_of_nodes()
        total_edges = g.number_of_edges()
        avg_degree = (2 * total_edges / max(1, total_nodes)) if total_nodes else 0
        density = nx.density(g) if total_nodes > 1 else 0

        col_m1.metric("Domain Entities (Nodes)", total_nodes)
        col_m2.metric("Co-occurrences (Edges)", total_edges)
        col_m3.metric("Avg Degree Connectivity", f"{avg_degree:.1f}")
        col_m4.metric("Graph Density", f"{density:.3f}")

        # Interactive Controls
        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            filter_query = st.text_input(
                "🔎 Filter Entities by Keyword",
                placeholder="e.g. kalpana, insat, oceansat, cyclone, meteorological, data",
                help="Enter a keyword to isolate specific entities and their connected 1-hop neighbors.",
            )
        with ctrl_col2:
            max_nodes_slider = st.slider(
                "Max Visible Nodes",
                min_value=15,
                max_value=max(15, total_nodes),
                value=min(45, total_nodes),
                help="Control network density for optimal physics layout.",
            )

        # Generate and render interactive HTML canvas
        html_content = kg_tool.generate_graph_html(
            search_filter=filter_query,
            max_nodes=max_nodes_slider,
            height="580px",
        )
        components.html(html_content, height=600, scrolling=False)

        # Entity Hubs Analysis Table
        with st.expander("📊 Key Entity Hubs (Highest Degree Centrality)", expanded=False):
            ranked_nodes = sorted(g.degree, key=lambda x: x[1], reverse=True)[:10]
            hub_df = pd.DataFrame(
                [
                    {
                        "Rank": idx,
                        "Entity Keyword": node.upper(),
                        "Connected Entities (Degree)": deg,
                        "Significance": "Core domain anchor" if deg >= 15 else "Domain connector",
                    }
                    for idx, (node, deg) in enumerate(ranked_nodes, 1)
                ]
            )
            st.dataframe(hub_df, use_container_width=True, hide_index=True)

    except Exception:
        LOGGER.exception("Knowledge Graph visualization failed")
        st.error("Could not load the Knowledge Graph visualization.")
