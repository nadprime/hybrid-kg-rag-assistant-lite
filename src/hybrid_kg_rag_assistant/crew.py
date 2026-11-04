"""
crew.py
-------
Defines the HybridKgRagAssistant CrewAI crew.

This module wires together:
  - Two agents: hybrid_data_retriever and response_synthesizer
  - Two tasks:  hybrid_data_retrieval_task and response_synthesis_task
  - Tools:      RagTool (ChromaDB + Google embeddings) and LocalKGTool
                (in-memory NetworkX knowledge graph — replaces Neo4j)

The crew runs sequentially and maintains memory across turns so
multi-turn conversations feel coherent.
"""

import os
import warnings
import logging
from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import RagTool
from dotenv import load_dotenv

from .tools.custom_tool import LocalKGTool

# ---------------------------------------------------------------------------
# Suppress common noisy deprecation warnings from third-party libraries.
# These do not affect functionality.
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="alembic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince211.*")
warnings.filterwarnings("ignore", message=".*Using extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Extra keys.*")
warnings.filterwarnings("ignore")  # Catch-all for any remaining warnings

# ---------------------------------------------------------------------------
# Environment & LLM configuration
# ---------------------------------------------------------------------------

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini/gemini-3.8-flash"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"


def _get_llm() -> LLM:
    """
    Retrieve or create the LLM instance dynamically using the current API key.
    This allows updating the key at runtime without requiring a full application restart.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    model = os.getenv("MODEL", DEFAULT_MODEL).strip()
    if not model.startswith("gemini/"):
        model = f"gemini/{model}"

    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is not configured. "
            "Please provide a valid Google Gemini API key in your .env file."
        )
    return LLM(model=model, api_key=api_key)


# ---------------------------------------------------------------------------
# Crew definition
# ---------------------------------------------------------------------------

@CrewBase
class HybridKgRagAssistant:
    """
    CrewAI crew for the Hybrid KG-RAG Assistant.

    Combines a vector-store RAG tool (ChromaDB) with a local NetworkX
    knowledge graph to answer questions about MOSDAC satellite data.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def hybrid_data_retriever(self) -> Agent:
        """
        Data retrieval agent equipped with both a local knowledge graph tool
        and a vector-store RAG tool.

        - LocalKGTool: keyword-graph search over the MOSDAC CSV using NetworkX
        - RagTool: semantic vector search over the MOSDAC CSV via ChromaDB
        """
        # Resolve the project root from this file's location:
        # crew.py → hybrid_kg_rag_assistant/ → src/ → project_root/
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        csv_path = os.path.join(project_root, "knowledge", "mosdac_scraped_data.csv")

        # Synchronize Google API key with downstream embedder requirements
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if api_key:
            os.environ["EMBEDDINGS_GOOGLE_API_KEY"] = api_key
            os.environ["GEMINI_API_KEY"] = api_key

        tools = []

        # 1. Local Knowledge Graph Tool (NetworkX in-memory graph)
        try:
            local_kg_tool = LocalKGTool()
            tools.append(local_kg_tool)
        except Exception as exc:
            LOGGER.warning("LocalKGTool could not be initialised: %s", exc)

        # 2. RAG Tool (ChromaDB + Google's current embedding model)
        try:
            rag_config = {
                "embedding_model": {
                    "provider": "google-generativeai",
                    "config": {
                        "model_name": DEFAULT_EMBEDDING_MODEL,
                    },
                },
                "vectordb": {
                    "provider": "chromadb",
                    "config": {
                        "collection_name": "hybrid-kg-rag-collection",
                        "dir": os.path.join(project_root, "chroma_db"),
                    },
                },
            }
            rag_tool = RagTool(config=rag_config)
            tools.append(rag_tool)
        except Exception as exc:
            # Keep the local graph available when the optional vector store is unavailable.
            LOGGER.warning("RAG vector store initialisation deferred: %s", exc)

        # Guarantee at least one functional retrieval tool
        if not tools:
            tools.append(LocalKGTool())

        return Agent(
            config=self.agents_config["hybrid_data_retriever"],  # type: ignore[index]
            verbose=True,
            allow_delegation=False,
            tools=tools,
            llm=_get_llm(),
        )

    @agent
    def response_synthesizer(self) -> Agent:
        """
        Response synthesis agent.

        Receives the structured data package from the retrieval agent and
        crafts a clear, conversational answer for the end user.
        """
        return Agent(
            config=self.agents_config["response_synthesizer"],  # type: ignore[index]
            verbose=True,
            allow_delegation=False,
            llm=_get_llm(),
        )

    @task
    def hybrid_data_retrieval_task(self) -> Task:
        """Task that drives the data retrieval agent."""
        return Task(
            config=self.tasks_config["hybrid_data_retrieval_task"],  # type: ignore[index]
        )

    @task
    def response_synthesis_task(self) -> Task:
        """Task that drives the response synthesis agent."""
        return Task(
            config=self.tasks_config["response_synthesis_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """
        Assemble and return the HybridKgRagAssistant crew.

        Runs tasks sequentially. Multi-turn dialogue context is handled
        cleanly by the application layer so the execution remains fast,
        predictable, and free of unnecessary vector-database overhead.
        """
        return Crew(
            agents=self.agents,   # Populated automatically by @agent decorators
            tasks=self.tasks,     # Populated automatically by @task decorators
            process=Process.sequential,  # Tasks run sequentially
            verbose=True,
        )
