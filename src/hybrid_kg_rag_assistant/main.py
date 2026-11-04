"""
main.py
-------
Command-line entry-point for the Hybrid KG-RAG Assistant.

Provides four sub-commands that delegate to the CrewAI crew:

  run    — interactive query from stdin
  train  — train the crew for N iterations
  replay — replay from a specific task ID
  test   — evaluate the crew over N iterations

Usage
-----
    python -m hybrid_kg_rag_assistant.main   (defaults to ``run``)

Or, if installed via pyproject.toml:
    hybrid-kg-rag-assistant run
"""

#!/usr/bin/env python

import contextlib
import os
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment setup — must happen before any crew imports
# ---------------------------------------------------------------------------

# Direct all CrewAI persistence files into the project tree (not ~/.crewai)
project_root = Path(__file__).parent
storage_dir = project_root / "crewai_storage"
os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress noisy deprecation warnings globally
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Suppress persistent Pydantic stderr warning on import
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to /dev/null."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# Import the crew with stderr suppressed to hide the unavoidable pydantic notice
with suppress_stderr():
    from hybrid_kg_rag_assistant.crew import HybridKgRagAssistant

# Fine-grained runtime warning filters (belt-and-suspenders approach)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince.*")
warnings.filterwarnings("ignore", message=".*No path_separator found.*")
warnings.filterwarnings("ignore", message=".*model_fields.*")
warnings.filterwarnings("ignore", message=".*Using extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Extra keys.*")


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Prompt the user for a query and run the crew once, printing the result.

    This is the primary interactive entry-point for CLI usage.
    """
    query = input("Enter your query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    inputs = {"query": query}

    try:
        result = HybridKgRagAssistant().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise RuntimeError(f"An error occurred while running the crew: {e}") from e

    print(f"\nResult:\n{result}")


def train() -> None:
    """
    Train the crew for a given number of iterations.

    Usage:
        python main.py train <n_iterations> <output_filename>
    """
    inputs = {
        "query": "Tell me about MOSDAC mission and its objectives",
    }
    try:
        HybridKgRagAssistant().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while training the crew: {e}") from e


def replay() -> None:
    """
    Replay the crew execution from a specific task ID.

    Usage:
        python main.py replay <task_id>
    """
    try:
        HybridKgRagAssistant().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise RuntimeError(f"An error occurred while replaying the crew: {e}") from e


def test() -> None:
    """
    Evaluate the crew over a number of iterations using a chosen LLM judge.

    Usage:
        python main.py test <n_iterations> <eval_llm>
    """
    inputs = {
        "query": "Tell me about MOSDAC mission and its objectives",
    }
    try:
        HybridKgRagAssistant().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise RuntimeError(f"An error occurred while testing the crew: {e}") from e


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
