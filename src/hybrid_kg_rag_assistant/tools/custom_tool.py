"""
custom_tool.py
--------------
Defines the LocalKGTool — a CrewAI BaseTool that builds an in-memory
NetworkX knowledge graph from the MOSDAC scraped CSV and supports
keyword-based entity search.  This replaces the former Neo4j-backed
Neo4jAuraDBTool with a fully local, dependency-light alternative.

CSV columns used:
  title, url, text_content, has_satellite_content, content_category,
  missions_mentioned
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Type

import networkx as nx
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------

class LocalKGInput(BaseModel):
    """Input schema for the Local Knowledge Graph Tool."""

    query: str = Field(
        ...,
        description=(
            "Natural language query or keyword(s) to search for in the "
            "local MOSDAC knowledge graph."
        ),
    )
    limit: int = Field(
        default=10,
        description="Maximum number of results to return (default: 10).",
    )


# ---------------------------------------------------------------------------
# Stop-words to skip when tokenising titles into graph nodes
# ---------------------------------------------------------------------------

_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "in", "for", "to", "is",
    "are", "was", "were", "with", "on", "at", "by", "from", "its",
    "it", "be", "as", "has", "have", "this", "that", "data", "page",
}

# Minimum token length to be treated as a meaningful keyword
_MIN_TOKEN_LEN: int = 3


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class LocalKGTool(BaseTool):
    """
    Local in-memory knowledge graph tool backed by NetworkX.

    At initialisation time the tool:
      1. Loads the MOSDAC CSV from the ``knowledge/`` directory.
      2. Tokenises the ``title`` column of each row into keywords.
      3. Builds a NetworkX graph where keyword nodes are connected by
         co-occurrence edges (keywords that appear in the same CSV row).
      4. Stores a reverse index: keyword → list of matching row dicts.

    The ``_run`` method accepts a free-text query, tokenises it the same
    way, looks up matching rows via the index, and returns a formatted
    string summary of the top *limit* hits.
    """

    name: str = "Local Knowledge Graph Tool"
    description: str = (
        "A local in-memory knowledge graph built from MOSDAC satellite and "
        "oceanographic data.  Use this tool to look up information about "
        "MOSDAC missions, satellites (INSAT, KALPANA-1, OCEANSAT, etc.), "
        "data products, and related topics.  Accepts natural language "
        "queries; returns structured text summaries."
    )
    args_schema: Type[BaseModel] = LocalKGInput

    # ------------------------------------------------------------------
    # Private state — stored as plain Python attributes, not Pydantic
    # fields, to avoid serialisation issues with large objects.
    # ------------------------------------------------------------------

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        # Locate the CSV relative to this file:
        # tools/ → hybrid_kg_rag_assistant/ → src/ → project_root/
        csv_path: Path = (
            Path(__file__).parents[3] / "knowledge" / "mosdac_scraped_data.csv"
        )

        # Build graph and index at init time (once per process)
        graph, index = self._build_graph(csv_path)
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_index", index)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(
        self, csv_path: Path
    ) -> Tuple[nx.Graph, Dict[str, List[dict]]]:
        """
        Load the CSV and construct the keyword co-occurrence graph.

        Returns
        -------
        graph : nx.Graph
            Nodes are keyword strings; edges connect keywords that
            co-occur in the same CSV row (weight = number of shared rows).
        index : dict[str, list[dict]]
            Maps every keyword to a list of row dicts containing that keyword.
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"MOSDAC CSV not found at: {csv_path}\n"
                "Ensure the 'knowledge/' directory is present in the project root."
            )

        # Load only the columns we actually need to keep memory usage low
        cols_needed = [
            "url", "title", "text_content",
            "has_satellite_content", "content_category", "missions_mentioned",
        ]
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols_needed, low_memory=False)

        # Normalise missing values
        df["title"] = df["title"].fillna("").astype(str)
        df["url"] = df["url"].fillna("").astype(str)
        df["text_content"] = df["text_content"].fillna("").astype(str)
        df["content_category"] = df.get("content_category", pd.Series(dtype=str)).fillna("")
        df["missions_mentioned"] = df.get("missions_mentioned", pd.Series(dtype=str)).fillna("")

        graph: nx.Graph = nx.Graph()
        # keyword → list of row dicts
        index: Dict[str, List[dict]] = defaultdict(list)

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            keywords = self._tokenise(row["title"])

            # Add nodes and edges
            for kw in keywords:
                if not graph.has_node(kw):
                    graph.add_node(kw)
                index[kw].append(row_dict)

            # Connect co-occurring keywords with weighted edges
            kw_list = list(keywords)
            for i, kw_a in enumerate(kw_list):
                for kw_b in kw_list[i + 1 :]:
                    if graph.has_edge(kw_a, kw_b):
                        graph[kw_a][kw_b]["weight"] += 1  # type: ignore[index]
                    else:
                        graph.add_edge(kw_a, kw_b, weight=1)

        return graph, index

    # ------------------------------------------------------------------
    # BaseTool required method
    # ------------------------------------------------------------------

    def _run(self, query: str, limit: int = 10) -> str:
        """
        Search the knowledge graph for rows matching the query.

        Strategy
        --------
        1. Tokenise the query into keywords.
        2. For each keyword, collect all indexed rows.
        3. Score rows by how many query keywords matched them.
        4. Return the top *limit* rows formatted as text.

        Parameters
        ----------
        query : str
            The search query from the agent.
        limit : int
            Maximum number of rows to return.

        Returns
        -------
        str
            Formatted text summarising the top matching knowledge graph entries.
        """
        query_keywords = self._tokenise(query)

        if not query_keywords:
            return "No meaningful keywords found in query. Please provide more specific terms."

        # Aggregate rows, counting keyword-hit frequency for ranking
        hit_count: Dict[int, int] = defaultdict(int)
        id_to_row: Dict[int, dict] = {}

        index: Dict[str, List[dict]] = object.__getattribute__(self, "_index")

        for kw in query_keywords:
            # Also match partial tokens (substring search on index keys)
            matched_keys = [k for k in index if kw in k or k in kw]
            for mk in matched_keys:
                for row_dict in index[mk]:
                    # Use url as a stable row identifier
                    row_id = id(row_dict)
                    id_to_row[row_id] = row_dict
                    hit_count[row_id] += 1

        if not hit_count:
            return (
                f"No knowledge graph entries found for query: '{query}'. "
                "Try different keywords related to MOSDAC, satellites, or missions."
            )

        # Sort by hit frequency (descending) and take top *limit*
        ranked = sorted(hit_count.items(), key=lambda x: x[1], reverse=True)[:limit]

        return self._format_results(query, ranked, id_to_row)

    # ------------------------------------------------------------------
    # Graph Visualization & Introspection
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.Graph:
        """Access the underlying NetworkX graph instance."""
        return object.__getattribute__(self, "_graph")

    @property
    def index(self) -> Dict[str, List[dict]]:
        """Access the inverted keyword-to-records index."""
        return object.__getattribute__(self, "_index")

    def generate_graph_html(
        self,
        search_filter: str | None = None,
        max_nodes: int = 50,
        height: str = "600px",
    ) -> str:
        """
        Generate an interactive PyVis network visualization in HTML.

        Parameters
        ----------
        search_filter : str, optional
            Keyword filter term. If provided, focuses on matching nodes
            and their 1-hop neighborhood.
        max_nodes : int
            Upper bound of nodes to visualize to maintain smooth 60fps rendering.
        height : str
            CSS height of the canvas.

        Returns
        -------
        str
            Complete standalone HTML string with interactive physics network.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            return (
                "<div style='color:#ef4444;padding:20px;'>"
                "PyVis is required for graph visualization. Install with: pip install pyvis"
                "</div>"
            )

        g: nx.Graph = self.graph
        if not g.nodes:
            return "<div style='color:#94a3b8;padding:20px;'>No graph nodes available.</div>"

        # Filter nodes based on search or degree centrality
        if search_filter and search_filter.strip():
            filter_term = search_filter.strip().lower()
            matched = [n for n in g.nodes if filter_term in n.lower()]
            if matched:
                sub_nodes: Set[str] = set(matched)
                for n in matched:
                    sub_nodes.update(g.neighbors(n))
                chosen_nodes = list(sub_nodes)[:max_nodes]
            else:
                # If no direct match, fallback to top connected nodes
                ranked = sorted(g.degree, key=lambda x: x[1], reverse=True)
                chosen_nodes = [n for n, _ in ranked[:max_nodes]]
        else:
            ranked = sorted(g.degree, key=lambda x: x[1], reverse=True)
            chosen_nodes = [n for n, _ in ranked[:max_nodes]]

        sub_g = g.subgraph(chosen_nodes)

        net = Network(height=height, width="100%", bgcolor="#0b0f19", font_color="#e2e8f0")
        net.barnes_hut(
            gravity=-2800,
            central_gravity=0.35,
            spring_length=90,
            spring_strength=0.04,
            damping=0.88,
        )

        # Add styled nodes with degree-proportional sizing and hover tooltips
        for node in sub_g.nodes():
            degree = g.degree(node)
            size = min(36, max(12, int(degree * 1.4)))
            if degree >= 15:
                color = "#38bdf8"  # High connectivity hub (Sky Blue)
            elif degree >= 8:
                color = "#818cf8"  # Medium connectivity (Indigo)
            else:
                color = "#94a3b8"  # Peripheral entity (Slate)

            tooltip = (
                f"<div style='font-family:sans-serif;padding:6px;'>"
                f"<b>Entity:</b> {node.upper()}<br>"
                f"<b>Connected Relationships:</b> {degree}"
                f"</div>"
            )
            net.add_node(
                node,
                label=node.upper(),
                size=size,
                color=color,
                title=tooltip,
                font={"size": 11, "face": "Helvetica", "color": "#f1f5f9"},
            )

        # Add styled edges with co-occurrence strength
        for u, v, data in sub_g.edges(data=True):
            weight = data.get("weight", 1)
            net.add_edge(
                u,
                v,
                value=weight,
                color="#334155",
                title=f"Co-occurrences: {weight}",
            )

        return net.generate_html()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> Set[str]:
        """
        Lowercase and split text into meaningful keyword tokens,
        filtering out stop-words and short tokens.
        """
        # Replace non-alphanumeric characters with spaces
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = cleaned.split()
        return {
            t for t in tokens
            if len(t) >= _MIN_TOKEN_LEN and t not in _STOP_WORDS
        }

    @staticmethod
    def _format_results(
        query: str,
        ranked: List[Tuple[int, int]],
        id_to_row: Dict[int, dict],
    ) -> str:
        """
        Build a human-readable string from the ranked result rows.

        Each entry shows the title, URL, and a short content preview.
        """
        lines: List[str] = [
            f"=== Local Knowledge Graph Results for: '{query}' ===\n"
        ]

        for rank, (row_id, hits) in enumerate(ranked, start=1):
            row = id_to_row[row_id]

            title: str = row.get("title", "N/A")
            url: str = row.get("url", "")
            content: str = row.get("text_content", "")
            category: str = str(row.get("content_category", ""))
            missions: str = str(row.get("missions_mentioned", ""))

            # Truncate content preview to 300 characters
            preview = (content[:300] + "…") if len(content) > 300 else content

            entry = (
                f"[{rank}] Title   : {title}\n"
                f"    URL     : {url}\n"
            )
            if category and category not in ("nan", ""):
                entry += f"    Category: {category}\n"
            if missions and missions not in ("nan", "[]", ""):
                entry += f"    Missions: {missions}\n"
            entry += f"    Matched : {hits} keyword(s)\n"
            if preview.strip():
                entry += f"    Preview : {preview}\n"

            lines.append(entry)

        return "\n".join(lines)
