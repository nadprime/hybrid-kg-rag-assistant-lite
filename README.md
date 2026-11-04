# 🛰️ Hybrid Knowledge Graph RAG Assistant

A production-ready conversational AI assistant built for **MOSDAC** (Meteorological & Oceanographic Satellite Data Archival Centre, ISRO). This system combines **Vector-Store RAG (Retrieval-Augmented Generation)** with an **In-Memory Knowledge Graph** to provide highly factual, relationship-grounded responses about Indian satellites, earth-observation missions, and meteorological data.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│                      CrewAI Pipeline                      │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │       hybrid_data_retriever (Agent 1)               │  │
│  │                                                     │  │
│  │   ┌─────────────────────┐   ┌───────────────────┐   │  │
│  │   │  LocalKGTool        │   │  RagTool          │   │  │
│  │   │  (NetworkX Graph)   │   │  (ChromaDB +      │   │  │
│  │   │  • Entity Relations │   │   Gemini Embeds)  │   │  │
│  │   │  • Inverted Index   │   │  • Semantic Pass. │   │  │
│  │   └─────────────────────┘   └───────────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
│                            │                              │
│                            ▼                              │
│  ┌─────────────────────────────────────────────────────┐  │
│  │       response_synthesizer (Agent 2)                │  │
│  │       • Model: Google Gemini 3.8 Flash               │  │
│  │       • Cross-references retrieved graph + passages │  │
│  │       • Synthesizes multi-turn conversational reply │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
    │
    ▼
Streaming Markdown Response (Streamlit UI / CLI)
```

### 🧩 Core Components

| Component | Technology | Purpose |
|---|---|---|
| **Knowledge Graph** | [NetworkX](https://networkx.org/) | In-memory entity & co-occurrence graph parsed from MOSDAC records; $O(1)$ inverted index lookup. |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Semantic vector indexing over unstructured passage chunks. |
| **Embeddings** | `models/gemini-embedding-001` | High-dimensional semantic vectors via Google Generative AI. |
| **LLM Reasoning** | [Google Gemini 3.8 Flash](https://aistudio.google.com/) | Stable, fast response synthesis and contextual reasoning. |
| **Multi-Agent Orchestration** | [CrewAI](https://crewai.com/) | Sequential pipeline managing specialized retrieval and synthesis roles. |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Responsive web chat interface with live activity logs and streaming output. |

> **Self-Contained & Offline-Resilient**: No external graph database (e.g. Neo4j) is needed. The knowledge graph is constructed entirely in-memory from `knowledge/mosdac_scraped_data.csv` on startup.

---

## 💡 Why Hybrid KG-RAG? (Interview Talking Points)

- **Vector RAG Limitation**: Traditional vector search relies solely on cosine similarity in embedding space. It excels at answering *"What does X mean?"* but struggles with structured connections, multi-hop queries, and entity relationships (e.g., *"Which satellites carry payloads for cyclone tracking?"*).
- **Knowledge Graph Strength**: A graph explicitly captures entities (satellites, payloads, missions) and relational edges (co-occurrence, missions mentioned).
- **The Hybrid Advantage**: Combining both approaches allows the retrieval agent to query semantic passages from ChromaDB while grounding facts against relational links from NetworkX, drastically reducing hallucinations and improving answer precision.

---

## 📁 Repository Structure

```
hybrid-kg-rag-assistant/
├── app.py                          # Streamlit interactive web chat interface
├── Dockerfile                      # Container deployment specification
├── pyproject.toml                  # Project packaging and metadata
├── requirements.txt                # Pinned production dependencies
├── .env.example                    # Environment configuration template
├── .gitignore                      # Git exclusion rules (protects credentials & venv)
├── knowledge/
│   └── mosdac_scraped_data.csv     # Curated MOSDAC dataset (satellites, sensors, missions)
└── src/
    └── hybrid_kg_rag_assistant/
        ├── __init__.py
        ├── crew.py                 # Multi-agent crew definition & task wiring
        ├── main.py                 # CLI execution entry-point (run, train, replay, test)
        ├── config/
        │   ├── agents.yaml         # Agent role, goal, and backstory prompts
        │   └── tasks.yaml          # Task instructions & expected output schemas
        └── tools/
            ├── __init__.py
            └── custom_tool.py      # LocalKGTool (NetworkX knowledge graph implementation)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your machine.
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/hybrid-kg-rag-assistant.git
cd hybrid-kg-rag-assistant
```

### 3. Create & Activate Virtual Environment

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
# Or in cmd.exe: venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy the template and add your Gemini API key:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` in any text editor:
```env
GOOGLE_API_KEY=AIzaSyYourActualKeyHere
MODEL=gemini/gemini-3.8-flash
```

---

## 🖥️ Running the Application

### Option A: Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
- Opens automatically in your browser at `http://localhost:8501`.
- **Tab 1 — 💬 Conversational Assistant**: Real-time response streaming, one-click example prompts, and live agent activity console.
- **Tab 2 — 🕸️ Knowledge Graph Explorer**: Interactive physics-based visualization of the MOSDAC NetworkX graph (drag nodes, zoom, filter by keyword, and inspect entity connectivity & degree centrality).

### Option B: Command Line Interface (CLI)
```bash
python src/hybrid_kg_rag_assistant/main.py
```

### Option C: Docker Container
```bash
# Build the Docker image
docker build -t hybrid-kg-rag-assistant .

# Run the container
docker run -p 8501:8501 --env-file .env hybrid-kg-rag-assistant
```

---

## 💬 Sample Queries

- *"Tell me about the MOSDAC mission and its core objectives."*
- *"What is the Kalpana-1 satellite and its historical significance?"*
- *"Explain ISRO's earth observation satellites and remote sensing applications."*
- *"Describe the INSAT satellite series and their meteorological instruments."*
- *"How do oceanographic satellites monitor sea surface temperatures and cyclones?"*

---

## 📤 Pushing to GitHub

Ensure your virtual environment is active and `.env` contains your private key (it is protected by `.gitignore`):

```bash
# 1. Initialize git
git init

# 2. Stage files
git add .

# 3. Create your initial commit
git commit -m "feat: complete hybrid KG-RAG assistant with NetworkX, CrewAI, and Gemini 3.8 Flash"

# 4. Set main branch and link to remote repository
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 5. Push to GitHub
git push -u origin main
```

---

Built for educational and research demonstration purposes.
