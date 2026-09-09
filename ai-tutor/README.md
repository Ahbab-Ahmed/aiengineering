# 🎓 AI Engineering Tutor

> An interactive learning app powered by **Hybrid RAG + HyDE + gpt-4o**, built on Chip Huyen's *AI Engineering* (O'Reilly, 2025).

---

## Prerequisites

Install these once on your machine before anything else.

| Tool | Install | Verify |
|------|---------|--------|
| **Python 3.11+** | [python.org](https://python.org) | `python3 --version` |
| **uv** (Python package manager) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| **Docker Desktop** | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) | `docker --version` |
| **OpenAI API key** | [platform.openai.com](https://platform.openai.com) | — |

---

## One-Time Setup

### 1. Clone / open the project
```bash
cd /path/to/aiengineering
```

### 2. Install Python dependencies
```bash
uv sync
```

### 3. Add your OpenAI API key
Create a `.env` file at the project root:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 4. Add the book PDF
Place the PDF at:
```
data/ai_engineering.pdf
```

### 5. Start Qdrant (vector database)
```bash
docker compose up -d
```

### 6. Index the book (~3–4 min, costs ~$0.08 in OpenAI API)
```bash
make index
```
> ✅ This only needs to be done **once**. It extracts all text, detects chapters, creates parent-child chunks, and embeds everything into Qdrant.

---

## Daily Use

```bash
make start    # Start Qdrant + launch the web app
```
Then open **http://localhost:8000** in your browser.

```bash
make stop     # Shut everything down when done
```

---

## All Commands

```bash
make help      # Show all available commands
make start     # Start Qdrant + web app (http://localhost:8000)
make stop      # Stop everything
make restart   # Stop then start fresh
make status    # Check if Qdrant + web app are running
make open      # Open http://localhost:8000 in browser
make index     # Delete old data and re-index the PDF
```

---

## Using the App

### 3 Learning Modes

| Mode | What it does | How to activate |
|------|-------------|-----------------|
| **📖 Learn** | Deep structured explanation: concept → why it matters → how it works → real example → book reference | Default mode. Just ask anything. |
| **🧠 Quiz** | Generates 5 questions, grades your answers, explains correct ones | Switch to Quiz tab, type a topic |
| **🗺️ Curriculum** | Guided 10-session course through the whole book | Switch to Curriculum tab, click a session in the sidebar |

### Example questions to ask

```
# Learn mode
What is AI Engineering and how does it differ from ML Engineering?
Explain RAG — how does retrieval work?
When should I finetune vs use RAG?
How do AI Agents use tools?
What is KV cache and why does it matter for inference?

# Quiz mode
Quiz me on Prompt Engineering
Quiz me on RAG
Quiz me on Finetuning

# Curriculum mode
(Click session ① in the sidebar to start the structured course)
```

### Features

| Feature | How to use |
|---------|-----------|
| **📝 Notes** | Click the `📝` button (top-right) — notes auto-save per session |
| **👍/👎 Feedback** | Appears below every response — click to rate |
| **Sessions** | Each conversation is saved — click `+ New Session` to start fresh |
| **Curriculum progress** | Sessions show ✓ once completed (tracked in browser) |
| **Source chips** | Every response shows `Ch.N p.X` — exactly where in the book it came from |

---

## How It Works (Technical)

```
Your question
    ↓
Query expansion (gpt-4o-mini)
    ↓
HyDE: generate hypothetical answer → embed it   ← better semantic matching
    ↓
Multi-query: 3 rephrasings → retrieve for each  ← catches missed docs
    ↓
BM25 keyword search (child chunks, cached on disk)
    ↓
Qdrant MMR vector search (text-embedding-3-large)
    ↓
Reciprocal Rank Fusion (combine all result lists)
    ↓
Cross-encoder reranking (ms-marco-MiniLM, runs locally)
    ↓
Return parent content (1400-char context) to LLM
    ↓
gpt-4o streams structured teaching response
    ↓
Live markdown rendering as tokens arrive
```

**Estimated cost per question:** ~$0.01–0.02

---

## Project Structure

```
aiengineering/
├── Makefile                  # Start/stop/index commands
├── docker-compose.yml        # Qdrant vector database
├── pyproject.toml            # Python dependencies (uv)
├── .env                      # OPENAI_API_KEY (not committed)
├── data/
│   ├── ai_engineering.pdf    # The book (you provide this)
│   ├── sessions.db           # SQLite: conversations + notes + feedback
│   └── bm25_cache.pkl        # BM25 index cache (auto-generated)
└── code/rag/
    ├── app.py                # FastAPI server
    ├── config.py             # Settings + curriculum data
    ├── indexing.py           # PDF → Qdrant (run once)
    ├── retriever.py          # HybridRetriever (BM25+MMR+HyDE+rerank)
    ├── tutor.py              # gpt-4o streaming + system prompts
    ├── session_manager.py    # SQLite sessions/notes/feedback
    └── static/
        ├── index.html        # Web UI
        ├── style.css         # Dark glassmorphism theme
        └── app.js            # Frontend SPA logic
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on start | Docker isn't running — open Docker Desktop first |
| Loading screen never finishes | Check `.env` has a valid `OPENAI_API_KEY` |
| `Collection not found` error | Run `make index` to index the book first |
| Port 8000 already in use | Run `make stop` then `make start` |
| Slow first startup (~10s) | Cross-encoder model loading, normal on first run |
