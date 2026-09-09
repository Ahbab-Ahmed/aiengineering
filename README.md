# 🤖 AI Engineering — Hands-On Portfolio

A collection of production-grade AI engineering systems built alongside *AI Engineering* by Chip Huyen (O'Reilly, 2025).

---

## 🚀 Projects Overview

| Project | Description | Key AI Engineering Techniques | Stack | Documentation |
|---|---|---|---|:---:|
| **[medical-rag](medical-rag/)** | **Medical Guideline Assistant** grounded in 21 official MOHFW standard treatment guidelines with hard boundaries against personal diagnosis | • Hierarchical Parent-Child Chunking<br>• Hybrid Search (Dense + BM25Okapi)<br>• Reciprocal Rank Fusion (RRF)<br>• Cross-Encoder Reranking (`ms-marco`)<br>• Calibrated Confidence Gating (-4.0)<br>• Multi-Tier Input/Output Guardrails<br>• 24-Case Automated Benchmark (96% accuracy, 100% safe refusal) | FastAPI<br>Qdrant<br>OpenAI GPT-4o<br>SentenceTransformers<br>Guardrails AI Hub | [📖 Deep Dive](medical-rag/DEEP_DIVE.md)<br>[🎙️ Presentation Guide](medical-rag/PRESENTATION_GUIDE.md)<br>[📄 README](medical-rag/README.md) |

---

## 🛠️ Repository Setup & Installation

Dependencies across projects are managed at the workspace root using [`uv`](https://astral.sh/uv).

```bash
# 1. Clone the repository
git clone https://github.com/Ahbab-Ahmed/aiengineering.git
cd aiengineering

# 2. Sync all virtual environment dependencies
uv sync

# 3. Configure environment variables
# (Copy .env.example into each project folder and add your OPENAI_API_KEY)
cp medical-rag/.env.example medical-rag/.env
```

---

## 🏥 Running Medical RAG

```bash
cd medical-rag

# 1. Start Qdrant vector database container
docker compose up -d

# 2. Ingest & index MOHFW guidelines
make index

# 3. Run the automated 24-case benchmark
make evaluate

# 4. Start the interactive assistant server (port 8001)
make start
```
Open **http://localhost:8001** to use the application.
