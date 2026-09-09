# 🤖 AI Engineering — Hands-On Portfolio

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![Evaluation](https://img.shields.io/badge/Benchmark-96%25%20Accuracy-success.svg)](medical-rag/evaluator.py)
[![Safety](https://img.shields.io/badge/Refusal%20Rate-100%25%20Safe-brightgreen.svg)](medical-rag/safety.py)

A portfolio of production-grade AI engineering systems implementing state-of-the-art information retrieval, safety-critical guardrails, calibrated reranking, and evaluation harnesses.

---

## 🚀 Projects Overview

| Project | Description | Core AI Engineering Technologies | Key Metrics | Documentation |
|---|---|---|:---:|:---:|
| **[medical-rag](medical-rag/)** | **Medical Guideline Assistant** grounded strictly in 21 official MOHFW standard treatment guidelines with hard boundaries against personal diagnosis | • Hierarchical Parent-Child Chunking (1200 / 300 chars)<br>• Dual Hybrid Search (OpenAI 3072D + BM25Okapi)<br>• Reciprocal Rank Fusion (RRF, $K=60$)<br>• Local Cross-Encoder Reranking (`ms-marco-MiniLM`)<br>• Calibrated Cross-Encoder Confidence Gate ($-4.0$)<br>• Multi-Tier Guardrails (0ms Regex + ALBERT Detoxify)<br>• Programmatic Citation Integrity Verification<br>• Server-Sent Events (SSE) Streaming (~350ms TTFT) | **96% Overall Accuracy**<br>**100% Refusal Rate**<br>(18/18 unsafe queries blocked)<br>15,559 Indexed Chunks | [📖 Deep Dive Manual](medical-rag/DEEP_DIVE.md)<br>[📄 Project README](medical-rag/README.md) |

---

## 🏛️ Flagship Architecture: Medical Guideline Assistant

```
User Query (HTTP POST)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT GUARDRAILS (safety.py)                             │
│   • Layer A: Regex Fast-Path (0.1ms) ── Crisis/Harm/Advice  │
│   • Layer B: Guardrails AI Hub ToxicLanguage ML Classifier  │
└──────────────────────────────┬──────────────────────────────┘
                               │ ALLOW
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. TWO-STREAM HYBRID RETRIEVAL (retriever.py)                │
│   • Dense: OpenAI text-embedding-3-large (3072D, Qdrant)    │
│   • Sparse: BM25Okapi for rare drug names and posology      │
│   • Fusion: Reciprocal Rank Fusion (RRF, K=60)              │
│   • Reranker: ms-marco-MiniLM-L-6-v2 Cross-Encoder (Local)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Top Chunks + Relevance Logits
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. CONFIDENCE GATE (safety.py)                              │
│   • Is Top-1 Logit < -4.0? ──► YES ──► Refuse ("Not found") │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS (Score >= -4.0)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. GROUNDED GENERATION & VERIFICATION (assistant.py)         │
│   • Context: 1,200-char PARENT chunks with stable [chunk_id]│
│   • Model: GPT-4o (T=0.1, JSON Mode, Epistemic Amnesia)     │
│   • Output Guardrail: Citation set containment check        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
            Streamed SSE Tokens + Verified PDF Citations
```

---

## 🛠️ Repository Setup & Installation

Dependencies across projects are managed at the workspace root using [`uv`](https://astral.sh/uv).

```bash
# 1. Clone the repository
git clone https://github.com/Ahbab-Ahmed/aiengineering.git
cd aiengineering

# 2. Sync all virtual environment dependencies
uv sync

# 3. Configure environment variables for medical-rag
cp medical-rag/.env.example medical-rag/.env
# Edit medical-rag/.env and add your OPENAI_API_KEY
```

---

## 🏥 Running Medical RAG

```bash
cd medical-rag

# 1. Start Qdrant vector database container
docker compose up -d

# 2. Ingest & index all 21 MOHFW guidelines (one-time setup)
make index

# 3. Execute the 24-case automated benchmark
make evaluate

# 4. Start the interactive assistant server (port 8001)
make start
```
Open **[http://localhost:8001](http://localhost:8001)** in your browser to interact with the application.

---

## 📚 Deep Dive Technical Manual

For an encyclopedic breakdown of all mathematics, theoretical derivations (2-Poisson model, Robertson-Spärck Jones IDF, Cormack's RRF theorem, unit hypersphere geometry), clinical pharmacology scenarios, 5 systematic ablation studies, and end-to-end query traces, consult:

👉 **[medical-rag/DEEP_DIVE.md](medical-rag/DEEP_DIVE.md)** (**2,600+ lines**)
