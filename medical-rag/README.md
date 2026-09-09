# 🏥 Medical Guideline Assistant (Medical RAG)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o%20%7C%20Embeddings-green.svg)](https://openai.com/)
[![Evaluation](https://img.shields.io/badge/Benchmark-96%25%20Accuracy-success.svg)](evaluator.py)
[![Safety](https://img.shields.io/badge/Refusal%20Rate-100%25%20Safe-brightgreen.svg)](safety.py)

> An enterprise-grade, safety-critical **Retrieval-Augmented Generation (RAG)** assistant that answers clinical inquiries **strictly using official standard treatment guidelines published by the Ministry of Health & Family Welfare (MOHFW), Government of India**.

---

## 📚 Essential Documentation

| Document | Description |
|---|---|
| 📖 **[DEEP_DIVE.md](DEEP_DIVE.md)** | **Exhaustive Technical Manual (~53KB)** covering all mathematical foundations (Cosine, BM25Okapi, RRF, Cross-Encoders), Parent-Child chunking, code walkthroughs, and post-mortem lessons. |
| 🎙️ **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)** | **Saturday Showcase Playbook** featuring the 12-minute presentation agenda, live demo script (4 exact queries to type), architecture cues, benchmark analysis, and instructor Q&A defense. |

---

## 🎯 Clinical & System Problem Statement

When individuals self-diagnose using commercial conversational LLMs, severe clinical failures occur:
1. **Stochastic Hallucination**: Generates plausible but fabricated medication regimens.
2. **Knowledge Blending**: Blends unvetted blogs, forums (Reddit), and conflicting international protocols.
3. **Unauthorized Personal Prescriptions**: Diagnoses patients and prescribes dosages without medical authority.
4. **Lack of Verifiability**: Omits exact, verifiable page-level citations to trusted national guidelines.

**Our Solution**: A system grounded exclusively in **21 official MOHFW standard treatment guideline PDFs** with hard boundaries against personal advice, multi-tier guardrails, and programmatic citation verification.

---

## 🏛️ System Architecture

```
User Query (HTTP POST)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT GUARDRAILS (safety.py)                             │
│   • Layer A: Regex Fast-Path (0ms) ── Crisis/Harm/Advice/Dose│
│   • Layer B: Guardrails AI Hub ToxicLanguage ML Classifier  │
└──────────────────────────────┬──────────────────────────────┘
                               │ ALLOW
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. QUERY NORMALIZATION (assistant.py)                       │
│   • Expand medical abbreviations (BP → blood pressure)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TWO-STAGE HYBRID RETRIEVAL (retriever.py)                │
│   • Dense: OpenAI text-embedding-3-large (3072 dims, Qdrant)│
│   • Sparse: BM25Okapi for exact clinical drug terminology   │
│   • Fusion: Reciprocal Rank Fusion (RRF, k=60)              │
│   • Reranker: ms-marco-MiniLM-L-6-v2 Cross-Encoder (Local)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Top 5 Chunks + Logit Scores
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. CONFIDENCE GATE (safety.py)                              │
│   • Is Top-1 Score < -4.0? ──► YES ──► Refuse ("Not found") │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS (Score >= -4.0)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. GROUNDED GENERATION (assistant.py)                       │
│   • Context: 1200-char PARENT chunks with [chunk_id]        │
│   • Model: GPT-4o (temperature=0.1, JSON Mode)              │
│   • Mandatory inline citation: [chunk_xxxxxx]               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. OUTPUT GUARDRAIL & CITATION VERIFICATION (safety.py)     │
│   • Prescriptive phrasing scan (regex + Detoxify ML)        │
│   • Programmatic check: Are cited IDs in retrieved context? │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
            Streamed SSE Answer + Page Citations + Disclaimer
```

---

## 🏆 Automated Benchmark & Evaluation Results

Validated against a rigorous **24-case benchmark** across 6 distinct categories in [`evaluator.py`](evaluator.py):

| Metric | Result | Target | Notes |
|---|:---:|:---:|---|
| **Overall Accuracy** | **96% (23/24)** | $>90\%$ | 23 of 24 cases handled with 100% precision |
| **Refusal Accuracy** | **100% (18/18)** | $100\%$ | **Zero unsafe or out-of-domain queries leaked through** |
| **Answer Accuracy** | **83% (5/6)** | $>80\%$ | 5 of 6 answerable medical guideline queries correctly answered |
| **Citation Rate** | **67% (4/6)** | $>60\%$ | Grounded answers include verified inline `[chunk_id]` tags |

### Category Breakdown
- **Answerable Guidelines (6 cases)**: 5 passed (Tuberculosis, Hypertension, Paediatric Diarrhoea, Snakebite, Dengue). Case 04 (ORS) failed due to multi-column table layout in PDF.
- **Unanswerable / Off-Topic (4 cases)**: 4 passed (FIFA World Cup, Chocolate Cake, AI Frameworks, Blockchain). Score $< -7.5$, intercepted by Confidence Gate.
- **Crisis / Emergency (4 cases)**: 4 passed (Heart attack, Overdose, Stroke, Dying friend). Intercepted at Layer A in 0ms.
- **Personal Advice (5 cases)**: 5 passed (Fever self-diagnosis, Blood pressure advice, Paediatric medication).
- **Dosing Guidance (3 cases)**: 3 passed (Malaria pills, Rifampicin adult dosage, Hourly ORS rate).
- **Toxic / Malicious Queries (2 cases)**: 2 passed. Blocked by Guardrails AI Hub `ToxicLanguage` ML validator.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- macOS or Linux
- Python 3.12+ and [`uv`](https://astral.sh/uv)
- Docker Desktop (for Qdrant)
- OpenAI API Key

### 2. Environment Configuration
Create `.env` inside `medical-rag/`:
```bash
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://localhost:6333
```

### 3. Start Infrastructure & Ingest Documents
```bash
# Start Qdrant vector database container
docker compose up -d

# Run one-time indexing pipeline (extracts, chunks, embeds, and builds BM25)
make index
```

### 4. Launch the Web Assistant
```bash
make start
```
Open **[http://localhost:8001](http://localhost:8001)** in your browser.

---

## 🧪 Testing & Daily Commands

```bash
make start     # Launch FastAPI server on port 8001
make stop      # Terminate background server process
make evaluate  # Execute the 24-case automated evaluation suite
make status    # Check container and server health status
```

---

## 📦 Project Structure

```
medical-rag/
├── DEEP_DIVE.md          # Exhaustive technical, mathematical, and architectural manual
├── PRESENTATION_GUIDE.md # Saturday cohort showcase script and Q&A defense playbook
├── README.md             # This document
├── Makefile              # Automation shortcuts (start, stop, evaluate, index)
├── docker-compose.yml    # Qdrant vector database container configuration
├── app.py                # FastAPI server, Server-Sent Events (SSE) streaming, endpoints
├── assistant.py          # Grounded prompt engineering, query rewriting, JSON generation
├── config.py             # Centralized parameters, thresholds, and regex patterns
├── indexing.py           # PyMuPDF extraction, parent-child chunking, vector ingestion
├── retriever.py          # Hybrid search (Dense + BM25) + RRF + Cross-Encoder reranking
├── safety.py             # Multi-tier input/output guardrails & confidence gate
├── session_manager.py    # SQLite state management, conversation logs, feedback ratings
├── evaluator.py          # Automated 24-case test benchmark and scoring harness
├── static/               # Responsive single-page web application (HTML/CSS/JS)
└── data/
    ├── guidelines/       # 21 Official MOHFW treatment guideline PDFs
    ├── manifest.json     # Ingestion audit trail with timestamps and chunk hashes
    └── sessions.db       # Local conversation history database (git-ignored)
```

---

## ⚖️ Legal & Clinical Disclaimer

*This assistant is designed strictly as an educational and informational reference tool grounded in official Ministry of Health and Family Welfare (MOHFW) guidelines. It does not provide personalized medical advice, individual clinical diagnosis, or customized pharmaceutical dosing. Always consult a licensed healthcare professional for medical decisions.*
