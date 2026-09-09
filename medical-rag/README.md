# 🏥 Medical Guideline Assistant

> A RAG-based assistant that answers health questions **strictly from official MOHFW government guidelines**, with hard safety boundaries against personalised medical advice.

---

## What it does

| Feature | Details |
|---------|---------|
| **Grounded answers** | Every claim is cited with a `[chunk_id]` pointing to a specific page of a specific guideline |
| **Input guardrails** | Personal advice, dosing, crisis queries are blocked *before* any retrieval |
| **Confidence gate** | If retrieval confidence is too low, refuses rather than hallucinating |
| **Output safety** | Generated text is scanned for unsafe language before sending to the user |
| **Disclaimer** | Appended to every response automatically |

---

## Setup (one time)

### 1. Download the PDFs
Get the MOHFW guidelines from the shared Google Drive and place them in:
```
medical-rag/data/guidelines/
```

### 2. Start Qdrant
```bash
cd medical-rag
docker compose up -d
```

### 3. Index the PDFs
```bash
make index
```
This extracts text, creates chunks, embeds them, and stores everything in Qdrant + BM25. Takes a few minutes.

### 4. Start the app
```bash
make start
```
Open **http://localhost:8001**

---

## Daily use
```bash
make start    # start (port 8001)
make stop     # stop everything
make evaluate # run the benchmark suite
make status   # check if services are up
```

---

## How it works (the full pipeline)

```
User query
  ↓
[1] Input Guardrail (guardrails.py)
    ├── Crisis? → refer to emergency services
    ├── Personal advice? → refuse
    └── Dosing? → refuse
  ↓
[2] Query Rewriting (assistant.py)
    → Expand abbreviations (BP→blood pressure, TB→tuberculosis)
  ↓
[3] Hybrid Retrieval (retriever.py)
    ├── Dense: Qdrant vector search (semantic meaning)
    └── BM25: keyword search (exact terms)
    → Fuse via Reciprocal Rank Fusion
    → Cross-encoder rerank → top-5 chunks + confidence scores
  ↓
[4] Confidence Gate (guardrails.py)
    → Top score < threshold → refuse with "not in guidelines"
  ↓
[5] Grounded Generation (assistant.py)
    → LLM sees only retrieved chunks
    → Must cite [chunk_id] for every claim
    → Structured JSON output: {answer, citations}
  ↓
[6] Output Guardrail (guardrails.py)
    ├── Scan for personalised/diagnostic language
    └── Verify all cited chunk_ids exist in retrieved context
  ↓
Response + Disclaimer
```

---

## Evaluation

```bash
make evaluate
```

Tests three query categories:
- **Answerable** — should return grounded answer with citations
- **Unanswerable** — should be refused (low confidence)
- **Personal/advice** — should be blocked at input guardrail

Metrics tracked: refusal accuracy, answer accuracy, citation rate.
Results saved to `data/evaluation_results.json`.

---

## Project files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server |
| `config.py` | All settings and thresholds |
| `indexing.py` | PDF → chunks → Qdrant (run once) |
| `retriever.py` | Hybrid retrieval + reranking |
| `guardrails.py` | Input + output safety |
| `assistant.py` | LLM generation with citations |
| `session_manager.py` | SQLite session storage |
| `evaluator.py` | Benchmark suite |
| `data/guidelines/` | Drop PDFs here |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Collection not found` | Run `make index` first |
| Loading stuck | Check `.env` has `OPENAI_API_KEY` |
| Port 8001 in use | Run `make stop` then `make start` |
| No PDFs indexed | Place PDFs in `data/guidelines/` and run `make index` |
