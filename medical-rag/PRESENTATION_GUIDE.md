# 🎙️ Saturday Class Showcase Playbook: Medical Guideline Assistant

> **A Complete Walkthrough & Presentation Guide for Ahbab Ahmed**
> 
> *Showcasing the Medical RAG system in the AI Engineering cohort.*

---

## ⏱️ The 12-Minute Presentation Agenda

```
┌───────────────┬─────────────────────────────────────────────────────────────┐
│ Timing        │ Section                                                     │
├───────────────┼─────────────────────────────────────────────────────────────┤
│ 0:00 - 2:00   │ 1. The Hook & Clinical Problem Statement                    │
│ 2:00 - 5:00   │ 2. Live Interactive Demo (4 Strategic Queries)              │
│ 5:00 - 8:30   │ 3. Architecture Deep Dive (Two-Stage Hybrid + Guardrails)   │
│ 8:30 - 10:30  │ 4. Benchmark Verification & Engineering Trade-Offs (96%)    │
│ 10:30 - 12:00 │ 5. Q&A Defense & Future Vision                              │
└───────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 1. The Opening Hook (0:00 - 2:00)

### What to Say:
> *"Hi everyone. Today I'm excited to showcase the **Medical Guideline Assistant**.*
> 
> *When people feel sick, their first instinct is to ask ChatGPT or search the web. But in clinical medicine, standard conversational LLMs have catastrophic failure modes:*
> 1. *They **hallucinate** plausible-sounding drug regimens.*
> 2. *They **blend** unvetted internet forums, Reddit threads, and conflicting international protocols.*
> 3. *They give **unauthorized personal diagnoses and dosages**, opening up severe clinical and legal liability.*
> 
> *Our goal was to build a **strictly grounded, safety-first RAG system** restricted exclusively to **21 official standard treatment guidelines published by India's Ministry of Health and Family Welfare (MOHFW)**.*
> 
> *Instead of treating RAG as a simple wrapper around an LLM, we engineered a production-grade pipeline featuring:*
> - *Hierarchical Parent-Child chunking,*
> - *Hybrid retrieval combining Dense semantic vectors with BM25 keyword matching,*
> - *Reciprocal Rank Fusion (RRF),*
> - *Cross-encoder reranking,*
> - *A calibrated confidence gate, and*
> - *Multi-tier input/output safety guardrails.*
> 
> *Let me demonstrate how it works live."*

---

## 2. The Live Interactive Demo (2:00 - 5:00)

Have the web interface open in your browser at **`http://localhost:8001`**. Type these 4 queries in this exact sequence:

### Query 1: The Grounded Answer (Hybrid Retrieval + Page Citations)
👉 **Type:**
```
What are the standard treatment guidelines for tuberculosis?
```

- **What happens on screen:**
  - The answer streams smoothly via Server-Sent Events (SSE).
  - Sentences include inline citations like `[chunk_08a3d1]`.
  - The right-hand sidebar populates the **Citations Panel** displaying the exact document (`tb_standards.pdf`) and **exact page numbers**.
  - A mandatory MOHFW disclaimer is automatically appended.
- **What to say:**
  > *"Notice that the model doesn't just answer; it grounds every single claim with an inline citation tag. The citations panel extracts the metadata, pointing the clinician to the exact PDF and physical page number in the MOHFW standard treatment guidelines."*

---

### Query 2: The 0ms Regex Fast-Path (Zero-Cost Safety Intercept)
👉 **Type:**
```
I have a severe fever and headache, should I take ibuprofen?
```

- **What happens on screen:**
  - An instant refusal appears without waiting for an LLM generation:
    > *"I am not able to provide personalized medical advice or treatment recommendations..."*
- **What to say:**
  > *"Notice how fast that answered—under 1 millisecond. This query didn't invoke OpenAI, didn't run vector search, and cost zero tokens. Our Layer A regex fast-path in `safety.py` intercepted the personalized advice pattern immediately, protecting the user from self-diagnosis while saving compute."*

---

### Query 3: The Cross-Encoder Confidence Gate (Anti-Hallucination)
👉 **Type:**
```
Who won the 2022 FIFA World Cup?
```

- **What happens on screen:**
  - The model does **not** answer with Argentina, even though GPT-4o obviously knows the answer.
  - Instead, it responds:
    > *"This information is not available in the MOHFW guidelines I have access to."*
- **What to say:**
  > *"In standard RAG, vector search will find the 5 'closest' chunks (perhaps sports injury or trauma guidelines) and GPT-4o will gladly answer. In our system, the cross-encoder scored the top retrieved chunk at `-7.5`, which is well below our calibrated confidence threshold of `-4.0`. The gate fired and refused before generation ever began."*

---

### Query 4: The Crisis Emergency Protocol
👉 **Type:**
```
I think I am having a heart attack right now
```

- **What happens on screen:**
  - Instant redirection to national emergency lifelines:
    > *"If you are experiencing a medical emergency, please immediately call emergency services (112 / 911 / 999)..."*
- **What to say:**
  > *"In a true medical emergency, conversational latency can cost lives. The emergency guardrail immediately overrides the standard pipeline and provides direct emergency hotline instructions."*

---

## 3. Architecture Deep Dive (5:00 - 8:30)

Walk through the system architecture using this flow:

```
User Query
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
│ 2. QUERY EXPANSION (assistant.py)                           │
│   • Expand medical abbreviations (BP → blood pressure)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TWO-STAGE HYBRID RETRIEVAL (retriever.py)                │
│   • Dense: OpenAI text-embedding-3-large + Qdrant (HNSW)    │
│   • Sparse: BM25Okapi for exact clinical drug terminology   │
│   • Fusion: Reciprocal Rank Fusion (RRF with k=60)          │
│   • Reranker: ms-marco-MiniLM-L-6-v2 Cross-Encoder          │
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
│   • Model: GPT-4o with temperature=0.1                      │
│   • Output Format: Native JSON mode                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. OUTPUT GUARDRAIL & CITATION VERIFICATION (safety.py)     │
│   • Scan for prescriptive language                          │
│   • Programmatic check: Did LLM hallucinate any chunk_id?   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                   Streamed Answer + Disclaimer
```

### Key Technical Concepts to Highlight:

1. **Parent-Child Chunking**:
   - Small **300-character child chunks** for vector search precision.
   - Large **1200-character parent chunks** passed to the LLM for clinical context.
2. **Why Hybrid Search (Dense + BM25)**:
   - Dense embeddings handle medical concepts and paraphrases.
   - BM25 handles exact drug names (*rifampicin*, *pyrazinamide*) and acronyms (*DOTS*, *ORS*).
3. **Reciprocal Rank Fusion (RRF)**:
   $$\text{Score}(d) = \sum_{m \in M} \frac{1}{60 + \text{rank}_m(d)}$$
   - Rank-based fusion prevents BM25's unbounded scores from dominating cosine similarities.
4. **Bi-Encoder vs. Cross-Encoder**:
   - Bi-encoder filters 14,800+ chunks down to 16 candidates in $O(1)$ vector distance.
   - Cross-encoder runs full pairwise transformer cross-attention locally in ~80ms to score true relevance.

---

## 4. Benchmark Verification & Engineering Trade-Offs (8:30 - 10:30)

Switch to your terminal or show the benchmark results:

```bash
uv run python evaluator.py
```

### Benchmark Results Table:
```
================================================================================
🏥 MEDICAL GUIDELINE RAG — BENCHMARK RESULTS
================================================================================
Total Test Cases:       24
Overall Accuracy:       96%  (23/24)
Refusal Accuracy:      100%  (18/18)   ◄ ZERO UNSAFE QUERIES SLIPPED THROUGH
Answer Accuracy:        83%  (5/6)
Citation Rate:          67%  (4/6 answered queries had complete citations)
================================================================================
```

### The Engineering Story: Calibrating the Confidence Threshold (-4.0)
*This is the story instructors love because it demonstrates real AI engineering judgment:*

1. *"We initially calibrated our confidence threshold at `0.0`."*
2. *"However, Case 03 (paediatric diarrhoea) and Case 04 (ORS) were falsely refused. When we inspected the raw PDFs, we found that ORS formulation is presented in a **multi-column visual table**."*
3. *"PyMuPDF extracts tables as fragmented text strings, which causes the cross-encoder to assign lower relevance logits (`-3.7` and `-5.4`)."*
4. *"We analyzed the score spectrum of completely off-topic queries: FIFA World Cup scored `-7.5`, chocolate cake scored `-9.2`, and blockchain scored `-11.0`."*
5. *"By adjusting the threshold to **`-4.0`**, paediatric diarrhoea passed (`-3.7`), while maintaining a safe **3.5-point margin** against off-topic queries."*
6. *"We deliberately chose NOT to lower the threshold to `-6.0` to force ORS to pass, because that would have left only a 1.5-point margin against FIFA. In medical systems, **safety margin > edge-case recall**."*

---

## 5. Anticipated Instructor Questions & Defense (10:30 - 12:00)

### Q1: "Why did you use RRF instead of linear interpolation ($\alpha \cdot S_{\text{dense}} + (1-\alpha) \cdot S_{\text{sparse}}$)?"
> **Your Answer:** *"BM25 scores are unbounded $[0, \infty)$ and vary wildly depending on query length and term rarity, while cosine similarity is bounded $[0, 1]$. Normalizing BM25 dynamically per query is brittle and sensitive to outlier tokens. RRF operates purely on ordinal ranks, making it completely invariant to score distributions."*

### Q2: "What prevents the model from hallucinating a citation like `[chunk_fake99]`?"
> **Your Answer:** *"Our Layer D output guardrail in `safety.py`. After generation completes, `check_output()` programmatically parses every cited `[chunk_xxxxxx]` tag and validates that it belongs to the `retrieved_ids` set passed to the prompt. If any cited ID is missing, the response is flagged as a hallucinated citation and redacted."*

### Q3: "Why use PyMuPDF instead of OCR or multimodal vision models?"
> **Your Answer:** *"PyMuPDF is fast, lightweight, and preserves exact physical page numbers, which is essential for our citation traceability. For a v2 production release, we would add a vision model or table-transformer pre-processor specifically to serialize multi-column clinical tables into clean Markdown tables before chunking."*

### Q4: "How does this scale if we ingest 1,000 guideline PDFs?"
> **Your Answer:** *"The architecture scales seamlessly:*
> - *Qdrant uses HNSW indexing, keeping vector search time at $O(\log N)$.*
> - *BM25 token indices can be backed by Tantivy or Elasticsearch for distributed inverted indexing.*
> - *The cross-encoder reranker only ever processes the top 16 candidates, meaning reranking latency remains constant (~80ms) regardless of whether the corpus has 20 PDFs or 20,000 PDFs."*

---

## 📋 Saturday Morning Pre-Flight Checklist

- [ ] Docker container is running: `docker compose up -d`
- [ ] Test Qdrant connectivity: `curl http://localhost:6333/collections`
- [ ] Start server: `cd medical-rag && uv run uvicorn app:app --app-dir . --port 8001`
- [ ] Browser open to **`http://localhost:8001`**
- [ ] Terminal split with `evaluator.py` ready to execute
- [ ] GitHub repository open at **`https://github.com/Ahbab-Ahmed/aiengineering`**
