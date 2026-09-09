"""
retriever.py — Hybrid retrieval with confidence scoring.

Two-stage pipeline (as discussed in the notes):
  Stage 1 — Bi-encoder (fast, high recall)
    • Dense vector search via Qdrant  ← semantic meaning
    • BM25 keyword search             ← exact term matching
    • Combine both via Reciprocal Rank Fusion (RRF)
    → Returns top-K candidates

  Stage 2 — Cross-encoder (slower, high precision)
    • Scores every (query, chunk) pair jointly
    • Produces a relevance score for each candidate
    → Returns top rerank_k results, ordered by score

The top-1 score is exposed for the confidence gate in guardrails.py.
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

from config import (
    DATA_DIR, QDRANT_URL, COLLECTION_NAME,
    EMBED_MODEL, RERANK_MODEL, TOP_K, RERANK_TOP_K,
)


class MedicalRetriever:
    """
    Hybrid retriever for medical guideline chunks.

    Usage:
        retriever = MedicalRetriever()
        results = retriever.retrieve("What are TB treatment guidelines?", top_k=5)

    Each result dict contains:
        text        — child chunk text (what was embedded)
        parent_text — richer parent chunk (what LLM will read)
        score       — cross-encoder relevance score
        metadata    — chunk_id, source, page, doc_id, etc.
    """

    def __init__(self):
        print("⚙️  Loading retriever...")

        # Dense embeddings (same model used during indexing)
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

        # Qdrant vector store
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
        )

        # Cross-encoder reranker (runs locally, no API cost)
        # ms-marco is trained on real search relevance — works well for
        # (question, document passage) pairs.
        print("⚙️  Loading cross-encoder reranker (first time may download model)...")
        self.reranker = CrossEncoder(RERANK_MODEL)

        # BM25 keyword index (built during indexing.py)
        bm25_path = DATA_DIR / "bm25_cache.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                cache = pickle.load(f)
            self.bm25   = cache["bm25"]
            self.texts  = cache["texts"]
            self.metas  = cache["metas"]
            self.bm25_ready = True
            print("✅ BM25 index loaded")
        else:
            self.bm25_ready = False
            print("⚠️  BM25 cache not found — run indexing.py first. Falling back to dense only.")

        print("✅ Retriever ready")

    # ── Dense retrieval ────────────────────────────────────────────

    def _dense_retrieve(self, query: str, k: int) -> list[dict]:
        """
        Search Qdrant using the query embedding.
        Returns up to k chunks as dicts with text + metadata.
        """
        docs = self.vector_store.similarity_search(query, k=k)
        results = []
        for doc in docs:
            results.append({
                "text":        doc.page_content,
                "parent_text": doc.metadata.get("parent_text", doc.page_content),
                "metadata":    doc.metadata,
                "source":      "dense",
            })
        return results

    # ── BM25 retrieval ─────────────────────────────────────────────

    def _bm25_retrieve(self, query: str, k: int) -> list[dict]:
        """
        Keyword search using BM25.
        Returns up to k chunks ranked by token overlap with the query.
        """
        if not self.bm25_ready:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Get indices of top-k scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:   # only include chunks with non-zero match
                results.append({
                    "text":        self.texts[idx],
                    "parent_text": self.metas[idx].get("parent_text", self.texts[idx]),
                    "metadata":    self.metas[idx],
                    "source":      "bm25",
                    "bm25_score":  float(scores[idx]),
                })
        return results

    # ── Reciprocal Rank Fusion ─────────────────────────────────────

    def _fuse(self, dense_results: list[dict], bm25_results: list[dict]) -> list[dict]:
        """
        Combine dense and BM25 results using Reciprocal Rank Fusion (RRF).

        RRF idea: instead of using raw scores (which are on different scales),
        use each result's *rank* in its list. A result ranked #1 in both lists
        gets a very high combined score.

        Formula: score(d) = Σ 1 / (k + rank(d))   where k=60 is a smoothing constant
        """
        K = 60
        scores: dict[str, float] = {}
        chunks: dict[str, dict]  = {}

        def add_list(results):
            for rank, item in enumerate(results):
                cid = item["metadata"].get("chunk_id", str(rank))
                scores[cid] = scores.get(cid, 0) + 1.0 / (K + rank + 1)
                chunks[cid] = item

        add_list(dense_results)
        add_list(bm25_results)

        # Sort by fused score descending
        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [chunks[cid] for cid in sorted_ids]

    # ── Cross-encoder reranking ────────────────────────────────────

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Score each candidate against the query using the cross-encoder.

        The cross-encoder reads both query and chunk together (unlike the
        bi-encoder which embeds them separately). This gives a more accurate
        relevance score but is slower — so we only run it on the small
        candidate set from stage 1.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        # Attach scores to each candidate
        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)

        # Sort by score descending, return top_k
        ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
        return ranked[:top_k]

    # ── Public interface ───────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
        """
        Full two-stage retrieval pipeline.

        Returns reranked chunks, each with a 'score' field.
        The caller (app.py) uses results[0]['score'] for the confidence gate.
        """
        # Stage 1: hybrid candidate retrieval
        dense   = self._dense_retrieve(query, k=TOP_K)
        bm25    = self._bm25_retrieve(query,  k=TOP_K)
        fused   = self._fuse(dense, bm25)

        # Stage 2: cross-encoder reranking
        ranked  = self._rerank(query, fused, top_k=top_k)

        return ranked
