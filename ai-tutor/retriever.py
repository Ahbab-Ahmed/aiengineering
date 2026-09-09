import sys
import json
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from openai import OpenAI

from config import QDRANT_URL, COLLECTION_NAME, EMBED_MODEL, DATA_DIR, BOOK_TITLE

BM25_CACHE_PATH = DATA_DIR / "bm25_cache.pkl"


class HybridRetriever:
    """
    Five-stage retrieval pipeline:
      1. HyDE  — embed a hypothetical answer instead of raw query
      2. Multi-query — rephrase query 3 ways, retrieve for each
      3. BM25 keyword search (persisted to disk, loaded instantly on restart)
      4. Reciprocal Rank Fusion of all result lists
      5. Cross-encoder reranking → return parent_content for rich LLM context
    """

    def __init__(self):
        self.llm = OpenAI()
        print("🔌 Connecting to Qdrant...")
        self.embedding_model = OpenAIEmbeddings(model=EMBED_MODEL)
        self.vector_db = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding_model,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
        )
        self.qdrant_client = QdrantClient(url=QDRANT_URL)

        print("🤖 Loading cross-encoder reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

        self._build_bm25_index()
        print("✅ HybridRetriever ready.")

    # ── BM25 (with disk cache) ───────────────────────────────────────

    def _build_bm25_index(self):
        """Load BM25 from disk if cache matches current Qdrant count, else rebuild."""
        current_count = self.qdrant_client.count(COLLECTION_NAME).count

        if BM25_CACHE_PATH.exists():
            try:
                with open(BM25_CACHE_PATH, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("doc_count") == current_count:
                    print(f"📂 BM25 cache hit — {current_count} docs, loading instantly...")
                    self.docs: List[Dict[str, Any]] = cached["docs"]
                    self.bm25: BM25Okapi = cached["bm25"]
                    return
            except Exception:
                pass  # corrupt cache, rebuild

        print(f"🔄 Building BM25 index from Qdrant ({current_count} docs)...")
        self.docs = []
        offset = None
        while True:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                self.docs.append({
                    "text": payload.get("page_content", ""),
                    "metadata": payload.get("metadata", {}),
                })
            if next_offset is None:
                break
            offset = next_offset

        tokenized = [doc["text"].lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized)

        # Persist to disk
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"docs": self.docs, "bm25": self.bm25, "doc_count": current_count}, f)
        print(f"✅ BM25 index built and cached — {len(self.docs)} documents.")

    # ── Stage 1: HyDE ───────────────────────────────────────────────

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Generate a fake textbook answer to use as the embedding query."""
        try:
            resp = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an AI engineering expert. Write one detailed paragraph answering "
                            f"the question below, in the style of the textbook '{BOOK_TITLE}'. "
                            "Be technically precise. No preamble."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=280,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return query  # fallback to original query

    # ── Stage 2: Multi-query ────────────────────────────────────────

    def _generate_alt_queries(self, query: str) -> List[str]:
        """Rephrase the query 3 ways to improve recall."""
        try:
            resp = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate 3 alternative phrasings of this question to improve document retrieval. "
                            "Return only the 3 questions, one per line, no numbering or explanation."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            alt = [q.strip() for q in resp.choices[0].message.content.strip().split("\n") if q.strip()]
            return alt[:3]
        except Exception:
            return []

    # ── Stage 3: BM25 ───────────────────────────────────────────────

    def _bm25_search(self, query: str, k: int = 20) -> List[Dict]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"doc": self.docs[i], "rank": rank} for rank, i in enumerate(top_idx)]

    # ── Stage 3b: Dense MMR ─────────────────────────────────────────

    def _vector_search(self, query: str, k: int = 12) -> List[Dict]:
        langchain_docs = self.vector_db.max_marginal_relevance_search(
            query=query, k=k, fetch_k=30, lambda_mult=0.55
        )
        return [
            {"doc": {"text": d.page_content, "metadata": d.metadata}, "rank": i}
            for i, d in enumerate(langchain_docs)
        ]

    # ── Stage 4: RRF ────────────────────────────────────────────────

    def _rrf(self, *result_lists, k: int = 60) -> List[Dict]:
        scores: Dict[str, float] = {}
        docs_by_text: Dict[str, Dict] = {}
        for results in result_lists:
            for r in results:
                text = r["doc"]["text"]
                scores[text] = scores.get(text, 0.0) + 1.0 / (k + r["rank"] + 1)
                docs_by_text[text] = r["doc"]
        return [docs_by_text[t] for t in sorted(scores, key=lambda t: -scores[t])]

    # ── Stage 5: Cross-encoder rerank ───────────────────────────────

    def _rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        if not docs:
            return []
        pairs = [(query, doc["text"]) for doc in docs]
        rerank_scores = self.reranker.predict(pairs)
        return [doc for doc, _ in sorted(zip(docs, rerank_scores), key=lambda x: -x[1])[:top_k]]

    # ── Public API ──────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        """
        Full pipeline:
        HyDE + multi-query vector search + BM25 → RRF → cross-encoder → parent_content
        """
        # Stage 1: HyDE
        hyde_doc = self._generate_hypothetical_doc(query)

        # Stage 2: Multi-query
        alt_queries = self._generate_alt_queries(query)

        # Stage 3: BM25 on original query
        bm25_results = self._bm25_search(query, k=20)

        # Stage 3b: Dense search — HyDE + original + alt queries
        all_vector: List[Dict] = []
        seen_texts: set = set()
        for q in [hyde_doc, query] + alt_queries:
            for r in self._vector_search(q, k=10):
                text = r["doc"]["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    all_vector.append({"doc": r["doc"], "rank": len(all_vector)})

        # Stage 4: RRF
        fused = self._rrf(bm25_results, all_vector)[:30]

        # Stage 5: Rerank
        reranked = self._rerank(query, fused, top_k)

        # Return parent_content (large context) instead of child text
        results = []
        seen_parents: set = set()
        for doc in reranked:
            parent = doc["metadata"].get("parent_content") or doc["text"]
            if parent not in seen_parents:
                seen_parents.add(parent)
                results.append({"text": parent, "metadata": doc["metadata"]})

        return results[:top_k]
