"""
indexing.py — One-time ingestion pipeline for the Medical Guideline Assistant.

What this does (step by step):
  1. Load every PDF from data/guidelines/
  2. Extract text page by page
  3. Split into small "child" chunks (for precise retrieval)
     and larger "parent" chunks (for richer LLM context)
  4. Embed each child chunk using OpenAI embeddings
  5. Store embeddings + metadata in Qdrant
  6. Build a BM25 index for keyword search (hybrid retrieval)
  7. Write a manifest.json so we know what was indexed and when

Run once:  uv run python indexing.py
"""

import sys
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import pymupdf as fitz                 # PyMuPDF — reads PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rank_bm25 import BM25Okapi

from config import (
    GUIDELINES_DIR, DATA_DIR, QDRANT_URL, COLLECTION_NAME,
    EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, PARENT_SIZE,
)


# ── Helper: extract text from a PDF ────────────────────────────────

def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns a list of dicts:
      { "page": int, "text": str, "source": filename }

    Why page by page?
    Preserves page numbers for citations — users can check
    exactly which page of which guideline the answer came from.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:   # skip blank/image-only pages
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": pdf_path.name,
            })
    doc.close()
    return pages


# ── Helper: create child + parent chunks ───────────────────────────

def make_chunks(pages: list[dict], doc_id: str) -> list[dict]:
    """
    Split page text into overlapping child chunks and wider parent chunks.

    Child chunks  → used for retrieval (small = precise match)
    Parent chunks → stored alongside; sent to LLM (large = more context)

    The chunk_id is a stable hash so re-indexing the same file
    produces the same IDs (useful for citation verification).
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = []
    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]
        source   = page_info["source"]

        # Build parent chunks first (for richer LLM context)
        parent_texts = parent_splitter.split_text(text)

        for parent_idx, parent_text in enumerate(parent_texts):
            # Each parent is split into smaller child chunks
            child_texts = child_splitter.split_text(parent_text)

            for child_idx, child_text in enumerate(child_texts):
                # Stable hash ID — same text always → same ID
                raw_id = f"{doc_id}:p{page_num}:par{parent_idx}:ch{child_idx}"
                chunk_id = "chunk_" + hashlib.md5(raw_id.encode()).hexdigest()[:10]

                chunks.append({
                    "chunk_id":      chunk_id,
                    "text":          child_text,        # what gets embedded
                    "parent_text":   parent_text,       # what LLM sees
                    "metadata": {
                        "chunk_id":    chunk_id,
                        "doc_id":      doc_id,
                        "source":      source,
                        "page":        page_num,
                        "parent_idx":  parent_idx,
                        "child_idx":   child_idx,
                    },
                })
    return chunks


# ── Main indexing pipeline ──────────────────────────────────────────

def index_guidelines():
    pdf_files = sorted(GUIDELINES_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in {GUIDELINES_DIR}")
        print("   Download the MOHFW guidelines and place them there.")
        return

    print(f"\n📁 Found {len(pdf_files)} PDFs:")
    for f in pdf_files:
        print(f"   • {f.name}")

    # ── Step 1: Qdrant setup ───────────────────────────────────────
    print("\n🔌 Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Delete old collection if it exists (clean re-index)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"🗑️  Deleting old collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    # Create fresh collection
    # 3072 = dimension of text-embedding-3-large
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # ── Step 2: Process each PDF ───────────────────────────────────
    all_chunks = []
    manifest_entries = []

    for pdf_path in pdf_files:
        print(f"\n📖 Processing: {pdf_path.name}")
        doc_id = pdf_path.stem.lower().replace(" ", "_")

        pages  = extract_pages(pdf_path)
        chunks = make_chunks(pages, doc_id)

        print(f"   Pages: {len(pages)}  |  Chunks: {len(chunks)}")
        all_chunks.extend(chunks)

        manifest_entries.append({
            "file":       pdf_path.name,
            "doc_id":     doc_id,
            "pages":      len(pages),
            "chunks":     len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        })

    # ── Step 3: Embed and store in Qdrant ─────────────────────────
    print(f"\n🔢 Embedding {len(all_chunks)} chunks (this takes a few minutes)...")

    # Prepare data for LangChain's Qdrant wrapper
    texts     = [c["text"]     for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # LangChain handles batching + upload automatically
    vector_store = QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print("✅ Chunks embedded and stored in Qdrant")

    # ── Step 4: Build BM25 index (keyword search) ─────────────────
    # BM25 works on exact token matches — complements dense retrieval.
    # Together they form "hybrid search" for better recall.
    print("\n📑 Building BM25 keyword index...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    bm25_cache = {
        "bm25":   bm25,
        "texts":  texts,
        "metas":  metadatas,
    }
    cache_path = DATA_DIR / "bm25_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(bm25_cache, f)
    print(f"✅ BM25 index saved → {cache_path.name}")

    # ── Step 5: Write manifest ────────────────────────────────────
    # Tracks what was indexed and when — useful for versioning.
    manifest = {
        "indexed_at":  datetime.now(timezone.utc).isoformat(),
        "collection":  COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "chunk_size":  CHUNK_SIZE,
        "total_chunks": len(all_chunks),
        "documents":   manifest_entries,
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✅ Manifest saved → {manifest_path.name}")

    # ── Done ──────────────────────────────────────────────────────
    print(f"\n🎉 Indexing complete!")
    print(f"   Total chunks indexed: {len(all_chunks)}")
    print(f"   Collection: {COLLECTION_NAME} @ {QDRANT_URL}")
    print("\n   Next step: run  make start  to launch the assistant.\n")


if __name__ == "__main__":
    index_guidelines()
