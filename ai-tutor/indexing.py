import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import re
import fitz  # pymupdf
from collections import Counter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from config import QDRANT_URL, COLLECTION_NAME, EMBED_MODEL, DATA_DIR

load_dotenv()

PDF_PATH = DATA_DIR / "ai_engineering.pdf"
BM25_CACHE_PATH = DATA_DIR / "bm25_cache.pkl"

# Parent chunks → what the LLM reads (rich context)
PARENT_CHUNK_SIZE = 1400
PARENT_CHUNK_OVERLAP = 150

# Child chunks → what gets embedded (precise retrieval)
CHILD_CHUNK_SIZE = 380
CHILD_CHUNK_OVERLAP = 40

SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]


# ── PDF Extraction ─────────────────────────────────────────────────

def detect_body_font_size(doc: fitz.Document) -> float:
    """Find the most common font size (= body text size) across the document."""
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if len(span["text"].strip()) > 5:
                        sizes.append(round(span["size"], 1))
    return Counter(sizes).most_common(1)[0][0] if sizes else 10.0


def extract_structured_pages(pdf_path: Path) -> list:
    """
    Extract text from each page using pymupdf.
    Detects chapter and section headings from font sizes.
    Returns list of dicts with text + full metadata.
    """
    doc = fitz.open(str(pdf_path))
    body_size = detect_body_font_size(doc)
    chapter_threshold = body_size * 1.5
    section_threshold = body_size * 1.12

    pages = []
    current_chapter_num = 0
    current_chapter_title = "Introduction"
    current_section_title = ""

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        text_parts = []

        for block in blocks:
            if "lines" not in block:
                continue

            block_lines = []
            max_size = 0

            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue
                for span in line["spans"]:
                    if span["size"] > max_size:
                        max_size = span["size"]
                block_lines.append(line_text)

            block_text = " ".join(block_lines).strip()
            if not block_text:
                continue

            # Skip bare page numbers
            if re.match(r"^\d{1,4}$", block_text):
                continue

            # Detect chapter heading
            ch_match = re.match(r"(chapter\s+\d+)[:\.\s]*(.*)", block_text, re.I)
            if max_size >= chapter_threshold and ch_match:
                current_chapter_num += 1
                current_chapter_title = block_text.strip()
                current_section_title = ""
                continue  # don't include heading in text body

            # Detect section heading (larger than body, short, no terminal period)
            if (max_size >= section_threshold
                    and len(block_text) < 120
                    and not block_text.endswith(".")):
                current_section_title = block_text
                text_parts.append(f"[{block_text}]")  # include section title as context
                continue

            text_parts.append(block_text)

        page_text = "\n\n".join(text_parts).strip()
        if len(page_text) < 60:
            continue

        pages.append({
            "text": page_text,
            "page": page_num + 1,
            "chapter_num": current_chapter_num,
            "chapter_title": current_chapter_title,
            "section_title": current_section_title,
            "source": str(pdf_path),
        })

    doc.close()
    print(f"📄 Extracted {len(pages)} content pages with chapter/section metadata.")
    return pages


# ── Parent-Document Chunking ────────────────────────────────────────

def create_parent_child_documents(pages: list) -> list[Document]:
    """
    Parent-document retrieval pattern:
    - Parent chunks (1400 chars): stored in metadata as 'parent_content' → LLM sees this
    - Child chunks (380 chars):  embedded in Qdrant → retrieval precision
    Result: precise retrieval + rich context.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    child_docs = []

    for page_info in pages:
        base_meta = {
            "page": page_info["page"],
            "chapter_num": page_info["chapter_num"],
            "chapter_title": page_info["chapter_title"],
            "section_title": page_info["section_title"],
            "source": page_info["source"],
        }

        parent_doc = Document(page_content=page_info["text"], metadata=base_meta)

        for parent_chunk in parent_splitter.split_documents([parent_doc]):
            for child in child_splitter.split_documents([parent_chunk]):
                if len(child.page_content.strip()) < 30:
                    continue
                child_docs.append(Document(
                    page_content=child.page_content,       # small → embedded
                    metadata={
                        **child.metadata,
                        "parent_content": parent_chunk.page_content,  # large → LLM
                    },
                ))

    print(f"✅ Created {len(child_docs)} child chunks (each backed by a 1400-char parent).")
    return child_docs


# ── Main ───────────────────────────────────────────────────────────

def main():
    client = QdrantClient(url=QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"⚠️  Collection '{COLLECTION_NAME}' already exists — skipping indexing.")
        print("   To re-index, delete the collection first:")
        print(f"   uv run python -c \"from qdrant_client import QdrantClient; QdrantClient('http://localhost:6333').delete_collection('{COLLECTION_NAME}')\"")
        return

    # Clear stale BM25 cache so retriever rebuilds from new chunks
    if BM25_CACHE_PATH.exists():
        BM25_CACHE_PATH.unlink()
        print("🗑️  Cleared stale BM25 cache.")

    print(f"📖 Extracting text from: {PDF_PATH.name}")
    pages = extract_structured_pages(PDF_PATH)

    print("✂️  Building parent-child document pairs...")
    docs = create_parent_child_documents(pages)

    print(f"🔢 Embedding {len(docs)} child chunks via OpenAI ({EMBED_MODEL})...")
    embedding_model = OpenAIEmbeddings(model=EMBED_MODEL)

    try:
        QdrantVectorStore.from_documents(
            documents=docs,
            embedding=embedding_model,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
        )
        print(f"✅ Indexed {len(docs)} chunks into Qdrant collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        raise

    print("\n🎉 Indexing complete! Run the app with:")
    print("   uv run uvicorn app:app --app-dir code/rag --reload --port 8000")


if __name__ == "__main__":
    main()