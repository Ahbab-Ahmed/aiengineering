"""
config.py — Central settings for the Medical Guideline Assistant.

All tunable parameters live here so you never need to hunt through
multiple files to change a threshold or model name.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────
# Everything is relative to this file's parent (medical-rag/)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
GUIDELINES_DIR = DATA_DIR / "guidelines"   # drop your PDFs here

# ── Vector database ────────────────────────────────────────────────
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "medical_guidelines"    # separate from ai-tutor

# ── Embedding & LLM models ─────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-large"     # OpenAI dense embeddings
LLM_MODEL   = "gpt-4o"                    # generation model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local reranker

# ── Chunking ───────────────────────────────────────────────────────
# Smaller than ai-tutor (300 vs 1400) for precise, citable chunks
CHUNK_SIZE    = 300   # tokens per child chunk
CHUNK_OVERLAP = 50    # overlap to preserve context across chunk boundaries
PARENT_SIZE   = 1200  # parent chunk stored for richer LLM context

# ── Retrieval ──────────────────────────────────────────────────────
TOP_K          = 8    # number of candidates from initial retrieval
RERANK_TOP_K   = 5    # number kept after reranking

# Safety ─────────────────────────────────────────────────────────────────────
# Reranker confidence: if the top-1 score is below this, we refuse.
# ms-marco score ranges: >1.0 = highly relevant, -4.0 to 0.0 = weakly relevant,
# <-7.0 = irrelevant (unanswerable topics score -7 to -11).
# Set to -4.0 to allow weakly-indexed medical content while blocking off-topic queries.
CONFIDENCE_THRESHOLD = -4.0

# Standard disclaimer appended to every response
DISCLAIMER = (
    "\n\n---\n"
    "⚠️ **Disclaimer:** This assistant provides information from official "
    "medical guidelines only and does not give personalised medical advice, "
    "diagnosis, or treatment recommendations. Always consult a qualified "
    "healthcare professional."
)

# ── Input guardrail patterns ───────────────────────────────────────
# Queries matching these patterns are blocked BEFORE retrieval.
# This avoids wasting API calls and prevents unsafe generations.

PERSONAL_PATTERNS = [
    r"\bmy\b", r"\bi have\b", r"\bi am\b", r"\bi've\b",
    r"\bshould i\b", r"\bcan i take\b", r"\bam i\b",
    r"\bmy child\b", r"\bmy son\b", r"\bmy daughter\b",
    r"\bmy symptoms\b", r"\bi feel\b", r"\bi'm feeling\b",
]

CRISIS_PATTERNS = [
    r"\boverdose\b", r"\bsuicid\b", r"\bdying\b",
    # \bemergency\b alone is too broad (e.g. "emergency setting" in a clinical question).
    # Require it adjacent to first-person or urgency phrases instead.
    r"(this is a|it's a|having a).*\bemergency\b",
    r"\bmedical emergency\b",
    r"\bcall 911\b", r"\bcall 999\b", r"\bcall 112\b",
    r"\bheart attack\b", r"\bstroke\b",
]

DOSING_PATTERNS = [
    r"\bhow much.*take\b", r"\bdosage for me\b",
    r"\bwhat dose\b", r"\bhow many.*pills\b",
    r"\bcan i take.*mg\b",
    r"\bcorrect dosage\b", r"\bdosage of\b",   # catches "dosage of rifampicin"
    r"\bdose of ors\b", r"\bdose per (hour|day|kg)\b",
]

# Deliberate harmful intent — caught by Layer A regex regardless of Hub ML score.
# Distinct from CRISIS (genuine emergencies) — these are requests to cause harm.
HARMFUL_PATTERNS = [
    r"\bhow to (harm|hurt|poison|kill|injure)\b",
    r"\bwhat (poison|drug|substance) to use\b",
    r"\bwhat poison\b",
    r"\bhow to (make|create|synthesise|synthesize).*poison\b",
    r"\bkill (someone|myself|himself|herself|themselves)\b",
    r"\bhow to (overdose on|abuse|misuse)\b",
]

