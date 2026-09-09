"""
assistant.py — Grounded answer generation with mandatory citations.

Key design principles:
  1. The LLM only sees retrieved chunks — no general medical knowledge
  2. Every factual claim must be cited with a [chunk_id]
  3. Output is structured JSON: { answer, citations }
  4. If the answer isn't in the chunks, the model must say so
  5. All responses get a disclaimer appended

This replaces tutor.py from ai-tutor with stricter citation enforcement.
"""

import sys
import json
import re
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import openai
from config import LLM_MODEL, DISCLAIMER


# ── System prompt ──────────────────────────────────────────────────
# This is the most important safety control in the generation step.
# It tells the model exactly what it can and cannot do.

SYSTEM_PROMPT = """You are a Medical Guideline Information Assistant.

Your ONLY job is to help users understand information contained in the official medical guidelines provided to you as context. You do NOT have general medical knowledge — you only know what is in the provided context.

STRICT RULES:
1. Answer ONLY using information from the provided guideline excerpts.
2. Cite the chunk_id for EVERY factual claim you make, like this: [chunk_abc123]
3. If the answer is not in the provided excerpts, say: "This information is not available in the guidelines I have access to."
4. Never give personalised medical advice, diagnosis, specific dosing, or treatment recommendations.
5. Never speculate or fill gaps with general medical knowledge.
6. Do not recommend any specific medication or treatment for an individual.

OUTPUT FORMAT — respond with valid JSON only:
{
  "answer": "Your grounded answer with inline citations like [chunk_abc123]...",
  "citations": [
    {
      "chunk_id": "chunk_abc123",
      "source": "filename.pdf",
      "page": 12,
      "doc_id": "document_name"
    }
  ]
}

If you cannot answer from the provided context, return:
{
  "answer": "This information is not available in the guidelines I have access to.",
  "citations": []
}"""


# ── Context builder ────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    Each chunk is labeled with its chunk_id so the model can cite it.
    We use parent_text (richer, ~1200 chars) rather than the smaller
    child text used for retrieval — gives the LLM more context.
    """
    parts = []
    for chunk in chunks:
        meta    = chunk["metadata"]
        cid     = meta.get("chunk_id", "unknown")
        source  = meta.get("source",   "unknown")
        page    = meta.get("page",     "?")
        text    = chunk.get("parent_text", chunk["text"])

        parts.append(
            f"[{cid}] Source: {source}, Page {page}\n{text}"
        )

    return "\n\n---\n\n".join(parts)


# ── Citation extractor ─────────────────────────────────────────────

def extract_cited_ids(answer: str) -> list[str]:
    """
    Pull out all [chunk_xxxxx] references from the generated answer.
    Used by the output guardrail to verify citation integrity.
    """
    return re.findall(r"\[chunk_[a-f0-9]+\]", answer)


# ── Query rewriting ────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """
    Normalize the user query for better retrieval against guideline text.

    Medical guidelines use formal terminology. A user might write
    "BP" while the guideline says "blood pressure". This step bridges that gap.

    Kept simple here — expands common abbreviations and adds context.
    """
    abbreviations = {
        r"\bBP\b": "blood pressure",
        r"\bDM\b": "diabetes mellitus",
        r"\bT2DM\b": "type 2 diabetes mellitus",
        r"\bHTN\b": "hypertension",
        r"\bTB\b": "tuberculosis",
        r"\bARI\b": "acute respiratory infection",
        r"\bORS\b": "oral rehydration solution",
        r"\bIMCI\b": "integrated management of childhood illness",
        r"\bANC\b": "antenatal care",
        r"\bCOPD\b": "chronic obstructive pulmonary disease",
    }
    rewritten = query
    for pattern, expansion in abbreviations.items():
        rewritten = re.sub(pattern, expansion, rewritten, flags=re.IGNORECASE)
    return rewritten


# ── Main generation function ───────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> dict:
    """
    Generate a grounded, cited answer from retrieved chunks.

    Returns a dict:
      {
        "answer":       str,       # answer text with [chunk_id] citations
        "citations":    list[dict],# citation metadata
        "chunk_ids":    list[str], # IDs cited by the model (for guardrail check)
        "raw_response": str,       # raw LLM output (for debugging)
      }
    """
    context = build_context(chunks)
    history = history or []

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
    ]

    # Include conversation history for multi-turn context
    for msg in history[-6:]:   # last 3 turns
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"GUIDELINE EXCERPTS:\n\n{context}\n\n"
            f"---\n\nQUESTION: {query}\n\n"
            "Remember: cite every claim with its [chunk_id]. "
            "Respond with valid JSON only."
        ),
    })

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,       # low temperature for factual grounding
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    # Parse the JSON response
    try:
        parsed = json.loads(raw)
        answer     = parsed.get("answer", "")
        citations  = parsed.get("citations", [])
    except json.JSONDecodeError:
        # Fallback if model returned malformed JSON
        answer     = raw
        citations  = []

    chunk_ids_cited = extract_cited_ids(answer)

    return {
        "answer":       answer,
        "citations":    citations,
        "chunk_ids":    chunk_ids_cited,
        "raw_response": raw,
    }


# ── Streaming version ──────────────────────────────────────────────

def stream_answer(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """
    Streaming version of generate_answer for real-time UI updates.

    Yields Server-Sent Event strings:
      data: {"type": "token", "content": "..."}
      data: {"type": "citations", "citations": [...]}
      data: {"type": "done"}
      data: {"type": "error", "content": "..."}
    """
    context = build_context(chunks)
    history = history or []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": (
            f"GUIDELINE EXCERPTS:\n\n{context}\n\n"
            f"---\n\nQUESTION: {query}\n\n"
            "Cite every claim with its [chunk_id]. Respond with valid JSON only."
        ),
    })

    client = openai.OpenAI()

    try:
        full_text = ""
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            text_chunk = delta.content or ""
            if text_chunk:
                full_text += text_chunk
                yield f"data: {json.dumps({'type': 'token', 'content': text_chunk})}\n\n"

        # After streaming, parse citations from the complete response
        try:
            parsed    = json.loads(full_text)
            citations = parsed.get("citations", [])
        except json.JSONDecodeError:
            citations = []

        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
