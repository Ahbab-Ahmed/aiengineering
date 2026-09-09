"""
app.py — FastAPI server for the Medical Guideline Assistant.

Endpoints:
  GET  /                        → serve the frontend UI
  GET  /health                  → readiness check
  GET  /sessions                → list all sessions
  POST /sessions                → create a new session
  GET  /sessions/{id}           → get session + messages
  DELETE /sessions/{id}         → delete a session
  POST /chat                    → main RAG chat endpoint (streaming SSE)
  POST /feedback                → thumbs up/down on a response
  GET  /notes/{id}              → get session notes
  POST /notes/{id}              → save session notes

The /chat endpoint implements the full pipeline:
  input guardrail → query rewrite → retrieval → confidence gate
  → generation → output guardrail → stream response
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import session_manager as sessions
from retriever import MedicalRetriever
from safety import (
    check_input, check_confidence, check_output,
    GuardrailStatus, REFUSALS,
)
from assistant import rewrite_query, stream_answer, extract_cited_ids
from config import CONFIDENCE_THRESHOLD, DISCLAIMER, RERANK_TOP_K

# ── App setup ──────────────────────────────────────────────────────

app = FastAPI(title="Medical Guideline Assistant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

retriever_instance: Optional[MedicalRetriever] = None
startup_error: Optional[str] = None


@app.on_event("startup")
async def startup():
    global retriever_instance, startup_error
    try:
        sessions.init_db()
        retriever_instance = MedicalRetriever()
    except Exception as e:
        startup_error = str(e)
        print(f"❌ Startup error: {e}")


# ── Health + root ──────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"ready": retriever_instance is not None, "error": startup_error}


# ── Sessions ───────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    name: Optional[str] = None


@app.get("/sessions")
def list_sessions():
    return sessions.list_sessions()


@app.post("/sessions")
def create_session(body: SessionCreate):
    return sessions.create_session(body.name)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    s = sessions.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    sessions.delete_session(session_id)
    return {"ok": True}


# ── Chat (streaming SSE) ───────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(body: ChatRequest):
    if retriever_instance is None:
        raise HTTPException(503, startup_error or "Server initializing — please wait.")

    s = sessions.get_session(body.session_id)
    if not s:
        raise HTTPException(404, "Session not found")

    history = [{"role": m["role"], "content": m["content"]} for m in s["messages"]]
    query   = body.message.strip()

    # ── Layer 1: Input guardrail ───────────────────────────────────
    # Check BEFORE doing any retrieval or LLM calls.
    # Fast, cheap, first line of defence.
    input_check = check_input(query)

    if input_check.status != GuardrailStatus.ALLOW:
        # Log the refused query and return the safe message
        sessions.add_message(body.session_id, "user",      query)
        sessions.add_message(body.session_id, "assistant", input_check.message + DISCLAIMER)
        print(f"🛡️  Input blocked [{input_check.status.value}]: {input_check.reason}")

        def refuse_stream():
            msg = input_check.message + DISCLAIMER
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'guardrail', 'status': input_check.status.value})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(refuse_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── Layer 2: Query rewriting + retrieval ───────────────────────
    sessions.add_message(body.session_id, "user", query)

    try:
        rewritten = rewrite_query(query)
        docs      = retriever_instance.retrieve(rewritten, top_k=RERANK_TOP_K)
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {e}")

    # ── Layer 3: Confidence gate ───────────────────────────────────
    # If no docs returned or top score is too low → refuse
    if not docs:
        conf_result = check_confidence(-999, CONFIDENCE_THRESHOLD)
    else:
        top_score   = docs[0].get("score", 0.0)
        conf_result = check_confidence(top_score, CONFIDENCE_THRESHOLD)

    if conf_result.status != GuardrailStatus.ALLOW:
        sessions.add_message(body.session_id, "assistant", conf_result.message + DISCLAIMER)
        print(f"🛡️  Confidence gate [{conf_result.reason}]")

        def low_conf_stream():
            msg = conf_result.message + DISCLAIMER
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'guardrail', 'status': 'low_confidence'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(low_conf_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── Layer 4: Generation + output guardrail ─────────────────────
    retrieved_ids = [d["metadata"].get("chunk_id", "") for d in docs]
    session_id    = body.session_id

    def generate():
        full_text    = ""
        all_citations = []

        try:
            for chunk in stream_answer(query, docs, history):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:].strip())

                        if data["type"] == "token":
                            full_text += data["content"]

                        elif data["type"] == "citations":
                            all_citations = data.get("citations", [])

                        elif data["type"] == "done":
                            # Output guardrail check
                            cited_ids   = extract_cited_ids(full_text)
                            out_check   = check_output(full_text, cited_ids, retrieved_ids)

                            if out_check.status != GuardrailStatus.ALLOW:
                                # Unsafe output — override with safe message
                                safe_msg = out_check.message + DISCLAIMER
                                print(f"🛡️  Output blocked [{out_check.reason}]")
                                sessions.add_message(session_id, "assistant", safe_msg)
                                yield f"data: {json.dumps({'type': 'override', 'content': safe_msg})}\n\n"
                            else:
                                # Safe — save full answer + disclaimer
                                final = full_text + DISCLAIMER
                                sessions.add_message(session_id, "assistant", final)

                            yield f"data: {json.dumps({'type': 'citations', 'citations': all_citations})}\n\n"

                    except Exception:
                        pass

                yield chunk

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── Feedback ───────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id:    str
    message_index: int
    score:         int   # 1 = 👍, -1 = 👎


@app.post("/feedback")
def post_feedback(body: FeedbackRequest):
    sessions.add_feedback(body.session_id, body.message_index, body.score)
    return {"ok": True}


# ── Notes ──────────────────────────────────────────────────────────

class NotesRequest(BaseModel):
    content: str


@app.get("/notes/{session_id}")
def get_notes(session_id: str):
    return {"content": sessions.get_notes(session_id)}


@app.post("/notes/{session_id}")
def save_notes(session_id: str, body: NotesRequest):
    sessions.save_notes(session_id, body.content)
    return {"ok": True}
