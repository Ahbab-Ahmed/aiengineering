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

from config import CURRICULUM
import session_manager as sessions
from retriever import HybridRetriever
import tutor

# ── App ────────────────────────────────────────────────────────────

app = FastAPI(title="AI Engineering Tutor", version="3.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

retriever_instance: Optional[HybridRetriever] = None
startup_error: Optional[str] = None


@app.on_event("startup")
async def startup():
    global retriever_instance, startup_error
    try:
        sessions.init_db()
        retriever_instance = HybridRetriever()
    except Exception as e:
        startup_error = str(e)
        print(f"❌ Startup error: {e}")


# ── Static + Health ────────────────────────────────────────────────

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


# ── Chat (streaming) ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "learn"


@app.post("/chat")
def chat(body: ChatRequest):
    if retriever_instance is None:
        raise HTTPException(503, startup_error or "Server initializing — please wait.")

    s = sessions.get_session(body.session_id)
    if not s:
        raise HTTPException(404, "Session not found")

    history = [{"role": m["role"], "content": m["content"]} for m in s["messages"]]
    sessions.add_message(body.session_id, "user", body.message, body.mode)

    try:
        expanded = tutor.expand_query(body.message, history)
        docs = retriever_instance.retrieve(expanded, top_k=6)
    except Exception as e:
        raise HTTPException(500, f"Retrieval failed: {e}")

    # Build context using parent_content (rich 1400-char chunks)
    context_parts, sources, seen_pages = [], [], set()
    for doc in docs:
        page = doc["metadata"].get("page", 0) + 1
        ch_title = doc["metadata"].get("chapter_title", "")
        ch_num = doc["metadata"].get("chapter_num", 0)
        section = doc["metadata"].get("section_title", "")

        label = f"[Page {page}"
        if ch_num:
            label += f" | {ch_title}"
        if section:
            label += f" › {section}"
        label += "]"

        context_parts.append(f"{label}\n{doc['text']}")
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({
                "page": page,
                "chapter_title": ch_title,
                "chapter_num": ch_num,
            })

    context = "\n\n---\n\n".join(context_parts)
    session_id, mode = body.session_id, body.mode

    def generate():
        full_text = ""
        try:
            for chunk in tutor.stream_response(mode, body.message, context, sources, history):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:].strip())
                        if data["type"] == "token":
                            full_text += data["content"]
                        elif data["type"] == "done":
                            sessions.add_message(session_id, "assistant", full_text, mode)
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
    session_id: str
    message_index: int
    score: int  # 1 = 👍, -1 = 👎


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


# ── Curriculum ─────────────────────────────────────────────────────

@app.get("/curriculum")
def get_curriculum():
    return CURRICULUM
