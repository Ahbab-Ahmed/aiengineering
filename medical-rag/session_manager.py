import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# DB lives at project_root/data/sessions.db
DB_PATH = Path(__file__).parent / "data" / "sessions.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                mode TEXT DEFAULT 'learn',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                score INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(session_id, message_index)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                session_id TEXT PRIMARY KEY,
                content TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()


def create_session(name: Optional[str] = None) -> dict:
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    name = name or f"Session {datetime.utcnow().strftime('%b %d, %H:%M')}"
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (sid, name, now, now),
        )
        conn.commit()
    return {"id": sid, "name": name, "created_at": now, "updated_at": now, "messages": []}


def list_sessions() -> List[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        msgs = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return {**dict(row), "messages": [dict(m) for m in msgs]}


def add_message(session_id: str, role: str, content: str, mode: str = "learn"):
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, mode, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, mode, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()


def add_feedback(session_id: str, message_index: int, score: int):
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO feedback (session_id, message_index, score, created_at) VALUES (?, ?, ?, ?)",
            (session_id, message_index, score, now),
        )
        conn.commit()


def get_notes(session_id: str) -> str:
    with _conn() as conn:
        row = conn.execute(
            "SELECT content FROM notes WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row["content"] if row else ""


def save_notes(session_id: str, content: str):
    now = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO notes (session_id, content, updated_at) VALUES (?, ?, ?)",
            (session_id, content, now),
        )
        conn.commit()


def delete_session(session_id: str):
    with _conn() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
