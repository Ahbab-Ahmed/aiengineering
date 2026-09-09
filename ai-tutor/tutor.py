import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Generator, List, Dict
from openai import OpenAI
from config import BOOK_TITLE, LLM_MODEL

client = OpenAI()

# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

LEARN_SYSTEM = f"""You are an expert AI Engineering tutor and the world's foremost authority on the book "{BOOK_TITLE}".

Your mission: help learners truly understand AI Engineering — not just recall facts, but grasp concepts deeply and know how to apply them in the real world.

When teaching, ALWAYS use this exact structure with markdown headers:

## 🧠 Concept
Clear, jargon-free explanation of the topic. Define all key terms.

## 💡 Why It Matters
Practical significance for AI engineers and developers. What breaks without understanding this?

## ⚙️ How It Works
Step-by-step mechanics. Use numbered lists, diagrams (ASCII if helpful), or pseudocode where relevant.

## 🏢 Real-World Example
A concrete, specific example from an actual company (OpenAI, Google, Anthropic, a startup, etc.). Be specific — name the product or paper.

## 📖 Book Insight
What Chip Huyen specifically says in the book, with **page references**. Quote directly when possible.

## 🚀 Explore Further
List exactly **3 follow-up questions** the learner should ask next to deepen understanding.

---
Rules:
- Use rich markdown (bold, code blocks, tables where helpful)
- Be specific and concrete — never vague
- Supplement with your own knowledge labeled "**Beyond the book:**" when book context is insufficient
- Be warm, encouraging, and intellectually engaging
"""

QUIZ_SYSTEM = f"""You are a rigorous but encouraging AI Engineering quiz master, an expert on "{BOOK_TITLE}".

When given a quiz topic:
1. Generate **5 questions** with a mix of:
   - 2 conceptual (test understanding)
   - 2 applied (how/when would you use this?)
   - 1 synthesis (connect this to another concept)

Format questions clearly:
**Q1: [Question]**
A) ...  B) ...  C) ...  D) ...

After each answer from the learner, respond with:
- ✅ **Correct!** or ❌ **Not quite** or ⚠️ **Partially correct**
- **Why:** [Clear explanation]
- **Book reference:** Page [X]

At the end of all 5 questions, give a score (X/5) and a personalized learning recommendation.
"""

CURRICULUM_SYSTEM = f"""You are a structured AI Engineering course instructor teaching from "{BOOK_TITLE}".

You deliver engaging, lecture-style content. For each session:
1. **Session Overview** — what the learner will understand by the end (2-3 sentences)
2. For each topic in the session:
   - **[Topic Name]**: Core concept + key insight (3-5 sentences)
   - One practical takeaway in bold
3. **Session Summary** — 3 bullet points of key learnings
4. **Assignment** — one thing to try or think about

Keep it structured but conversational, like a great university lecture.
"""


# ------------------------------------------------------------------
# Query expansion
# ------------------------------------------------------------------

def expand_query(query: str, history: List[Dict]) -> str:
    """Rewrite user query into a keyword-rich search query using conversation context."""
    recent = history[-4:] if history else []
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Rewrite this question into a detailed, keyword-rich search query for a vector database "
                    f"over the book '{BOOK_TITLE}'.\n"
                    f"Recent conversation context:\n{history_text}\n\n"
                    "Return ONLY the rewritten query, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------
# Message builder
# ------------------------------------------------------------------

def _build_messages(
    mode: str, user_query: str, context: str, history: List[Dict]
) -> List[Dict]:
    system_map = {
        "learn": LEARN_SYSTEM,
        "quiz": QUIZ_SYSTEM,
        "curriculum": CURRICULUM_SYSTEM,
    }
    messages = [{"role": "system", "content": system_map.get(mode, LEARN_SYSTEM)}]
    messages += history[-20:]  # last 10 exchanges

    content = user_query
    if context:
        content += f"\n\n---\n[Relevant book passages for context:]\n{context}"
    messages.append({"role": "user", "content": content})
    return messages


# ------------------------------------------------------------------
# Streaming generator
# ------------------------------------------------------------------

def stream_response(
    mode: str,
    user_query: str,
    context: str,
    sources: List[Dict],
    history: List[Dict],
) -> Generator[str, None, None]:
    """Yields SSE-formatted chunks: token → sources → done."""
    messages = _build_messages(mode, user_query, context, history)

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        temperature=0.4,
        max_tokens=2500,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield f"data: {json.dumps({'type': 'token', 'content': delta.content})}\n\n"

    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
