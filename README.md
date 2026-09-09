# 🤖 AI Engineering — Learning Projects

A collection of hands-on AI engineering projects built alongside *AI Engineering* by Chip Huyen (O'Reilly, 2025).

---

## Projects

| Project | Description | Stack |
|---------|-------------|-------|
| [ai-tutor](ai-tutor/README.md) | Interactive AI tutor powered by Hybrid RAG + GPT-4o | FastAPI · Qdrant · OpenAI |
| [medical-rag](medical-rag/README.md) | Medical Guideline Assistant grounded in MOHFW PDFs, with input/output guardrails and citation enforcement | FastAPI · Qdrant · OpenAI |

---

## Setup

Dependencies are managed at the repo root with [uv](https://astral.sh/uv).

```bash
uv sync          # install all dependencies
```

Each project has its own `README.md`, `Makefile`, and `setup.sh` inside its folder.
