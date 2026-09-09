# ─────────────────────────────────────────────────────────────────
# AI Tutor — Task Runner
# Run from the ai-tutor/ directory: make <command>
# ─────────────────────────────────────────────────────────────────

.PHONY: help start stop restart index status open logs

# Default target
help:
	@echo ""
	@echo "  🎓 AI Tutor"
	@echo "  ─────────────────────────────────────────"
	@echo "  make start     → Start Qdrant + web app"
	@echo "  make stop      → Stop everything"
	@echo "  make restart   → Stop then start"
	@echo "  make index     → (Re)index the PDF into Qdrant"
	@echo "  make status    → Check if services are running"
	@echo "  make open      → Open the app in your browser"
	@echo "  make logs      → Tail the app server log"
	@echo ""

# ── Start ──────────────────────────────────────────────────────────
start:
	@echo "🐋 Starting Qdrant..."
	@docker compose up -d
	@echo "⏳ Waiting for Qdrant to be ready..."
	@sleep 3
	@echo "🚀 Starting AI Tutor on http://localhost:8000"
	@uv run uvicorn app:app \
		--app-dir . \
		--port 8000 \
		--reload \
		--log-level info

# ── Stop ───────────────────────────────────────────────────────────
stop:
	@echo "🛑 Stopping Docker services..."
	@docker compose down
	@echo "🛑 Stopping any running uvicorn processes..."
	@pkill -f "uvicorn app:app" 2>/dev/null && echo "   ✓ uvicorn stopped" || echo "   (uvicorn was not running)"
	@echo "✅ All services stopped."

# ── Restart ────────────────────────────────────────────────────────
restart: stop start

# ── Index the PDF ──────────────────────────────────────────────────
index:
	@echo "⚠️  This will DELETE the existing vector database and re-index the PDF."
	@echo "   Press Ctrl+C to cancel, or wait 5 seconds to proceed..."
	@sleep 5
	@echo "🗑️  Deleting old collection..."
	@uv run python -c "\
from qdrant_client import QdrantClient; \
c = QdrantClient('http://localhost:6333'); \
cols = [x.name for x in c.get_collections().collections]; \
c.delete_collection('vector_database') if 'vector_database' in cols else print('(no collection to delete)'); \
print('✅ Done.')"
	@echo "📖 Indexing PDF..."
	@uv run python indexing.py

# ── Status ─────────────────────────────────────────────────────────
status:
	@echo ""
	@echo "  Service Status"
	@echo "  ─────────────────────────────────────────"
	@docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null || echo "  Docker: not running"
	@echo ""
	@curl -s http://localhost:8000/health 2>/dev/null \
		&& echo "  Web app: ✅ http://localhost:8000" \
		|| echo "  Web app: ❌ not running"
	@curl -s http://localhost:6333/collections 2>/dev/null \
		&& echo "  Qdrant:  ✅ http://localhost:6333" \
		|| echo "  Qdrant:  ❌ not running"
	@echo ""

# ── Open browser ───────────────────────────────────────────────────
open:
	@open http://localhost:8000

# ── Tail logs ──────────────────────────────────────────────────────
logs:
	@echo "📋 Tailing uvicorn output (Ctrl+C to exit)..."
	@tail -f /tmp/tutor.log 2>/dev/null || echo "No log file found. Run 'make start' first."
