# ─────────────────────────────────────────────────────────────────
# Medical Guideline Assistant — Task Runner
# Run from the medical-rag/ directory: make <command>
# ─────────────────────────────────────────────────────────────────

.PHONY: help start stop index evaluate status open

help:
	@echo ""
	@echo "  🏥 Medical Guideline Assistant"
	@echo "  ─────────────────────────────────────────"
	@echo "  make start     → Start Qdrant + web app (http://localhost:8001)"
	@echo "  make stop      → Stop everything"
	@echo "  make index     → (Re)index PDFs from data/guidelines/"
	@echo "  make evaluate  → Run evaluation benchmark"
	@echo "  make status    → Check if services are running"
	@echo "  make open      → Open the app in your browser"
	@echo ""

# Note: runs on port 8001 to avoid conflict with ai-tutor (8000)
start:
	@echo "🐋 Starting Qdrant..."
	@docker compose up -d
	@sleep 3
	@echo "🚀 Starting Medical Guideline Assistant on http://localhost:8001"
	@uv run uvicorn app:app \
		--app-dir . \
		--port 8001 \
		--reload \
		--log-level info

stop:
	@docker compose down
	@pkill -f "uvicorn app:app" 2>/dev/null && echo "✓ stopped" || echo "(not running)"

index:
	@echo "📖 Indexing PDFs from data/guidelines/..."
	@uv run python indexing.py

evaluate:
	@echo "📊 Running evaluation benchmark..."
	@uv run python evaluator.py

status:
	@echo ""
	@curl -s http://localhost:8001/health 2>/dev/null \
		&& echo "  Web app: ✅ http://localhost:8001" \
		|| echo "  Web app: ❌ not running"
	@curl -s http://localhost:6333/collections 2>/dev/null \
		&& echo "  Qdrant:  ✅ http://localhost:6333" \
		|| echo "  Qdrant:  ❌ not running"
	@echo ""

open:
	@open http://localhost:8001
