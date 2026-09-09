#!/bin/bash
# ─────────────────────────────────────────────
# START — Run this every time you want the app
# Usage: ./start.sh
# ─────────────────────────────────────────────
cd "$(dirname "$0")"

# Guard: check setup was done
if [ ! -f ".env" ]; then
  echo "❌ Not set up yet. Run: ./setup.sh"
  exit 1
fi
if [ ! -f "data/sessions.db" ] && ! curl -s http://localhost:6333/collections &>/dev/null; then
  echo "❌ Qdrant not indexed yet. Run: ./setup.sh"
  exit 1
fi

echo ""
echo "🎓 AI Engineering Tutor"
echo "────────────────────────────────────────────"

# Start Qdrant if not already running
if ! curl -s http://localhost:6333/healthz &>/dev/null; then
  echo "🐋 Starting Qdrant..."
  docker compose up -d
  sleep 3
else
  echo "✅ Qdrant already running"
fi

echo "🚀 Starting app → http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo "────────────────────────────────────────────"
echo ""

uv run uvicorn app:app \
  --app-dir . \
  --port 8000 \
  --log-level warning
