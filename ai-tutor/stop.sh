#!/bin/bash
# ─────────────────────────────────────────────
# STOP — Shut everything down
# Usage: ./stop.sh
# ─────────────────────────────────────────────
cd "$(dirname "$0")"

echo ""
echo "🛑 Stopping AI Engineering Tutor..."

# Stop uvicorn (if running in background)
pkill -f "uvicorn app:app" 2>/dev/null && echo "✅ Web app stopped" || echo "   (web app was not running)"

# Stop Qdrant
docker compose down 2>/dev/null && echo "✅ Qdrant stopped" || echo "   (Qdrant was not running)"

echo ""
echo "👋 All done. Run ./start.sh to start again."
echo ""
