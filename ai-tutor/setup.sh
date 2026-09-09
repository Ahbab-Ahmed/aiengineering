#!/bin/bash
# ─────────────────────────────────────────────
# SETUP — Run this ONCE before using the app
# Usage: ./setup.sh
# ─────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

echo ""
echo "🎓 AI Engineering Tutor — First-Time Setup"
echo "────────────────────────────────────────────"

# ── Check Docker ──────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "❌ Docker not found."
  echo "   Install Docker Desktop from: https://www.docker.com/products/docker-desktop"
  exit 1
fi
echo "✅ Docker found"

# ── Check uv ─────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "📦 Installing uv (Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env" 2>/dev/null || true
fi
echo "✅ uv found"

# ── Check OpenAI key ──────────────────────────
if [ ! -f ".env" ] || ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
  echo ""
  echo "🔑 OpenAI API key needed."
  echo -n "   Paste your key (starts with sk-...): "
  read -r api_key
  echo "OPENAI_API_KEY=$api_key" > .env
  echo "✅ API key saved to .env"
else
  echo "✅ OpenAI API key found"
fi

# ── Check PDF ─────────────────────────────────
if [ ! -f "data/ai_engineering.pdf" ]; then
  echo ""
  echo "❌ Book PDF not found at: data/ai_engineering.pdf"
  echo "   Place the AI Engineering PDF there, then run setup.sh again."
  exit 1
fi
echo "✅ Book PDF found"

# ── Install Python packages ───────────────────
echo ""
echo "📦 Installing Python dependencies..."
uv sync
echo "✅ Dependencies installed"

# ── Start Qdrant ─────────────────────────────
echo ""
echo "🐋 Starting Qdrant vector database..."
docker compose up -d
echo "⏳ Waiting for Qdrant to be ready..."
sleep 4
echo "✅ Qdrant running"

# ── Index the book ────────────────────────────
echo ""
echo "📖 Indexing the book (this takes 3–4 minutes and costs ~\$0.08)..."
echo "   This only runs once — never again after this."
uv run python indexing.py
echo "✅ Book indexed"

# ── Done ─────────────────────────────────────
echo ""
echo "────────────────────────────────────────────"
echo "🎉 Setup complete! You're ready to go."
echo ""
echo "   Next: run  ./start.sh"
echo "────────────────────────────────────────────"
echo ""
