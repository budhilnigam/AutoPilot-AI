#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Start AutoPilot AI (backend API + optional frontend dev server)
#
# Usage:
#   ./start.sh             # start only the FastAPI backend (port 8000)
#   ./start.sh --full      # start backend + Vite frontend in parallel
#   ./start.sh --frontend  # start only the Vite frontend (port 5173)
#   ./start.sh --prod      # build frontend then start backend (serves both)
# ─────────────────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[AutoPilot]${NC} $*"; }
success() { echo -e "${GREEN}[AutoPilot]${NC} $*"; }
warn()    { echo -e "${YELLOW}[AutoPilot]${NC} $*"; }
die()     { echo -e "${RED}[AutoPilot ERROR]${NC} $*"; exit 1; }

# ── Check .env ────────────────────────────────────────────────────────────────
if [[ ! -f "$ROOT/.env" ]]; then
    warn ".env not found — copy .env.example and fill in your credentials."
    warn "Continuing with environment variables already set (if any)."
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ -f "$ROOT/venv/bin/activate" ]]; then
    source "$ROOT/venv/bin/activate"
    success "Python venv activated"
else
    warn "No venv found at $ROOT/venv — using system Python"
fi

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE="${1:-backend}"
case "$MODE" in
    --full)     MODE=full ;;
    --frontend) MODE=frontend ;;
    --prod)     MODE=prod ;;
    --backend|"") MODE=backend ;;
    *)
        echo "Usage: $0 [--backend | --full | --frontend | --prod]"
        exit 1
        ;;
esac

# ── Start functions ───────────────────────────────────────────────────────────

start_backend() {
    info "Starting FastAPI backend on http://0.0.0.0:8000"
    info "  API docs : http://localhost:8000/docs"
    info "  Health   : http://localhost:8000/health"
    echo ""
    cd "$ROOT"
    exec uvicorn autopilot_ai.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --reload-include "*.py" \
        --log-level warning 2>&1
}

start_frontend() {
    if ! command -v node &>/dev/null; then
        die "Node.js not found. Install it with: curl -fsSL https://fnm.vercel.app/install | bash && fnm install 20"
    fi
    info "Starting Vite dev server on http://localhost:5173"
    cd "$ROOT/frontend"
    if [[ ! -d node_modules ]]; then
        info "Installing npm packages..."
        npm install
    fi
    exec npm run dev
}

build_frontend() {
    if ! command -v node &>/dev/null; then
        die "Node.js not found. Cannot build frontend."
    fi
    info "Building frontend for production..."
    cd "$ROOT/frontend"
    [[ ! -d node_modules ]] && npm install
    npm run build
    success "Frontend built → frontend/dist/"
}

# ── Execute ────────────────────────────────────────────────────────────────────

case "$MODE" in
    backend)
        start_backend
        ;;

    frontend)
        start_frontend
        ;;

    full)
        info "Starting FULL stack (backend + frontend dev server)"
        echo ""
        # Run backend in background, frontend in foreground
        uvicorn autopilot_ai.api.main:app \
            --host 0.0.0.0 --port 8000 \
            --reload --log-level warning &
        BACKEND_PID=$!
        # Kill backend cleanly when this script exits
        trap "kill $BACKEND_PID 2>/dev/null; wait $BACKEND_PID 2>/dev/null" EXIT
        info "Backend PID $BACKEND_PID — listening on :8000"
        sleep 1
        start_frontend
        ;;

    prod)
        build_frontend
        start_backend   # FastAPI will mount frontend/dist/ at "/"
        ;;
esac
