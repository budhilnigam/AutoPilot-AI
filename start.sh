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
VENV_CANDIDATES=("envn" "venv" "env" ".venv")
VENV_ACTIVATED=0
for v in "${VENV_CANDIDATES[@]}"; do
    if [[ -f "$ROOT/$v/bin/activate" ]]; then
        source "$ROOT/$v/bin/activate"
        success "Python venv activated: $v"
        VENV_ACTIVATED=1
        break
    fi
done

if [[ "$VENV_ACTIVATED" -eq 0 ]]; then
    warn "No virtual environment found in: ${VENV_CANDIDATES[*]} — using system Python"
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
    info "  Health   : http://localhost:8000/api/health"
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

cleanup_backend() {
    # Ensure uvicorn --reload parent and worker are both stopped.
    if [[ -n "${BACKEND_PID:-}" ]]; then
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            kill -TERM -- "-$BACKEND_PID" 2>/dev/null || kill -TERM "$BACKEND_PID" 2>/dev/null || true
            # Give processes a moment to exit gracefully before forcing.
            sleep 0.5
            kill -KILL -- "-$BACKEND_PID" 2>/dev/null || kill -KILL "$BACKEND_PID" 2>/dev/null || true
        fi
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
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
        # Run backend in its own process group so Ctrl+C can stop reloader + worker.
        if command -v setsid >/dev/null 2>&1; then
            setsid uvicorn autopilot_ai.api.main:app \
                --host 0.0.0.0 --port 8000 \
                --reload --log-level warning &
        else
            uvicorn autopilot_ai.api.main:app \
                --host 0.0.0.0 --port 8000 \
                --reload --log-level warning &
        fi
        BACKEND_PID=$!
        # Stop backend cleanly (and force if needed) when this script exits.
        trap cleanup_backend EXIT INT TERM
        info "Backend PID $BACKEND_PID — waiting for readiness..."
        # Wait for backend to be ready (max 10 seconds)
        for i in {1..20}; do
            if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then
                success "Backend ready!"
                break
            fi
            sleep 0.5
        done
        start_frontend
        ;;

    prod)
        build_frontend
        start_backend   # FastAPI will mount frontend/dist/ at "/"
        ;;
esac
