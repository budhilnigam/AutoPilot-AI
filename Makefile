# ─────────────────────────────────────────────────────────────────────────────
# Makefile — AutoPilot AI convenience targets
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-frontend backend frontend dev prod \
        test lint typecheck health query alerts

VENV_ACTIVATE := . venv/bin/activate
FRONTEND_DIR  := frontend

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────────
install:       ## Create venv and install Python deps
	python3 -m venv venv
	$(VENV_ACTIVATE) && pip install --upgrade pip
	$(VENV_ACTIVATE) && pip install -e ".[dev]"

install-frontend: ## Install Node deps for the frontend
	cd $(FRONTEND_DIR) && npm install

# ── Running ───────────────────────────────────────────────────────────────────
backend:       ## Start FastAPI on :8000 with auto-reload
	$(VENV_ACTIVATE) && uvicorn autopilot_ai.api.main:app \
		--host 0.0.0.0 --port 8000 --reload --log-level warning

frontend:      ## Start Vite dev server on :5173 (proxy → :8000)
	cd $(FRONTEND_DIR) && npm run dev

dev:           ## Start BOTH backend (:8000) and frontend (:5173) in parallel
	bash start.sh --full

prod:          ## Build frontend then start backend serving both at :8000
	bash start.sh --prod

# ── HTTP helpers (no frontend needed) ────────────────────────────────────────
health:        ## GET /health — check the API is alive
	curl -s http://localhost:8000/health | python3 -m json.tool

health-detail: ## GET /health/detail — full dependency status
	curl -s http://localhost:8000/health/detail | python3 -m json.tool

alerts:        ## GET /api/alerts — list recent alerts
	curl -s http://localhost:8000/api/alerts | python3 -m json.tool

query:         ## POST /api/query — ask a question (override Q="your question")
	$(eval Q ?= "Are there any performance issues right now?")
	curl -s -X POST http://localhost:8000/api/query \
		-H "Content-Type: application/json" \
		-d '{"query": "$(Q)", "mode": "query"}' \
		| python3 -m json.tool

stream:        ## POST /api/query/stream — stream SSE to terminal (set Q=)
	$(eval Q ?= "Give me a quick system health summary.")
	curl -N -X POST http://localhost:8000/api/query/stream \
		-H "Content-Type: application/json" \
		-H "Accept: text/event-stream" \
		-d '{"query": "$(Q)", "mode": "query"}'

# ── Quality ───────────────────────────────────────────────────────────────────
test:          ## Run the test suite
	$(VENV_ACTIVATE) && pytest tests/ -v

lint:          ## Lint with ruff
	$(VENV_ACTIVATE) && ruff check autopilot_ai/

typecheck:     ## Type-check with mypy
	$(VENV_ACTIVATE) && mypy autopilot_ai/
