# AutoPilot AI

A multi-agent AI SRE system for Indian startups, built on Amazon Bedrock.
Six specialised agents (Observability, Infra, DB, Cost, CICD, Tool Generator) are
coordinated by a Planner Agent that fans out work, then synthesises a single
narrative response. The UI is a three-panel dark web app — live chat, streaming
AI responses, and a real-time WebSocket alert feed.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 or 3.13 |
| Node.js (frontend only) | 18 or 20 |
| AWS account | Bedrock enabled in `ap-south-1` |

---

## Installation

### 1 — Clone and create a virtual environment

```bash
git clone https://github.com/your-org/autopilot-ai.git
cd autopilot-ai
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 2a — Install Python dependencies (recommended)

All dependencies live in `pyproject.toml`. Install everything in one command:

```bash
# Production deps only
pip install -e .

# Production + dev tools (pytest, ruff, mypy, moto for mocking)
pip install -e ".[dev]"
```

The `-e` flag makes the `autopilot_ai` package importable directly from source —
no reinstall needed when you edit code.

### 2b — Install from requirements.txt (alternative)

If you prefer a flat requirements file (CI, Docker, etc.):

```bash
# Production only
pip install -r requirements.txt

# With dev tools
pip install -r requirements.txt -r requirements-dev.txt
```

### 3 — Install frontend dependencies (optional — only if you want the UI)

```bash
cd frontend
npm install
cd ..
```

### 4 — Copy and configure `.env`

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   AWS_REGION, BEDROCK_MODEL_ID, BEDROCK_HAIKU_MODEL_ID
#   KNOWLEDGE_BASE_ID, S3_BUCKET_NAME
#   GITHUB_TOKEN, GITHUB_MONITORED_REPOS
```

---

## Starting the backend

The backend is a standalone FastAPI + Uvicorn server on port **8000**. It exposes
a full REST + WebSocket API — no frontend is required to use it.

### Option A — uvicorn directly

```bash
source venv/bin/activate
uvicorn autopilot_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B — Makefile shortcut

```bash
make install     # first time only — creates venv + installs deps
make backend     # start FastAPI with --reload
```

### Option C — start.sh

```bash
chmod +x start.sh
./start.sh           # backend only (same as make backend)
```

---

## Starting the frontend

The React dev server runs on port **5173** and proxies all `/api/*` calls to the
backend on `:8000`. The backend must be running first.

### Option A — npm directly

```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

### Option B — Makefile

```bash
make frontend
```

### Option C — start.sh

```bash
./start.sh --frontend
```

---

## Starting both together (full-stack dev mode)

This starts the backend in the background and the frontend in the foreground.
Ctrl+C stops both.

```bash
make dev
# or
./start.sh --full
```

---

## Production mode (single process)

Builds the React app and has FastAPI serve the static files — one process, one port.

```bash
make prod
# or
./start.sh --prod
# Open http://localhost:8000
```

---

## Curling the API directly

All these work without the frontend.

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", ...}

curl http://localhost:8000/health/detail
# Full dependency status (Bedrock, GitHub, poller)
```

### Ask a question (blocking, full JSON response)

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is checkout slow?", "mode": "query"}'
```

### Ask a question (streaming SSE — token by token)

```bash
curl -N -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query": "Give me a system health summary.", "mode": "query"}'
```

Each SSE line is one of:
```
event: agent_progress
data: {"agent": "observability", "status": "done", "insight_count": 3, ...}

event: done
data: {"narrative": "...", "total_insights": 7, ...}

event: error
data: {"detail": "..."}
```

### List recent alerts

```bash
curl http://localhost:8000/api/alerts
```

### Dismiss an alert

```bash
curl -X DELETE http://localhost:8000/api/alerts/<alert_id>
```

### WebSocket — live alert feed

```bash
# wscat (npm install -g wscat)
wscat -c ws://localhost:8000/api/alerts/ws
```

### Interactive API docs

```
http://localhost:8000/docs       # Swagger UI
http://localhost:8000/redoc      # ReDoc
```

---

## Makefile reference

```bash
make help          # list all targets with descriptions

# Setup
make install          # create venv + pip install -e ".[dev]"
make install-frontend # npm install in frontend/

# Running
make backend          # FastAPI on :8000 with --reload
make frontend         # Vite dev server on :5173
make dev              # both (backend bg + frontend fg)
make prod             # npm build + FastAPI serves static at :8000

# API shortcuts (no frontend needed)
make health           # GET /health
make health-detail    # GET /health/detail
make alerts           # GET /api/alerts
make query Q="Why is the DB slow?"
make stream Q="Health summary please"

# Quality
make test             # pytest tests/
make lint             # ruff check
make typecheck        # mypy
```

---

## pyproject.toml reference

`pyproject.toml` is the single source of truth for the Python package. It replaces
`requirements.txt` + `setup.py` + `setup.cfg` + `pytest.ini` + lint config.

```bash
# Install prod deps
pip install -e .

# Install prod + dev deps
pip install -e ".[dev]"

# Run linter directly
ruff check autopilot_ai/

# Run type checker
mypy autopilot_ai/

# Run tests
pytest tests/ -v
```

Tool configs in `pyproject.toml`:
- `[tool.ruff]` — linter, line length 100, Python 3.12 target
- `[tool.mypy]` — strict mode
- `[tool.pytest.ini_options]` — `asyncio_mode = auto`, `testpaths = ["tests"]`

---

## Environment variables

See [`.env.example`](.env.example) for all variables with inline documentation.
Required to start:

| Variable | Description |
|---|---|
| `AWS_REGION` | e.g. `ap-south-1` |
| `BEDROCK_MODEL_ID` | Primary Claude model ID |
| `BEDROCK_HAIKU_MODEL_ID` | Fast/cheap model ID (DeepSeek or Haiku) |
| `KNOWLEDGE_BASE_ID` | Bedrock KB ID |
| `S3_BUCKET_NAME` | S3 bucket for KB document storage |

Optional but useful:

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | For CICD agent + poller |
| `GITHUB_MONITORED_REPOS` | Comma-separated `org/repo` list |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` (default `INFO`) |
| `USD_TO_INR_RATE` | Exchange rate for cost display (default `84.5`) |

---

## Project structure

```
autopilot_ai/
├── api/            FastAPI app, routes (query SSE, alerts WS, health)
├── agents/         Planner + 6 specialised agents
├── core/           Config, logging, exceptions, retry/circuit-breaker
├── integrations/   AWS (Bedrock, CloudWatch, ECS, Billing, S3) + GitHub
├── models/         Pydantic models (metrics, insights, tasks, responses)
└── services/       Alerting, prediction, GitHub poller
frontend/
├── src/            React 18 app (chat panel, alert feed, agent status)
└── vite.config.js  Proxies /api/* to FastAPI :8000 in dev
```
