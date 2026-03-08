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

### 2 — Install Python dependencies

All dependencies are listed in the `requirements.txt` file.

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

## Starting the App (Separate Terminals)

It is highly recommended to run the backend and frontend in separate terminals to easily monitor their logs.

### 1. Starting the backend

The backend is a standalone FastAPI + Uvicorn server on port **8000**. It exposes
a full REST + WebSocket API — no frontend is required to use it.

**In Terminal 1:**
```bash
# Windows
venv\Scripts\activate
uvicorn autopilot_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

# Mac/Linux
source venv/bin/activate
uvicorn autopilot_ai.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Starting the frontend

The React dev server runs on port **5173** and proxies all `/api/*` calls to the
backend on `:8000`. The backend must be running first.

**In Terminal 2:**
```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

---

## Production mode (single process)

Builds the React app and has FastAPI serve the static files — one process, one port.

```bash
# 1. Build frontend
cd frontend
npm run build
cd ..

# 2. Run backend
uvicorn autopilot_ai.api.main:app --host 0.0.0.0 --port 8000
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
