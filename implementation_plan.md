# AutoPilot-AI: Implementation Plan

AutoPilot-AI is a multi-agent AI SRE system for Indian startups, built on Amazon Bedrock. Six specialized agents (Observability, Infra, DB, Cost, CICD, Tool Generator) are coordinated by a Planner Agent. The system targets production quality code for in-house deployment on EC2.

## System UX: Dual-Mode (Chat + Alerts)

The system operates in two complementary modes through a single web UI:

| Mode | Trigger | Flow |
|---|---|---|
| **Chatbot** (on-demand) | User types a question | User → chat panel → Planner Agent → agents → streamed response |
| **Alert** (proactive) | Anomaly or regression detected | GitHub poller / CloudWatch event → Planner Agent → alert pushed to UI via WebSocket |
| **Alert drill-down** | User clicks an alert | Alert context pre-loaded into chat → follow-up Q&A with full context |

Both modes share the same Planner Agent pipeline. Alerts are Planner-initiated queries; chat is user-initiated. Same agent logic, different trigger.

### Real-time Communication
- **Chat responses**: `POST /api/query` → `text/event-stream` (SSE) for token-by-token streaming
- **Live alerts**: `WebSocket /ws/alerts` — server pushes alerts to all connected UI clients
- **Agent status**: `GET /api/agents/status` — polled every 30s by UI sidebar

## Confirmed Setup (from .env)
- **Primary model**: `anthropic.claude-haiku-4-5-20251001-v1:0` (planning/reasoning)
- **Secondary model**: `deepseek.v3-v1:0` (fast/cheap tasks)
- **Region**: `ap-south-1`
- **USD→INR**: 84.5
- **Knowledge Base**: already provisioned (ID to be filled in [.env](file:///home/myuser/AutoPilot-AI/.env))

## Key Design Decisions

> [!IMPORTANT]
> **Tool Generation**: Amazon Q requirement → implemented as **Claude-via-Bedrock** generating Python code, which is then executed on the local/EC2 machine via sandboxed `subprocess`. No Amazon Q API needed.

> [!IMPORTANT]
> **GitHub Integration**: **Polling-first** (5-min `asyncio` background task). Webhook support added later as optional fast-path to the same internal event queue. No public endpoint required on day 1.

> [!NOTE]
> **Tests**: Not running a formal test suite during development. Tests written to `tests/` folder incrementally as directed. No CI gate.

---

## Technology Stack

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.12 | Best AWS/Bedrock SDK support, async, Hypothesis |
| Data validation | Pydantic v2 | Type safety, JSON parsing from LLM outputs |
| Web framework | FastAPI | Native async, auto OpenAPI |
| Config | pydantic-settings | Reads from [.env](file:///home/myuser/AutoPilot-AI/.env) automatically |
| Logging | structlog | Structured JSON logs with correlation IDs |
| Retry/resilience | tenacity | Retry decorator + easy exponential backoff |
| Async AWS | boto3 (async via run_in_executor) | Most complete for Bedrock |
| GitHub API | PyGithub | Mature, well-documented |

---

## Proposed Project Structure

```
autopilot_ai/
├── core/
│   ├── config.py          # pydantic-settings Settings class
│   ├── logging.py         # structlog setup, correlation ID middleware
│   ├── exceptions.py      # Custom exception hierarchy
│   └── retry.py           # @retry decorator, CircuitBreaker
├── models/
│   ├── metrics.py         # MetricData, Anomaly, TimeSeries
│   ├── insights.py        # Insight, Recommendation, CostImpact, enums
│   ├── tasks.py           # Task, AgentType, TaskType, Priority
│   ├── responses.py       # AgentResponse, ResponseStatus, ErrorResponse
│   └── domain.py          # BuildData, QueryPattern, Config, WorkerConfig…
├── integrations/
│   ├── aws/
│   │   ├── bedrock.py     # Async Bedrock invoke + stream
│   │   ├── cloudwatch.py  # Metrics with pagination
│   │   ├── ecs.py         # Task defs, service configs
│   │   ├── billing.py     # Cost Explorer + CSV
│   │   └── s3.py          # Upload/download
│   └── github/
│       └── client.py      # Commits, workflow runs, diff analysis
├── services/
│   ├── knowledge_base.py  # Bedrock KB RAG wrapper
│   ├── alerting.py        # Alert creation, dedup, SNS/webhook delivery
│   ├── prediction.py      # Saturation prediction, ranking
│   └── github_poller.py   # asyncio polling loop
├── agents/
│   ├── base.py            # Abstract BaseAgent (template method pattern)
│   ├── planner.py         # Orchestrator — fan-out + synthesis
│   ├── observability.py   # Metrics, anomalies, bottleneck
│   ├── infra.py           # Dockerfile, ECS, Terraform, worker sizing
│   ├── db.py              # EXPLAIN ANALYZE, index recs, Redis
│   ├── cost.py            # Cost impact, right-sizing, tradeoff
│   ├── cicd.py            # Build regression, trends, failure prediction
│   └── tool_generator.py  # Claude codegen + sandboxed subprocess execution
├── api/
│   ├── main.py            # FastAPI app with lifespan
│   └── routes/
│       ├── query.py       # POST /query
│       ├── alerts.py      # GET/POST /alerts
│       └── health.py      # GET /health/live, /health/ready
└── tests/
    ├── unit/              # Per-module unit tests (added incrementally)
    └── integration/       # End-to-end smoke tests
```

---

## Proposed Changes (Build Order)

### Phase 1 — Core Infrastructure

#### [NEW] `autopilot_ai/core/config.py`
`Settings` class via `pydantic-settings`. All values pulled from [.env](file:///home/myuser/AutoPilot-AI/.env). Includes: AWS creds, Bedrock model IDs, KB ID, GitHub token, USD→INR rate, log level, thresholds (anomaly sigma, regression multiplier, etc.). Singleton via `lru_cache`.

#### [NEW] `autopilot_ai/core/logging.py`
`structlog` configured for JSON output. `get_logger()` helper. `bind_correlation_id()` context manager used by Planner Agent when processing a query — threads trace ID through all downstream agent calls.

#### [NEW] `autopilot_ai/core/exceptions.py`
Exception hierarchy:
```
AutoPilotError
├── BedrockError (model invocations)
├── IntegrationError (AWS API, GitHub API)
│   ├── ThrottlingError
│   └── RateLimitError
├── AgentError (agent execution failures)
│   └── AgentTimeoutError
└── ValidationError (bad input data)
```

#### [NEW] `autopilot_ai/core/retry.py`
`@with_retry` decorator (tenacity, 3 attempts, exponential 2–10s, retries on `ThrottlingError`). `CircuitBreaker` class with `CLOSED/OPEN/HALF_OPEN` states — used to wrap all Bedrock calls.

---

### Phase 2 — Data Models

All models are **Pydantic v2 BaseModel** (not dataclasses). Include field validators that enforce the correctness properties at the data level.

#### [NEW] `autopilot_ai/models/insights.py`
`Insight`, `Recommendation`, `CostImpact`. `CostImpact` has validator: `annual ≈ monthly × 12`. `confidence` fields `Annotated[float, Field(ge=0.0, le=1.0)]`.

#### [NEW] `autopilot_ai/models/responses.py`
`AgentResponse` includes: `agent_type`, `task_id`, `status`, `execution_time_ms`, `insights`, `data`. Enforces Property 35.

---

### Phase 3 — Integration Clients

#### [NEW] `autopilot_ai/integrations/aws/bedrock.py`
Async wrapper around boto3 Bedrock runtime. Two methods:
- `invoke(prompt, model_id) -> str` — standard call
- `invoke_with_schema(prompt, schema: Type[BaseModel]) -> BaseModel` — parses structured JSON output from Claude directly into a Pydantic model

#### [NEW] `autopilot_ai/integrations/github/client.py`
PyGithub wrapper. `get_recent_commits(repo, since_sha)`, `get_workflow_runs(repo, since)`, `get_commit_diff(sha)`. Returns typed Pydantic models.

---

### Phase 4 — Knowledge Base Service

#### [NEW] `autopilot_ai/services/knowledge_base.py`
Wraps Bedrock Knowledge Base APIs. `store_configuration()`, `query_context()` (filters results by similarity > 0.6), `index_metrics()`. Uses S3 integration for raw document storage.

---

### Phase 5 — Base Agent

#### [NEW] `autopilot_ai/agents/base.py`
```python
class BaseAgent(ABC):
    agent_type: AgentType

    async def __call__(self, task: Task) -> AgentResponse:
        # Template method: timing, logging, circuit breaker, fallback
        ...

    @abstractmethod
    async def execute(self, task: Task) -> AgentResponse: ...

    async def _fallback(self, task, error) -> AgentResponse:
        # Returns PARTIAL status with heuristic analysis
        ...
```
Timing, correlation ID logging, and error wrapping happen **once here** — not in each agent.

---

### Phase 6 — Specialized Agents

Each agent: receives `Task` with `parameters` dict → queries KB for context → calls Bedrock → returns `AgentResponse`.

#### [NEW] `autopilot_ai/agents/observability.py`
- `analyze_metrics()`: CloudWatch data → Claude → `Insight` list
- `detect_anomalies()`: 2-sigma statistical check first, then Claude for root cause
- `attribute_bottleneck()`: correlates issue to component

#### [NEW] `autopilot_ai/agents/infra.py`
- `analyze_dockerfile()`, `detect_drift()`, `analyze_worker_sizing()`
- Parses Terraform HCL and docker-compose YAML locally (no LLM for parsing, just for reasoning)

#### [NEW] `autopilot_ai/agents/db.py`
- `analyze_query_plan()`: parses EXPLAIN ANALYZE text with regex/heuristics, sends to Claude for recommendations
- `recommend_indices()`: generates SQL DDL
- `analyze_redis_stats()`: parses Redis INFO key-value output

#### [NEW] `autopilot_ai/agents/cost.py`
- All monetary values in INR using configured rate
- `identify_optimization_opportunities()`: flags resources with <30% utilization over 7 days

#### [NEW] `autopilot_ai/agents/cicd.py`
- `track_build_times()`: stores 30-day history
- `detect_regression()`: >1.5x baseline triggers alert
- `predict_failures()`: failure rate >10% triggers prediction with confidence score

#### [NEW] `autopilot_ai/agents/tool_generator.py`
- Prompts Claude to generate Python code for a described task
- Validates generated code: `ast.parse()` to check syntax, `ast.walk()` to block dangerous calls (`os.system`, `eval`, `exec`, `__import__`)
- Executes via `subprocess.run()` with timeout=30s, `capture_output=True`
- Returns stdout/stderr to calling agent

---

### Phase 7 — Planner Agent

#### [NEW] `autopilot_ai/agents/planner.py`
- Parses query → determines which agents to invoke (keyword + intent matching, optionally Claude for classification)
- Dispatches via `asyncio.gather(*agent_coroutines, return_exceptions=True)`
- `synthesize_responses()`: passes all `AgentResponse` objects to Claude for unified narrative generation
- Maintains agent registry, health-checks agents on startup

---

### Phase 8 — Supporting Services

#### [NEW] `autopilot_ai/services/github_poller.py`
`asyncio` background task. Runs every 5 minutes. Stores last seen commit SHA per repo in memory (future: Redis/DynamoDB). On new commits: pushes commit data to internal `asyncio.Queue`. CICD Agent consumes from queue.

#### [NEW] `autopilot_ai/services/alerting.py`
`AlertService`: creates alerts with severity, deduplication by (metric + component + 5-min window), delivers via SNS or webhook. Tracks alert generation timestamps for the 60-second SLA (Property 28).

#### [NEW] `autopilot_ai/services/prediction.py`
Linear regression on last 7 days of utilization to predict saturation. `time_to_saturation()` returns hours-to-80% estimate. `rank_recommendations()` sorts by cost savings / implementation effort ratio.

---

### Phase 9 — API Layer

#### [NEW] `autopilot_ai/api/main.py`
FastAPI app. `lifespan` context manager starts GitHub poller background task and initialises agent registry on startup.

#### [NEW] `autopilot_ai/api/routes/query.py`
`POST /query` — accepts `{"query": str, "context": dict}`, runs through Planner Agent, returns `Response` with insights and recommendations. Async end-to-end.

#### [NEW] `autopilot_ai/api/routes/health.py`
`GET /health/live` — always 200. `GET /health/ready` — checks Bedrock connectivity and agent registry is populated.

---

## Verification Plan

Since we are not running a formal test suite, each module will be verified as follows:

### Per-module smoke verification
After implementing each phase, run the module directly:
```bash
cd /home/myuser/AutoPilot-AI
source venv/bin/activate
python -m autopilot_ai.core.config       # prints settings
python -m autopilot_ai.integrations.aws.bedrock  # test invoke with a simple prompt
```

### API verification
```bash
uvicorn autopilot_ai.api.main:app --reload --port 8000
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my current EC2 CPU utilization?", "context": {}}'
curl http://localhost:8000/health/ready
```

### Agent-level manual verification
Each agent will have a `__main__` block for standalone testing:
```bash
python -m autopilot_ai.agents.observability   # runs with mock data
python -m autopilot_ai.agents.cost            # prints sample cost impact
```

### Tests (as directed)
Written to `tests/unit/` and `tests/integration/` incrementally. Run individually with:
```bash
pytest tests/unit/test_<module>.py -v
```

---

## Phase 10 — Frontend (React + Vite)

**Stack**: React 18 + Vite + Vanilla CSS (no Tailwind). Served as static files by FastAPI.

**Structure**:
```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatPanel.jsx       # Chat interface with streaming SSE
│   │   ├── AlertFeed.jsx       # Real-time alert feed (WebSocket)
│   │   ├── AgentStatus.jsx     # 6-agent health sidebar
│   │   └── MessageBubble.jsx   # User/AI message rendering (markdown)
│   ├── hooks/
│   │   ├── useSSEStream.js     # SSE streaming hook
│   │   └── useAlertSocket.js   # WebSocket alerts hook
│   ├── App.jsx                 # Root layout (three-panel)
│   └── index.css               # Design system — dark mode, glassmorphism
└── vite.config.js              # Proxy /api → FastAPI :8000
```

**Layout** (three-panel dark UI):
- **Left sidebar**: Agent status indicators (green/yellow/red), system health score
- **Center**: Chat panel — user messages + streaming AI responses with markdown rendering
- **Right sidebar**: Live alert feed — severity-colored cards with drill-down-to-chat button

**Design**: Dark glassmorphism, Inter font, subtle animations on new alerts/messages. Premium feel — not a generic Bootstrap template.

**FastAPI integration**: In production, FastAPI mounts `frontend/dist/` as static files. In dev, Vite proxy forwards `/api/*` and `/ws/*` to FastAPI.
