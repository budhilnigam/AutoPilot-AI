"""
api/routes/query.py — Natural-language query endpoint with Server-Sent Events.

POST /api/query        — fire-and-wait: run the full planner pipeline and
                         return a QueryResponse JSON body when complete.

POST /api/query/stream — SSE streaming: emit one event per agent as results
                         come in, then emit a final "done" event with the
                         synthesised narrative.

SSE event format (text/event-stream):
  event: agent_progress
  data: {"agent": "observability", "status": "success", "insight_count": 3}

  event: agent_progress
  data: {"agent": "cost", "status": "failed", "error": "..."}

  event: done
  data: {"narrative": "...", "query_id": "...", "total_insights": 7, ...}

  event: error
  data: {"error": "planner_error", "message": "..."}

Request body (QueryRequest):
  query:    str          — natural-language question
  context:  dict         — optional extra context (commit SHA, alert payload, etc.)
  mode:     str          — "query" | "alert" | "dashboard"  (default "query")
  timeout:  float        — client-requested timeout in seconds (default 120)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from autopilot_ai.agents.planner import planner
from autopilot_ai.core.exceptions import (
    AgentTimeoutError,
    AutoPilotError,
    CircularDependencyError,
    PlannerError,
)
from autopilot_ai.core.logging import bind_correlation_id, get_logger
from autopilot_ai.models.responses import AgentResponse, ErrorResponse, QueryResponse
from autopilot_ai.models.tasks import AgentType, Priority, Task, TaskType

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["query"])

_DEFAULT_TIMEOUT = 120.0    # seconds
_MAX_TIMEOUT = 300.0


# ── Request / response models ──────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Natural-language question")
    context: dict[str, Any] = Field(default_factory=dict, description="Optional extra context")
    mode: str = Field(default="query", pattern=r"^(query|alert|dashboard)$")
    timeout: float = Field(default=_DEFAULT_TIMEOUT, gt=0, le=_MAX_TIMEOUT)


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_task(req: QueryRequest, query_id: str) -> Task:
    return Task(
        task_type=TaskType.PLAN_QUERY,
        agent_type=AgentType.PLANNER,
        priority=Priority.HIGH if req.mode == "alert" else Priority.MEDIUM,
        parameters={
            "query": req.query,
            "context": {**req.context, "query_id": query_id},
            "mode": req.mode,
        },
    )


def _extract_query_response(agent_resp: AgentResponse) -> QueryResponse:
    """
    The Planner wraps the QueryResponse inside AgentResponse.data.
    Unwrap it back to a QueryResponse.
    """
    raw = agent_resp.data.get("query_response")
    if raw:
        return QueryResponse.model_validate(raw)
    # Fallback: build a minimal QueryResponse from the AgentResponse
    return QueryResponse(
        query_id=agent_resp.task_id,
        narrative=agent_resp.error_message or "No narrative generated.",
        agent_responses=[],
        total_insights=len(agent_resp.insights),
        execution_time_ms=agent_resp.execution_time_ms,
    )


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ── Route: POST /api/query (sync JSON response) ────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    summary="Run a query and return complete results",
)
async def post_query(req: QueryRequest) -> QueryResponse:
    """
    Runs the full planner pipeline and returns the complete QueryResponse when
    all agents finish.  For long queries, prefer the `/api/query/stream` endpoint.
    """
    query_id = str(uuid.uuid4())
    bind_correlation_id(query_id)
    logger.info("query_received", query_id=query_id, mode=req.mode, query=req.query[:80])

    task = _build_task(req, query_id)

    try:
        agent_resp = await asyncio.wait_for(planner.execute(task), timeout=req.timeout)
    except asyncio.TimeoutError:
        logger.warning("query_timeout", query_id=query_id, timeout=req.timeout)
        raise HTTPException(
            status_code=504,
            detail={"error": "timeout", "message": f"Query exceeded {req.timeout}s limit."},
        )
    except (CircularDependencyError, PlannerError) as e:
        logger.error("query_planner_error", query_id=query_id, error=str(e))
        raise HTTPException(
            status_code=400,
            detail={"error": "planner_error", "message": str(e)},
        )
    except AutoPilotError as e:
        logger.error("query_error", query_id=query_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": str(e)},
        )

    result = _extract_query_response(agent_resp)
    logger.info(
        "query_complete",
        query_id=query_id,
        insights=result.total_insights,
        exec_ms=result.execution_time_ms,
    )
    return result


# ── Route: POST /api/query/stream (SSE) ───────────────────────────────────


@router.post(
    "/query/stream",
    response_class=StreamingResponse,
    summary="Run a query and stream per-agent progress via SSE",
)
async def post_query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    """
    Streams Server-Sent Events as each agent completes.

    Clients should consume the event stream and render results incrementally.
    The stream ends with an `event: done` containing the full QueryResponse,
    or `event: error` on failure.
    """
    query_id = str(uuid.uuid4())
    bind_correlation_id(query_id)
    logger.info("query_stream_start", query_id=query_id, mode=req.mode)

    return StreamingResponse(
        _stream_query(req, query_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable nginx buffering
            "X-Query-ID": query_id,
        },
    )


async def _stream_query(
    req: QueryRequest,
    query_id: str,
    http_request: Request,
) -> AsyncIterator[str]:
    """
    Generator that runs the planner and emits SSE events.

    Strategy:
      1. Build a PLAN_QUERY task and run it in a background asyncio task.
      2. While it runs, emit a heartbeat comment every 15s so the connection
         stays alive through proxies.
      3. When the planner finishes, emit per-agent progress events and the
         final 'done' event.
      4. On client disconnect, cancel the background task.
    """
    task = _build_task(req, query_id)

    planner_task = asyncio.create_task(planner.execute(task), name=f"planner_{query_id}")

    # Heartbeat interval so the connection isn't dropped by load balancers
    heartbeat_interval = 15.0
    deadline = time.monotonic() + req.timeout

    try:
        while not planner_task.done():
            # Check client disconnect
            if await http_request.is_disconnected():
                planner_task.cancel()
                logger.info("query_stream_client_disconnected", query_id=query_id)
                return

            try:
                await asyncio.wait_for(
                    asyncio.shield(planner_task),
                    timeout=min(heartbeat_interval, max(0.1, deadline - time.monotonic())),
                )
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    planner_task.cancel()
                    yield _sse_event("error", {
                        "error": "timeout",
                        "message": f"Query exceeded {req.timeout}s limit.",
                        "query_id": query_id,
                    })
                    return
                # Emit keep-alive comment
                yield ": keep-alive\n\n"

        # Task finished — check for exceptions
        if planner_task.cancelled():
            return

        exc = planner_task.exception()
        if exc:
            err_type = type(exc).__name__
            logger.error("query_stream_error", query_id=query_id, error=str(exc))
            yield _sse_event("error", {
                "error": err_type,
                "message": str(exc)[:300],
                "query_id": query_id,
            })
            return

        agent_resp = planner_task.result()
        qr = _extract_query_response(agent_resp)

        # Emit one progress event per agent response for partial rendering
        for ar in qr.agent_responses:
            yield _sse_event("agent_progress", {
                "agent": ar.agent_type.value,
                "status": ar.status.value,
                "insight_count": len(ar.insights),
                "execution_time_ms": round(ar.execution_time_ms, 1),
                "error": ar.error_message,
            })

        # Final done event
        yield _sse_event("done", {
            "query_id": qr.query_id,
            "narrative": qr.narrative,
            "total_insights": qr.total_insights,
            "execution_time_ms": round(qr.execution_time_ms, 1),
            "agent_responses": [
                {
                    "agent": ar.agent_type.value,
                    "status": ar.status.value,
                    "insight_count": len(ar.insights),
                    "insights": [i.model_dump(mode="json") for i in ar.insights],
                }
                for ar in qr.agent_responses
            ],
        })

        logger.info(
            "query_stream_complete",
            query_id=query_id,
            insights=qr.total_insights,
        )

    except asyncio.CancelledError:
        planner_task.cancel()
        return
    except Exception as exc:
        logger.error("query_stream_unexpected", query_id=query_id, error=str(exc), exc_info=True)
        yield _sse_event("error", {
            "error": "internal_error",
            "message": str(exc)[:300],
            "query_id": query_id,
        })
