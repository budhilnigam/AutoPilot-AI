"""
api/routes/health.py — Health and readiness endpoints.

GET /api/health         — liveness probe  (always 200 if the process is up)
GET /api/health/ready   — readiness probe (checks Bedrock + GitHub connectivity)
GET /api/health/detail  — full dependency status for dashboards / runbooks

The health router is intentionally lightweight and must never block for
more than a few hundred milliseconds.  Each dependency check runs with a
short timeout and fails gracefully so a single broken dependency does not
make the whole readiness check fail immediately.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import boto3
from fastapi import APIRouter
from pydantic import BaseModel

from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger
from autopilot_ai.services.github_poller import github_poller

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

# Maximum time (seconds) to wait for a single dependency check
_CHECK_TIMEOUT = 3.0


# ── Response models ────────────────────────────────────────────────────────


class DependencyStatus(BaseModel):
    name: str
    healthy: bool
    latency_ms: float
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded" | "unavailable"
    version: str
    uptime_seconds: float
    dependencies: list[DependencyStatus] = []


# Module-level start time for uptime calculation
_start_time = time.monotonic()
_VERSION = "0.1.0"


# ── Dependency checks ──────────────────────────────────────────────────────


async def _check_bedrock() -> DependencyStatus:
    """Verify we can reach the Bedrock endpoint (ListFoundationModels — free call)."""
    t0 = time.monotonic()
    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: boto3.client(
                    "bedrock",
                    region_name=settings.aws_region,
                ).list_foundation_models(byOutputModality="TEXT")
            ),
            timeout=_CHECK_TIMEOUT,
        )
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(name="bedrock", healthy=True, latency_ms=round(latency, 1))
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="bedrock",
            healthy=False,
            latency_ms=round(latency, 1),
            detail=str(e)[:120],
        )


async def _check_github() -> DependencyStatus:
    """Check GitHub token validity (only if a token is configured)."""
    t0 = time.monotonic()
    if not settings.github_token:
        return DependencyStatus(
            name="github",
            healthy=True,
            latency_ms=0.0,
            detail="no token configured — polling disabled",
        )
    try:
        from github import Github  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: Github(settings.github_token).get_rate_limit()),
            timeout=_CHECK_TIMEOUT,
        )
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(name="github", healthy=True, latency_ms=round(latency, 1))
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="github",
            healthy=False,
            latency_ms=round(latency, 1),
            detail=str(e)[:120],
        )


# ── Route handlers ─────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    """Always returns 200 while the process is alive."""
    return HealthResponse(
        status="ok",
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


@router.get("/health/ready", summary="Readiness probe")
async def readiness() -> dict[str, Any]:
    """
    Run all dependency checks in parallel.
    Returns 200 if all dependencies are healthy, 503 otherwise.
    """
    from fastapi.responses import JSONResponse  # noqa: PLC0415

    bedrock_status, github_status = await asyncio.gather(
        _check_bedrock(),
        _check_github(),
    )
    deps = [bedrock_status, github_status]
    all_healthy = all(d.healthy for d in deps)

    payload = HealthResponse(
        status="ok" if all_healthy else "degraded",
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        dependencies=deps,
    )

    status_code = 200 if all_healthy else 503
    return JSONResponse(content=payload.model_dump(mode="json"), status_code=status_code)


@router.get("/health/detail", response_model=HealthResponse, summary="Detailed dependency status")
async def detail() -> HealthResponse:
    """
    Full dependency breakdown including GitHub poller state.
    Always returns 200 — the per-dependency `healthy` flag tells callers
    what is broken.
    """
    bedrock_status, github_status = await asyncio.gather(
        _check_bedrock(),
        _check_github(),
    )
    poller_status = DependencyStatus(
        name="github_poller",
        healthy=github_poller.is_running,
        latency_ms=0.0,
        detail="running" if github_poller.is_running else "stopped",
    )

    deps = [bedrock_status, github_status, poller_status]
    all_healthy = all(d.healthy for d in deps)

    return HealthResponse(
        status="ok" if all_healthy else "degraded",
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        dependencies=deps,
    )
