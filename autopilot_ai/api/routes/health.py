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
import os
import time
from typing import Any

import boto3
from fastapi import APIRouter
from pydantic import BaseModel

from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["health"])

# Maximum time (seconds) to wait for a single dependency check.
# Uses env override to match standalone diagnostics behavior.
_DEFAULT_CHECK_TIMEOUT = 8.0


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
    """Verify we can reach the Bedrock endpoint (ListFoundationModels - free call)."""
    t0 = time.monotonic()
    check_timeout = float(os.getenv("BEDROCK_HEALTH_TIMEOUT_SECONDS", str(_DEFAULT_CHECK_TIMEOUT)))

    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                _call_bedrock,
            ),
            timeout=check_timeout,
        )
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(name="bedrock", healthy=True, latency_ms=round(latency, 1))
    except asyncio.TimeoutError:
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="bedrock",
            healthy=False,
            latency_ms=round(latency, 1),
            detail=f"timed out after {check_timeout:.1f}s",
        )
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="bedrock",
            healthy=False,
            latency_ms=round(latency, 1),
            detail=str(e)[:120],
        )


def _call_bedrock() -> None:
    """Build a boto session and call Bedrock with deterministic credential precedence."""
    access_key = settings.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("aws_access_key_id")
    secret_key = settings.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("aws_secret_access_key")
    profile = settings.aws_profile or os.getenv("AWS_PROFILE") or os.getenv("aws_profile")

    session_kwargs: dict[str, str] = {}
    if access_key and secret_key:
        session_kwargs["aws_access_key_id"] = access_key
        session_kwargs["aws_secret_access_key"] = secret_key
    elif profile:
        session_kwargs["profile_name"] = profile

    # If explicit keys exist, ignore profile env vars for this call to avoid profile lookup errors.
    restore_aws_profile = os.environ.pop("AWS_PROFILE", None) if access_key and secret_key else None
    restore_aws_profile_lower = os.environ.pop("aws_profile", None) if access_key and secret_key else None

    try:
        session = boto3.Session(**session_kwargs)
        client = session.client("bedrock", region_name=settings.aws_region)
        client.list_foundation_models(byOutputModality="TEXT")
    finally:
        if restore_aws_profile is not None:
            os.environ["AWS_PROFILE"] = restore_aws_profile
        if restore_aws_profile_lower is not None:
            os.environ["aws_profile"] = restore_aws_profile_lower


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
        
        def _test_github_connection() -> str:
            """Test GitHub connection by fetching authenticated user."""
            gh = Github(settings.github_token)
            user = gh.get_user()
            return user.login  # This makes the actual API call
        
        loop = asyncio.get_running_loop()
        check_timeout = float(os.getenv("GITHUB_HEALTH_TIMEOUT_SECONDS", str(_DEFAULT_CHECK_TIMEOUT)))
        user_login = await asyncio.wait_for(
            loop.run_in_executor(None, _test_github_connection),
            timeout=check_timeout,
        )
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="github", 
            healthy=True, 
            latency_ms=round(latency, 1),
            detail=f"authenticated as {user_login}",
        )
    except asyncio.TimeoutError:
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="github",
            healthy=False,
            latency_ms=round(latency, 1),
            detail=f"timed out after {check_timeout:.1f}s",
        )
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
    Full dependency breakdown for core services.
    Always returns 200 — the per-dependency `healthy` flag tells callers
    what is broken.
    """
    bedrock_status, github_status = await asyncio.gather(
        _check_bedrock(),
        _check_github(),
    )
    deps = [bedrock_status, github_status]
    all_healthy = all(d.healthy for d in deps)

    return HealthResponse(
        status="ok" if all_healthy else "degraded",
        version=_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        dependencies=deps,
    )
