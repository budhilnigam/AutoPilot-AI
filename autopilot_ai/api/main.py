"""
api/main.py — FastAPI application factory and ASGI entry point.

Usage:
    uvicorn autopilot_ai.api.main:app --host 0.0.0.0 --port 8000 --reload

Or programmatically:
    from autopilot_ai.api.main import app

Startup sequence:
  1. Configure structlog JSON logging.
  2. Mount all route routers.
  3. Set up CORS for the React frontend (localhost:5173 / localhost:3000).
  4. Register global exception handlers for consistent JSON error responses.
    5. On startup event: log readiness details.
    6. On shutdown event: clean up resources.

OpenAPI docs are available at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import (
    AgentTimeoutError,
    AutoPilotError,
    BedrockThrottlingError,
    CircularDependencyError,
    GitHubRateLimitError,
    PlannerError,
)
from autopilot_ai.core.logging import get_logger
from autopilot_ai.models.responses import ErrorResponse

logger = get_logger(__name__)


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    startup:  log service startup context
    shutdown: log a clean exit
    """
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info(
        "autopilot_starting",
        version="0.1.0",
        env=settings.aws_region,
        model=settings.bedrock_model_id,
    )

    # Temporarily disabled: GitHub poller startup.
    logger.info("github_poller_temporarily_disabled")

    yield  # application runs

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("autopilot_shutdown_complete")


# ── Application factory ────────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoPilot-AI",
        description=(
            "AI-powered infrastructure autopilot. "
            "Analyses metrics, costs, CI/CD, database performance, and ECS in real time."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Query-ID"],
    )

    # ── Correlation-ID middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        from autopilot_ai.core.logging import bind_correlation_id  # noqa: PLC0415
        bind_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

    # ── Routers ───────────────────────────────────────────────────────────
    from autopilot_ai.api.routes.health import router as health_router  # noqa: PLC0415
    from autopilot_ai.api.routes.query import router as query_router    # noqa: PLC0415
    from autopilot_ai.api.routes.alerts import router as alerts_router  # noqa: PLC0415

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(alerts_router)

    # ── Exception handlers ────────────────────────────────────────────────

    @app.exception_handler(BedrockThrottlingError)
    async def throttling_handler(request: Request, exc: BedrockThrottlingError):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=ErrorResponse(
                error="bedrock_throttling",
                message="Bedrock model is being throttled. Please retry shortly.",
                details={"retry_after": 60},
            ).model_dump(mode="json"),
            headers={"Retry-After": "60"},
        )

    @app.exception_handler(AgentTimeoutError)
    async def timeout_handler(request: Request, exc: AgentTimeoutError):
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ErrorResponse(
                error="agent_timeout",
                message=str(exc),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(CircularDependencyError)
    @app.exception_handler(PlannerError)
    async def planner_error_handler(request: Request, exc: AutoPilotError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="planner_error",
                message=str(exc),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(GitHubRateLimitError)
    async def github_rate_limit_handler(request: Request, exc: GitHubRateLimitError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="github_rate_limited",
                message="GitHub API rate limit exhausted.",
            ).model_dump(mode="json"),
        )

    @app.exception_handler(AutoPilotError)
    async def autopilot_error_handler(request: Request, exc: AutoPilotError):
        logger.error("unhandled_autopilot_error", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message=str(exc),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            error=str(exc),
            traceback=traceback.format_exc()[-500:],
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred.",
            ).model_dump(mode="json"),
        )

    # ── Static frontend (production) ──────────────────────────────────────
    # Mount LAST so it never shadows the API routes above.
    # In development, Vite serves from port 5173 — this mount is a no-op.
    try:
        from pathlib import Path  # noqa: PLC0415
        from fastapi.staticfiles import StaticFiles  # noqa: PLC0415

        dist_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
        if dist_dir.is_dir():
            app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")
            logger.info("frontend_mounted", path=str(dist_dir))
    except Exception as _e:
        logger.debug("frontend_not_mounted", reason=str(_e)[:80])

    return app


# ── Module-level app instance (used by uvicorn) ───────────────────────────
app = create_app()
