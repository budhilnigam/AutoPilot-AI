"""
core/logging.py — Structured logging via structlog.

Features:
  - JSON output in production, colourful console output in development
  - Correlation ID context: bind once in Planner, flows to all agents automatically
  - Standard structlog API: logger.info("msg", key=value, ...)
  - Exception info automatically included on logger.exception(...)

Usage:
    from autopilot_ai.core.logging import get_logger, bind_correlation_id

    logger = get_logger(__name__)

    async def handle_query(query_id: str):
        with bind_correlation_id(query_id):
            logger.info("query_received", query=query)
            # All log lines inside this context will carry correlation_id=query_id
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextlib import contextmanager
from typing import Generator

import structlog
from structlog.types import EventDict, WrappedLogger

from autopilot_ai.core.config import settings


def _add_log_level(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add log level string to every event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _drop_colour_message_key(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Remove uvicorn's colour_message key from logs if present."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    """
    Call once at application startup to configure structlog.
    Subsequent calls are idempotent.
    """
    is_dev = settings.log_level == "DEBUG"

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,  # merges correlation_id etc.
        structlog.stdlib.add_logger_name,
        _add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _drop_colour_message_key,
    ]

    if is_dev:
        # Human-readable coloured output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Machine-parseable JSON for production / CloudWatch
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level)

    # Silence noisy third-party loggers
    for noisy in ("botocore", "urllib3", "asyncio", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a named structlog logger.

    Example:
        logger = get_logger(__name__)
        logger.info("agent_started", agent="observability", task_id="abc123")
    """
    return structlog.stdlib.get_logger(name)


@contextmanager
def bind_correlation_id(
    correlation_id: str | None = None,
) -> Generator[str, None, None]:
    """
    Context manager that binds a correlation_id to all log lines within scope.

    Used by the Planner Agent when starting a new query so that all
    downstream agent log lines carry the same correlation_id.

        with bind_correlation_id() as cid:
            # cid is available if you need to pass it to agents
            ...

    Args:
        correlation_id: Explicit ID to use. Generates a UUID4 if None.

    Yields:
        The correlation ID string.
    """
    cid = correlation_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    try:
        yield cid
    finally:
        structlog.contextvars.unbind_contextvars("correlation_id")


def bind_agent_context(agent_type: str, task_id: str) -> None:
    """
    Bind agent_type and task_id to the current log context.
    Call at the start of BaseAgent.execute() so all logs from that
    agent execution carry these fields.
    """
    structlog.contextvars.bind_contextvars(agent_type=agent_type, task_id=task_id)


def clear_agent_context() -> None:
    """Remove agent context bindings (call in BaseAgent finally block)."""
    structlog.contextvars.unbind_contextvars("agent_type", "task_id")


# Configure on import so any module that imports get_logger gets a
# properly configured logger even without an explicit startup call.
configure_logging()
