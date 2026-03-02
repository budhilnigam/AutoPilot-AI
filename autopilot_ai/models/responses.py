"""
models/responses.py — Agent response and error response models.

Every specialized agent returns an AgentResponse. The Planner collects
these and synthesizes them into the final user-facing response.

Validates Property 35: agent responses include agent_type, task_id, status,
execution_time_ms, and structured data payload.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from autopilot_ai.models.insights import Insight
from autopilot_ai.models.tasks import AgentType


class ResponseStatus(str, enum.Enum):
    """
    Terminal status of an agent task execution.

    SUCCESS  — completed fully; all insights produced
    PARTIAL  — completed with degraded capability (e.g. fallback heuristics used)
    FAILED   — hard failure; no insights produced; error_message populated
    TIMEOUT  — exceeded the task deadline
    SKIPPED  — planner determined this agent was not needed for the query
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class AgentResponse(BaseModel):
    """
    Structured response returned by every specialized agent to the Planner.

    The `data` field carries agent-specific output (e.g. raw metric summaries,
    generated SQL, generated tool code) in addition to the normalised `insights`.

    Validates Property 35.
    """

    agent_type: AgentType
    task_id: str = Field(min_length=1)
    status: ResponseStatus
    execution_time_ms: float = Field(ge=0.0, description="Wall-clock execution time in milliseconds")
    insights: list[Insight] = Field(default_factory=list)
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific structured output beyond normalised Insight objects",
    )
    error_message: str | None = Field(
        default=None,
        description="Human-readable error detail when status is FAILED or PARTIAL",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: str | None = Field(
        default=None,
        description="Bedrock model ID used for this response, for audit/cost tracking",
    )


class ErrorResponse(BaseModel):
    """
    HTTP error response returned by the API layer.

    Used as the response body for 4xx/5xx responses so clients always
    receive consistent JSON rather than FastAPI's default detail string.
    """

    error: str = Field(description="Machine-readable error code, e.g. 'validation_error'")
    message: str = Field(description="Human-readable error message")
    correlation_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryResponse(BaseModel):
    """
    Top-level response returned by POST /api/query.

    Aggregates responses from all invoked agents into a unified object.
    The `narrative` is the Claude-synthesized plain-English summary.
    """

    query_id: str = Field(description="Correlation ID for this query")
    narrative: str = Field(description="Claude-synthesized plain-English summary of all insights")
    agent_responses: list[AgentResponse] = Field(default_factory=list)
    total_insights: int = Field(default=0)
    execution_time_ms: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def all_insights(self) -> list[Insight]:
        """Flat list of all insights across all agent responses."""
        return [insight for resp in self.agent_responses for insight in resp.insights]
