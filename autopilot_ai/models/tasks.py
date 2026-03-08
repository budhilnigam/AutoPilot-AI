"""
models/tasks.py — Task and agent-type enumerations for the multi-agent system.

A Task is the unit of work dispatched by the Planner Agent to a specialized
agent. Every agent call receives exactly one Task and returns one AgentResponse.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class AgentType(str, enum.Enum):
    """Identifies which specialized agent handles a task."""

    OBSERVABILITY = "observability"
    INFRA = "infra"
    DB = "db"
    COST = "cost"
    CICD = "cicd"
    PLANNER = "planner"


class TaskType(str, enum.Enum):
    """Fine-grained task classification within an agent domain."""

    # Observability
    ANALYZE_METRICS = "analyze_metrics"
    DETECT_ANOMALIES = "detect_anomalies"
    ATTRIBUTE_BOTTLENECK = "attribute_bottleneck"

    # Infra
    ANALYZE_DOCKERFILE = "analyze_dockerfile"
    DETECT_DRIFT = "detect_drift"
    ANALYZE_ECS = "analyze_ecs"
    ANALYZE_WORKER_SIZING = "analyze_worker_sizing"

    # DB
    ANALYZE_QUERY_PLAN = "analyze_query_plan"
    RECOMMEND_INDICES = "recommend_indices"
    ANALYZE_REDIS = "analyze_redis"

    # Cost
    CALCULATE_COST_IMPACT = "calculate_cost_impact"
    ANALYZE_COSTS = "analyze_costs"
    IDENTIFY_OPTIMIZATIONS = "identify_optimizations"
    IDENTIFY_OPTIMIZATION = "identify_optimization"
    RIGHT_SIZE = "right_size"
    FORECAST_COSTS = "forecast_costs"

    # CI/CD
    TRACK_BUILD_TIMES = "track_build_times"
    DETECT_REGRESSION = "detect_regression"
    DETECT_BUILD_REGRESSION = "detect_build_regression"
    PREDICT_FAILURES = "predict_failures"
    PREDICT_BUILD_FAILURE = "predict_build_failure"
    ANALYZE_WORKFLOW = "analyze_workflow"

    # Planner
    PLAN_QUERY = "plan_query"
    SYNTHESIZE_RESPONSES = "synthesize_responses"


class Priority(str, enum.Enum):
    """Task execution priority — used by Planner to order agent dispatch."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Task(BaseModel):
    """
    Unit of work passed from the Planner Agent to a specialized agent.

    The `parameters` dict carries agent-specific input data (metric data,
    config text, build history, etc.) in a loosely-typed way so the same
    Task class works for all agents without subclassing.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique task identifier (UUID4)",
    )
    agent_type: AgentType
    task_type: TaskType
    parameters: dict[str, object] = Field(
        default_factory=dict,
        description="Agent-specific input parameters",
    )
    priority: Priority = Priority.MEDIUM
    correlation_id: str | None = Field(
        default=None,
        description="Planner query correlation ID — links all tasks in one user request",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deadline: datetime | None = Field(
        default=None,
        description="Wall-clock deadline for this task. AgentTimeoutError raised if exceeded.",
    )
    context: dict[str, object] = Field(
        default_factory=dict,
        description="Additional context from KB retrieval or previous agent outputs",
    )
