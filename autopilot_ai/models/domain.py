"""
models/domain.py — Domain-specific data models for CI/CD, database, infrastructure, and alerting.

These are the typed data structures that flow between integrations and agents.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ── CI/CD Models ──────────────────────────────────────────────────────────


class BuildStatus(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"


class BuildData(BaseModel):
    """
    A single GitHub Actions workflow run.

    Used by the CICD Agent to track build-time trends and detect regressions.
    Validates Properties 17, 18, 33.
    """

    workflow_run_id: int
    commit_sha: str = Field(min_length=7, max_length=40)
    workflow_name: str = Field(min_length=1)
    branch: str = Field(default="main")
    duration_seconds: float = Field(ge=0.0)
    status: BuildStatus
    started_at: datetime
    finished_at: datetime | None = None
    logs: str | None = None
    runner_os: str | None = None
    triggered_by: str | None = Field(
        default=None, description="GitHub actor who triggered the run"
    )

    @model_validator(mode="after")
    def validate_duration_matches_timestamps(self) -> "BuildData":
        if self.finished_at and self.started_at:
            computed = (self.finished_at - self.started_at).total_seconds()
            if abs(computed - self.duration_seconds) > 5:  # 5s tolerance
                # Prefer computed over stored if they differ
                object.__setattr__(self, "duration_seconds", max(computed, 0.0))
        return self


# ── Database Models ────────────────────────────────────────────────────────


class QueryPattern(BaseModel):
    """
    A database query pattern extracted from PostgreSQL EXPLAIN ANALYZE output.

    Used by the DB Agent to identify missing index opportunities.
    Validates Property 11.
    """

    query_text: str = Field(min_length=1)
    execution_time_ms: float = Field(ge=0.0)
    rows_examined: int = Field(ge=0)
    rows_returned: int = Field(ge=0)
    indices_used: list[str] = Field(default_factory=list)
    table_names: list[str] = Field(default_factory=list)
    has_sequential_scan: bool = False
    has_sort: bool = False
    explain_output: str | None = Field(
        default=None, description="Raw EXPLAIN ANALYZE text"
    )


class RedisStats(BaseModel):
    """Parsed Redis INFO ALL output — key fields for DB Agent analysis."""

    used_memory_bytes: int = Field(ge=0)
    maxmemory_bytes: int = Field(ge=0, description="0 means no limit configured")
    mem_fragmentation_ratio: float = Field(ge=0.0)
    evicted_keys: int = Field(ge=0)
    keyspace_hits: int = Field(ge=0)
    keyspace_misses: int = Field(ge=0)
    connected_clients: int = Field(ge=0)
    eviction_policy: str = Field(default="noeviction")
    redis_version: str = Field(default="unknown")

    @property
    def memory_usage_ratio(self) -> float | None:
        if self.maxmemory_bytes == 0:
            return None
        return self.used_memory_bytes / self.maxmemory_bytes

    @property
    def hit_rate(self) -> float | None:
        total = self.keyspace_hits + self.keyspace_misses
        return self.keyspace_hits / total if total > 0 else None


# ── Infrastructure Models ──────────────────────────────────────────────────


class ConfigType(str, enum.Enum):
    DOCKERFILE = "dockerfile"
    DOCKER_COMPOSE = "docker_compose"
    TERRAFORM = "terraform"
    ECS_TASK_DEFINITION = "ecs_task_definition"
    GITHUB_ACTIONS = "github_actions"
    EXPLAIN_ANALYZE = "explain_analyze"
    REDIS_INFO = "redis_info"
    BILLING_CSV = "billing_csv"
    CUSTOM = "custom"


class Configuration(BaseModel):
    """
    A versioned infrastructure configuration document.

    Stored in the Knowledge Base (KB) and retrieved for context by agents.
    Validates Property 20 (round-trip consistency) via the checksum field.
    """

    config_type: ConfigType
    content: str = Field(min_length=1)
    source_path: str = Field(
        default="unknown",
        description="File path or S3 key where this config originated",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    commit_sha: str | None = None
    stored_at: datetime = Field(default_factory=datetime.utcnow)
    checksum: str | None = Field(
        default=None,
        description="SHA-256 hex digest of content for round-trip consistency checks",
    )


class WorkerConfig(BaseModel):
    """
    Worker pool configuration — input to the Infra Agent's worker-sizing analysis.

    Validates Property 10: if saturation or under-utilisation is detected, the
    Infra Agent must recommend specific concurrency values.
    """

    service_name: str = Field(min_length=1)
    worker_count: int = Field(ge=1)
    concurrency: int = Field(ge=1, description="Tasks per worker (e.g. Celery -c flag)")
    queue_size: int = Field(ge=0)
    memory_mb: int = Field(ge=1)
    cpu_units: int = Field(ge=1, description="ECS CPU units (1024 = 1 vCPU)")
    avg_task_duration_seconds: float | None = Field(
        default=None, ge=0.0
    )
    peak_tasks_per_minute: float | None = Field(
        default=None, ge=0.0
    )


class ECSServiceConfig(BaseModel):
    """ECS service configuration snapshot."""

    cluster_name: str
    service_name: str
    task_definition_arn: str
    desired_count: int = Field(ge=0)
    running_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    cpu_units: int = Field(ge=0)
    memory_mb: int = Field(ge=0)
    container_definitions: list[dict[str, Any]] = Field(default_factory=list)
    environment_variables: dict[str, str] = Field(default_factory=dict)
    last_updated: datetime | None = None


# ── Alerting Models ────────────────────────────────────────────────────────


class AlertSeverity(str, enum.Enum):
    CRITICAL = "critical"  # Page immediately (Property 28: 60s SLA)
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Alert(BaseModel):
    """
    A system-generated alert pushed to the UI via WebSocket or SNS.

    Validates Property 28 (generation within 60s for CRITICAL) and
    Property 30 (cost impact included when applicable).
    """

    alert_id: str = Field(min_length=1)
    severity: AlertSeverity
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    component: str = Field(min_length=1)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    delivered_at: datetime | None = None
    cost_impact_inr: float | None = Field(
        default=None,
        description="Projected cost impact if applicable (Property 30)",
    )
    commit_sha: str | None = Field(
        default=None,
        description="Commit SHA attributed to this alert (Property 31)",
    )
    recommendations: list[str] = Field(default_factory=list)
    dedup_key: str | None = Field(
        default=None,
        description="Deduplication key: hash of (metric + component + 5-min window)",
    )

    @property
    def delivery_latency_seconds(self) -> float | None:
        if self.delivered_at:
            return (self.delivered_at - self.detected_at).total_seconds()
        return None
