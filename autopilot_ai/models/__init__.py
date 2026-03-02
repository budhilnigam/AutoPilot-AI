"""autopilot_ai.models — Pydantic v2 data models for the multi-agent system."""

from autopilot_ai.models.domain import (
    Alert,
    AlertSeverity,
    BuildData,
    BuildStatus,
    Configuration,
    ConfigType,
    ECSServiceConfig,
    QueryPattern,
    RedisStats,
    WorkerConfig,
)
from autopilot_ai.models.insights import (
    CostImpact,
    ImplementationEffort,
    Insight,
    InsightCategory,
    Recommendation,
    Urgency,
)
from autopilot_ai.models.metrics import Anomaly, MetricData, MetricDataPoint, TimeSeries
from autopilot_ai.models.responses import AgentResponse, ErrorResponse, QueryResponse, ResponseStatus
from autopilot_ai.models.tasks import AgentType, Priority, Task, TaskType

__all__ = [
    # metrics
    "MetricDataPoint",
    "MetricData",
    "Anomaly",
    "TimeSeries",
    # insights
    "Urgency",
    "InsightCategory",
    "ImplementationEffort",
    "CostImpact",
    "Recommendation",
    "Insight",
    # tasks
    "AgentType",
    "TaskType",
    "Priority",
    "Task",
    # responses
    "ResponseStatus",
    "AgentResponse",
    "ErrorResponse",
    "QueryResponse",
    # domain
    "BuildStatus",
    "BuildData",
    "QueryPattern",
    "RedisStats",
    "ConfigType",
    "Configuration",
    "WorkerConfig",
    "ECSServiceConfig",
    "AlertSeverity",
    "Alert",
]
