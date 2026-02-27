"""
Agent Communication Protocol

Defines the structured data models for inter-agent communication.
All agents must return AgentResponse following this protocol.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
from datetime import datetime


class AgentType(str, Enum):
    """Types of agents in the system"""
    PLANNER = "planner"
    OBSERVABILITY = "observability"
    INFRA = "infra"
    DB = "db"
    COST = "cost"
    CICD = "cicd"


class TaskStatus(str, Enum):
    """Task execution status"""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"


class Severity(str, Enum):
    """Severity levels for insights and alerts"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Insight:
    """Semantic insight with business context"""
    summary: str
    business_impact: str
    severity: Severity
    recommendations: List[str]
    cost_impact_inr: float = 0.0
    confidence_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        """Validate insight data"""
        if not self.summary:
            raise ValueError("Insight summary cannot be empty")
        if not self.recommendations:
            raise ValueError("Insight must include at least one recommendation")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")


@dataclass
class AgentResponse:
    """
    Structured response from any agent.
    This is the mandatory communication protocol.
    """
    agent_type: AgentType
    task_id: str
    status: TaskStatus
    insights: List[Insight]
    data: Dict[str, Any]
    execution_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error_message: str = ""
    
    def __post_init__(self):
        """Validate agent response"""
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")
        if self.execution_time_ms < 0:
            raise ValueError("Execution time cannot be negative")
        if self.status == TaskStatus.SUCCESS and not self.insights:
            raise ValueError("Successful response must include insights")
