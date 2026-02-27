"""
Core Data Models

Defines structured data models for infrastructure telemetry,
analysis results, and agent inputs/outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class MetricType(str, Enum):
    """Types of metrics"""
    CPU = "cpu"
    MEMORY = "memory"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    DISK_IO = "disk_io"
    NETWORK = "network"
    CUSTOM = "custom"


@dataclass
class MetricData:
    """Infrastructure metric data point"""
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: str
    dimensions: Dict[str, str] = field(default_factory=dict)
    source: str = "cloudwatch"
    
    def __post_init__(self):
        """Validate metric data"""
        if not self.metric_name:
            raise ValueError("Metric name cannot be empty")
        if self.value < 0 and self.metric_type != MetricType.CUSTOM:
            raise ValueError("Metric value cannot be negative")


@dataclass
class Anomaly:
    """Detected anomaly in metrics"""
    metric_name: str
    expected_value: float
    observed_value: float
    deviation_sigma: float
    confidence_score: float
    timestamp: str
    context: str = ""
    
    def __post_init__(self):
        """Validate anomaly data"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")
        if self.deviation_sigma < 0:
            raise ValueError("Deviation sigma cannot be negative")


@dataclass
class Recommendation:
    """Actionable recommendation"""
    title: str
    description: str
    action_steps: List[str]
    priority: int  # 1 (highest) to 5 (lowest)
    cost_impact_monthly_inr: float = 0.0
    cost_impact_annual_inr: float = 0.0
    expected_benefit: str = ""
    risk_level: str = "LOW"
    
    def __post_init__(self):
        """Validate recommendation"""
        if not self.title:
            raise ValueError("Recommendation title cannot be empty")
        if not self.action_steps:
            raise ValueError("Recommendation must include action steps")
        if not 1 <= self.priority <= 5:
            raise ValueError("Priority must be between 1 and 5")
        if self.cost_impact_monthly_inr < 0:
            raise ValueError("Cost impact cannot be negative")


@dataclass
class CostImpact:
    """Cost impact analysis in INR"""
    current_monthly_cost_inr: float
    projected_monthly_cost_inr: float
    savings_monthly_inr: float
    savings_annual_inr: float
    confidence_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and calculate cost impact"""
        if self.current_monthly_cost_inr < 0 or self.projected_monthly_cost_inr < 0:
            raise ValueError("Cost values cannot be negative")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1")
        
        # Auto-calculate savings if not provided
        if self.savings_monthly_inr == 0:
            self.savings_monthly_inr = self.current_monthly_cost_inr - self.projected_monthly_cost_inr
        if self.savings_annual_inr == 0:
            self.savings_annual_inr = self.savings_monthly_inr * 12


@dataclass
class Configuration:
    """Infrastructure configuration snapshot"""
    config_type: str  # e.g., "dockerfile", "ecs_task", "docker_compose"
    config_content: str
    config_hash: str
    timestamp: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.config_content:
            raise ValueError("Configuration content cannot be empty")


@dataclass
class BuildData:
    """CI/CD build information"""
    build_id: str
    commit_sha: str
    build_time_seconds: float
    status: str  # "success", "failure", "in_progress"
    timestamp: str
    repository: str
    branch: str
    triggering_event: str = "push"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate build data"""
        if not self.build_id:
            raise ValueError("Build ID cannot be empty")
        if self.build_time_seconds < 0:
            raise ValueError("Build time cannot be negative")


@dataclass
class QueryPattern:
    """Database query pattern analysis"""
    query_template: str
    execution_count: int
    avg_duration_ms: float
    max_duration_ms: float
    tables_accessed: List[str]
    missing_indices: List[str] = field(default_factory=list)
    recommended_indices: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate query pattern"""
        if not self.query_template:
            raise ValueError("Query template cannot be empty")
        if self.execution_count < 0:
            raise ValueError("Execution count cannot be negative")
        if self.avg_duration_ms < 0 or self.max_duration_ms < 0:
            raise ValueError("Duration cannot be negative")


@dataclass
class Task:
    """Task for agent execution"""
    task_id: str
    task_type: str
    description: str
    priority: int
    assigned_agent: str
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        """Validate task"""
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")
        if not 1 <= self.priority <= 5:
            raise ValueError("Priority must be between 1 and 5")
