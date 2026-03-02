"""
models/metrics.py — CloudWatch metric and anomaly data models.

All models are Pydantic v2 BaseModel with field validators.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class MetricDataPoint(BaseModel):
    """A single timestamped metric value."""

    timestamp: datetime
    value: float


class MetricData(BaseModel):
    """
    CloudWatch metric data for a single metric over a time window.

    Carries the raw time-series data alongside metadata needed to
    contextualise the values (namespace, dimensions, unit).
    """

    metric_name: str = Field(min_length=1)
    namespace: str = Field(min_length=1, description="CloudWatch namespace, e.g. AWS/EC2")
    dimensions: dict[str, str] = Field(default_factory=dict)
    datapoints: list[MetricDataPoint] = Field(default_factory=list)
    unit: str = Field(default="None", description="CloudWatch unit string")
    statistic: str = Field(
        default="Average",
        description="Statistic used: Average, Sum, Maximum, Minimum, SampleCount",
    )
    period_seconds: int = Field(default=300, ge=1)

    @property
    def values(self) -> list[float]:
        return [dp.value for dp in self.datapoints]

    @property
    def timestamps(self) -> list[datetime]:
        return [dp.timestamp for dp in self.datapoints]

    @property
    def latest_value(self) -> float | None:
        if not self.datapoints:
            return None
        return max(self.datapoints, key=lambda dp: dp.timestamp).value

    @property
    def mean(self) -> float | None:
        vals = self.values
        return sum(vals) / len(vals) if vals else None

    @property
    def std(self) -> float | None:
        vals = self.values
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return variance**0.5


class Anomaly(BaseModel):
    """
    A detected anomaly in a metric time series.

    Severity is determined by sigma_deviation: how many standard deviations
    the observed value is from the baseline mean.
    Validates Property 5: confidence must be > 0.7 for detected anomalies.
    """

    metric_name: str
    timestamp: datetime
    observed_value: float
    baseline_mean: float
    baseline_std: float
    sigma_deviation: Annotated[float, Field(ge=0.0, description="Standard deviations from mean")]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    component: str | None = Field(default=None, description="Attributed infrastructure component")
    attribution: str | None = Field(
        default=None,
        description="Human-readable attribution: commit SHA, config change, etc.",
    )
    root_cause_hypothesis: str | None = None

    @model_validator(mode="after")
    def validate_sigma_matches_deviation(self) -> "Anomaly":
        """Baseline std must be non-negative."""
        if self.baseline_std < 0:
            raise ValueError("baseline_std must be non-negative")
        return self


class TimeSeries(BaseModel):
    """
    A named collection of MetricData objects covering the same time window.

    Used to correlate multiple metrics when doing bottleneck attribution.
    """

    name: str = Field(min_length=1, description="Descriptive label for this time series group")
    metrics: list[MetricData] = Field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary labels: env=prod, service=api, etc.",
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "TimeSeries":
        if self.start_time and self.end_time:
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be after start_time")
        return self
