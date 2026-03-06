"""
services/ — Supporting business-logic services.

Exports module-level singletons used by the API layer and agents.
"""

from autopilot_ai.services.alerting import alerting_service, AlertingService
from autopilot_ai.services.metrics_service import metrics_service, MetricsService
from autopilot_ai.services.prediction import prediction_service, PredictionService, SaturationPrediction, RankedInsight
from autopilot_ai.services.github_poller import github_poller, GitHubPoller

__all__ = [
    "alerting_service",
    "AlertingService",
    "metrics_service",
    "MetricsService",
    "prediction_service",
    "PredictionService",
    "SaturationPrediction",
    "RankedInsight",
    "github_poller",
    "GitHubPoller",
]
