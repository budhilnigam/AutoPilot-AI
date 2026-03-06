"""
services/metrics_service.py — CloudWatch Metrics Fetching Service.

Translates natural language queries and agent tasks into concrete CloudWatch
metric fetches. Handles the gap between "why is checkout slow?" and specific
API calls like get_metric_statistics(namespace="AWS/ECS", metric="CPUUtilization").

Usage:
    metrics = await metrics_service.fetch_metrics_for_query(
        query="why is checkout slow?",
        context={"component": "checkout-service"}
    )
    # Returns list[MetricData] ready for ObservabilityAgent
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.aws.cloudwatch import cloudwatch_client
from autopilot_ai.models.metrics import MetricData

logger = get_logger(__name__)


# Default metric configurations for common infrastructure components
DEFAULT_METRIC_CONFIGS = {
    "ecs": [
        {"namespace": "AWS/ECS", "metric_name": "CPUUtilization", "statistic": "Average"},
        {"namespace": "AWS/ECS", "metric_name": "MemoryUtilization", "statistic": "Average"},
    ],
    "alb": [
        {"namespace": "AWS/ApplicationELB", "metric_name": "TargetResponseTime", "statistic": "Average"},
        {"namespace": "AWS/ApplicationELB", "metric_name": "HTTPCode_Target_5XX_Count", "statistic": "Sum"},
        {"namespace": "AWS/ApplicationELB", "metric_name": "RequestCount", "statistic": "Sum"},
    ],
    "rds": [
        {"namespace": "AWS/RDS", "metric_name": "CPUUtilization", "statistic": "Average"},
        {"namespace": "AWS/RDS", "metric_name": "DatabaseConnections", "statistic": "Average"},
        {"namespace": "AWS/RDS", "metric_name": "ReadLatency", "statistic": "Average"},
        {"namespace": "AWS/RDS", "metric_name": "WriteLatency", "statistic": "Average"},
    ],
    "elasticache": [
        {"namespace": "AWS/ElastiCache", "metric_name": "CPUUtilization", "statistic": "Average"},
        {"namespace": "AWS/ElastiCache", "metric_name": "DatabaseMemoryUsagePercentage", "statistic": "Average"},
        {"namespace": "AWS/ElastiCache", "metric_name": "CurrConnections", "statistic": "Average"},
    ],
    "lambda": [
        {"namespace": "AWS/Lambda", "metric_name": "Duration", "statistic": "Average"},
        {"namespace": "AWS/Lambda", "metric_name": "Errors", "statistic": "Sum"},
        {"namespace": "AWS/Lambda", "metric_name": "Throttles", "statistic": "Sum"},
    ],
}


class MetricsService:
    """
    Service for fetching CloudWatch metrics based on queries and context.
    
    Bridges the gap between natural language queries and concrete CloudWatch API calls.
    """

    def __init__(self) -> None:
        self._client = cloudwatch_client

    async def fetch_metrics_for_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        lookback_minutes: int = 60,
    ) -> list[MetricData]:
        """
        Fetch relevant CloudWatch metrics based on a natural language query.
        
        Args:
            query: Natural language query (e.g., "why is checkout slow?")
            context: Optional context with hints like component, service_name, etc.
            lookback_minutes: How far back to fetch metrics (default 60 minutes)
        
        Returns:
            List of MetricData objects ready for analysis
        """
        context = context or {}
        logger.info(
            "metrics_fetch_start",
            query=query[:100],
            lookback_minutes=lookback_minutes,
            context_keys=list(context.keys()),
        )

        # Determine which metrics to fetch based on query keywords and context
        metric_configs = self._select_metrics(query, context)
        
        if not metric_configs:
            logger.warning("metrics_fetch_no_configs", query=query[:100])
            return []

        # Fetch metrics in parallel
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        tasks = [
            self._fetch_single_metric(
                config=config,
                start_time=start_time,
                end_time=end_time,
                context=context,
            )
            for config in metric_configs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures and None results
        metrics = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("metrics_fetch_error", error=str(result))
            elif result is not None:
                metrics.append(result)
        
        logger.info("metrics_fetch_complete", count=len(metrics), requested=len(metric_configs))
        return metrics

    def _select_metrics(
        self,
        query: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Select which metrics to fetch based on query keywords and context.
        
        Returns:
            List of metric config dicts with namespace, metric_name, statistic, dimensions
        """
        query_lower = query.lower()
        configs = []
        
        # Check for explicit component in context
        component = context.get("component", "").lower()
        service_name = context.get("service_name", "")
        
        # Parse query for infrastructure keywords
        if any(kw in query_lower for kw in ["slow", "latency", "response", "timeout"]):
            # Performance issue - fetch ALB and service metrics
            configs.extend(DEFAULT_METRIC_CONFIGS["alb"])
            configs.extend(DEFAULT_METRIC_CONFIGS["ecs"])
            
        if any(kw in query_lower for kw in ["cpu", "memory", "resource"]):
            # Resource issue - fetch all infrastructure metrics
            configs.extend(DEFAULT_METRIC_CONFIGS["ecs"])
            if "database" in query_lower or "db" in query_lower or component == "rds":
                configs.extend(DEFAULT_METRIC_CONFIGS["rds"])
            if "redis" in query_lower or "cache" in query_lower or component == "elasticache":
                configs.extend(DEFAULT_METRIC_CONFIGS["elasticache"])
        
        if any(kw in query_lower for kw in ["error", "5xx", "failed", "failure"]):
            # Error investigation - fetch error metrics
            configs.extend(DEFAULT_METRIC_CONFIGS["alb"])
            configs.extend(DEFAULT_METRIC_CONFIGS["ecs"])
        
        if "database" in query_lower or "db" in query_lower or component == "rds":
            configs.extend(DEFAULT_METRIC_CONFIGS["rds"])
        
        if "redis" in query_lower or "cache" in query_lower or component == "elasticache":
            configs.extend(DEFAULT_METRIC_CONFIGS["elasticache"])
        
        if "lambda" in query_lower or component == "lambda":
            configs.extend(DEFAULT_METRIC_CONFIGS["lambda"])
        
        # Fallback: if no keywords matched, fetch general health metrics
        if not configs:
            logger.info("metrics_fetch_fallback", query=query[:100])
            configs.extend(DEFAULT_METRIC_CONFIGS["ecs"])
            configs.extend(DEFAULT_METRIC_CONFIGS["alb"])
        
        # Add dimensions from context if available
        for config in configs:
            if "dimensions" not in config:
                config["dimensions"] = {}
            
            # Add service name if provided
            if service_name:
                if config["namespace"] == "AWS/ECS":
                    config["dimensions"]["ServiceName"] = service_name
                elif config["namespace"] == "AWS/ApplicationELB":
                    config["dimensions"]["LoadBalancer"] = service_name
        
        # Deduplicate configs
        seen = set()
        unique_configs = []
        for config in configs:
            key = (
                config["namespace"],
                config["metric_name"],
                tuple(sorted(config.get("dimensions", {}).items())),
            )
            if key not in seen:
                seen.add(key)
                unique_configs.append(config)
        
        return unique_configs

    async def _fetch_single_metric(
        self,
        config: dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        context: dict[str, Any],
    ) -> MetricData | None:
        """
        Fetch a single metric from CloudWatch.
        
        Args:
            config: Metric configuration (namespace, metric_name, statistic, dimensions)
            start_time: Start of time range
            end_time: End of time range
            context: Additional context for dimension inference
        
        Returns:
            MetricData object or None if fetch failed
        """
        try:
            metric_data = await self._client.get_metric_statistics(
                namespace=config["namespace"],
                metric_name=config["metric_name"],
                dimensions=config.get("dimensions", {}),
                period_seconds=300,  # 5-minute periods
                lookback_minutes=int((end_time - start_time).total_seconds() / 60),
                statistic=config.get("statistic", "Average"),
            )
            
            if metric_data and metric_data.datapoints:
                logger.debug(
                    "metric_fetched",
                    metric=config["metric_name"],
                    namespace=config["namespace"],
                    points=len(metric_data.datapoints),
                )
                return metric_data
            else:
                logger.debug(
                    "metric_no_data",
                    metric=config["metric_name"],
                    namespace=config["namespace"],
                )
                return None
                
        except Exception as e:
            logger.error(
                "metric_fetch_failed",
                metric=config["metric_name"],
                namespace=config["namespace"],
                error=str(e),
            )
            return None

    async def fetch_metrics_for_component(
        self,
        component: str,
        lookback_minutes: int = 60,
    ) -> list[MetricData]:
        """
        Fetch all relevant metrics for a specific infrastructure component.
        
        Args:
            component: Component name (e.g., "ecs", "rds", "alb", "elasticache", "lambda")
            lookback_minutes: How far back to fetch metrics
        
        Returns:
            List of MetricData objects
        """
        component_lower = component.lower()
        
        if component_lower in DEFAULT_METRIC_CONFIGS:
            configs = DEFAULT_METRIC_CONFIGS[component_lower]
        else:
            logger.warning("metrics_unknown_component", component=component)
            return []
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        tasks = [
            self._fetch_single_metric(
                config=config.copy(),
                start_time=start_time,
                end_time=end_time,
                context={},
            )
            for config in configs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("component_metric_error", component=component, error=str(result))
            elif result is not None:
                metrics.append(result)
        
        return metrics


# Module-level singleton
metrics_service = MetricsService()
