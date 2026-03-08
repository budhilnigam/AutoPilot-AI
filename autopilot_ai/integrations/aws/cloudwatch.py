"""integrations/aws/cloudwatch.py — CloudWatch metrics client using aws_api.

This implementation centralises boto3 usage in `autopilot_ai.integrations.aws.tool`.
All methods are async and use `aws_api` for API calls. Pagination is handled
internally so callers receive complete datasets.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from botocore.exceptions import ClientError

from autopilot_ai.core.exceptions import CloudWatchError, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.integrations.aws.tool import aws_api
from autopilot_ai.models.metrics import MetricData, MetricDataPoint

logger = get_logger(__name__)


class CloudWatchClient:
    """Async CloudWatch helper using the shared aws_api shim."""

    def __init__(self) -> None:
        pass

    @with_retry(retry_on=(ThrottlingError,))
    async def get_metric_statistics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: dict[str, str],
        period_seconds: int = 300,
        lookback_minutes: int = 60,
        statistic: str = "Average",
    ) -> MetricData:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)

        logger.debug(
            "cloudwatch_get_metric",
            namespace=namespace,
            metric=metric_name,
            lookback_minutes=lookback_minutes,
        )

        dim_list = [{"Name": k, "Value": v} for k, v in dimensions.items()]
        params = {
            "Namespace": namespace,
            "MetricName": metric_name,
            "Dimensions": dim_list,
            "StartTime": start_time,
            "EndTime": end_time,
            "Period": period_seconds,
            "Statistics": [statistic],
        }

        try:
            resp = await aws_api("cloudwatch", "get_metric_statistics", params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"CloudWatch throttled: {e}") from e
            raise CloudWatchError(f"CloudWatch GetMetricStatistics failed: {e}") from e

        datapoints = [
            MetricDataPoint(timestamp=dp["Timestamp"], value=dp.get(statistic, 0.0))
            for dp in resp.get("Datapoints", [])
        ]
        datapoints.sort(key=lambda dp: dp.timestamp)

        return MetricData(
            metric_name=metric_name,
            namespace=namespace,
            dimensions=dimensions,
            datapoints=datapoints,
            statistic=statistic,
            period_seconds=period_seconds,
        )

    @with_retry(retry_on=(ThrottlingError,))
    async def get_multiple_metrics(
        self,
        metric_specs: list[dict[str, Any]],
        lookback_minutes: int = 60,
        period_seconds: int = 300,
    ) -> list[MetricData]:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)

        queries: list[dict[str, Any]] = []
        for i, spec in enumerate(metric_specs):
            queries.append(
                {
                    "Id": f"m{i}",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": spec["namespace"],
                            "MetricName": spec["metric_name"],
                        },
                        "Period": period_seconds,
                        "Stat": spec.get("statistic", "Average"),
                    },
                    "ReturnData": True,
                }
            )

        params = {"MetricDataQueries": queries, "StartTime": start_time, "EndTime": end_time}

        result_map: dict[str, list[MetricDataPoint]] = {q["Id"]: [] for q in queries}
        next_token = None
        try:
            while True:
                if next_token:
                    params["NextToken"] = next_token
                resp = await aws_api("cloudwatch", "get_metric_data", params)
                for metric_result in resp.get("MetricDataResults", []):
                    mid = metric_result["Id"]
                    timestamps = metric_result.get("Timestamps", [])
                    values = metric_result.get("Values", [])
                    for ts, val in zip(timestamps, values):
                        result_map[mid].append(MetricDataPoint(timestamp=ts, value=val))
                next_token = resp.get("NextToken")
                if not next_token:
                    break
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"CloudWatch GetMetricData throttled: {e}") from e
            raise CloudWatchError(f"CloudWatch GetMetricData failed: {e}") from e

        for mid in result_map:
            result_map[mid].sort(key=lambda dp: dp.timestamp)

        out: list[MetricData] = []
        for i, spec in enumerate(metric_specs):
            series = result_map.get(f"m{i}", [])
            out.append(
                MetricData(
                    metric_name=spec["metric_name"],
                    namespace=spec["namespace"],
                    dimensions=spec.get("dimensions", {}),
                    datapoints=series,
                    statistic=spec.get("statistic", "Average"),
                    period_seconds=period_seconds,
                )
            )

        return out

    @with_retry(retry_on=(ThrottlingError,))
    async def get_log_events(
        self,
        log_group: str,
        log_stream: str | None = None,
        lookback_minutes: int = 30,
        filter_pattern: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)

        params = {
            "logGroupName": log_group,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": min(limit, 10000),
        }
        if log_stream:
            params["logStreamName"] = log_stream
        if filter_pattern:
            params["filterPattern"] = filter_pattern

        events: list[dict[str, Any]] = []
        next_token = None
        try:
            while len(events) < limit:
                if next_token:
                    params["nextToken"] = next_token
                resp = await aws_api("logs", "filter_log_events", params)
                events.extend(resp.get("events", []))
                next_token = resp.get("nextToken")
                if not next_token:
                    break
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"CloudWatch Logs throttled: {e}") from e
            raise CloudWatchError(f"CloudWatch Logs failed: {e}") from e

        return events[:limit]


# Module-level singleton
cloudwatch_client = CloudWatchClient()
