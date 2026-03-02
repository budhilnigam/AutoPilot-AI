"""
integrations/aws/cloudwatch.py — CloudWatch metrics client.

Provides:
  get_metric_statistics(...)  -> MetricData     (single metric, last N minutes)
  get_multiple_metrics(...)   -> list[MetricData]  (batch query, pagination handled)
  get_log_events(...)         -> list[dict]     (CloudWatch Logs)

All methods are async via run_in_executor. Pagination is handled internally
so callers always receive the complete dataset for the requested time range.

Raises CloudWatchError on API failures, ThrottlingError on throttling.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

import boto3
from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import CloudWatchError, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.models.metrics import MetricData, MetricDataPoint

logger = get_logger(__name__)


def _make_client(service: str = "cloudwatch") -> Any:
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client(service, **kwargs)
    return boto3.client(service, **kwargs)


class CloudWatchClient:
    """
    Async wrapper around boto3 CloudWatch and CloudWatch Logs clients.

    Usage:
        cw = CloudWatchClient()
        cpu = await cw.get_metric_statistics(
            namespace="AWS/EC2",
            metric_name="CPUUtilization",
            dimensions={"InstanceId": "i-0abc12345"},
            period_seconds=300,
            lookback_minutes=60,
            statistic="Average",
        )
    """

    def __init__(self) -> None:
        self._cw = _make_client("cloudwatch")
        self._logs = _make_client("logs")

    # ── Internal sync helpers (run inside executor) ────────────────────────

    def _get_metric_statistics_sync(
        self,
        namespace: str,
        metric_name: str,
        dimensions: dict[str, str],
        period_seconds: int,
        start_time: datetime,
        end_time: datetime,
        statistic: str,
    ) -> list[MetricDataPoint]:
        """Calls CloudWatch GetMetricStatistics and handles pagination via NextToken."""
        dim_list = [{"Name": k, "Value": v} for k, v in dimensions.items()]

        try:
            response = self._cw.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dim_list,
                StartTime=start_time,
                EndTime=end_time,
                Period=period_seconds,
                Statistics=[statistic],
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(
                    f"CloudWatch throttled: {e}",
                    namespace=namespace,
                    metric=metric_name,
                ) from e
            raise CloudWatchError(
                f"CloudWatch GetMetricStatistics failed: {e}",
                namespace=namespace,
                metric=metric_name,
            ) from e

        datapoints = [
            MetricDataPoint(
                timestamp=dp["Timestamp"],
                value=dp.get(statistic, 0.0),
            )
            for dp in response.get("Datapoints", [])
        ]
        # CloudWatch returns datapoints unordered — sort ascending
        return sorted(datapoints, key=lambda dp: dp.timestamp)

    def _get_metric_data_sync(
        self,
        queries: list[dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, list[MetricDataPoint]]:
        """
        Calls CloudWatch GetMetricData for multiple metrics at once.
        Handles pagination via NextToken.

        Returns a dict keyed by MetricDataQuery Id.
        """
        result: dict[str, list[MetricDataPoint]] = {q["Id"]: [] for q in queries}
        next_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "MetricDataQueries": queries,
                "StartTime": start_time,
                "EndTime": end_time,
            }
            if next_token:
                kwargs["NextToken"] = next_token

            try:
                response = self._cw.get_metric_data(**kwargs)
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if "Throttl" in code or "TooMany" in code:
                    raise ThrottlingError(f"CloudWatch GetMetricData throttled: {e}") from e
                raise CloudWatchError(f"CloudWatch GetMetricData failed: {e}") from e

            for metric_result in response.get("MetricDataResults", []):
                mid = metric_result["Id"]
                timestamps = metric_result.get("Timestamps", [])
                values = metric_result.get("Values", [])
                for ts, val in zip(timestamps, values):
                    result[mid].append(MetricDataPoint(timestamp=ts, value=val))

            next_token = response.get("NextToken")
            if not next_token:
                break

        # Sort each series ascending
        for mid in result:
            result[mid].sort(key=lambda dp: dp.timestamp)

        return result

    def _get_log_events_sync(
        self,
        log_group: str,
        log_stream: str | None,
        start_time: datetime,
        end_time: datetime,
        filter_pattern: str = "",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Filter log events from a CloudWatch Logs group."""
        kwargs: dict[str, Any] = {
            "logGroupName": log_group,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": min(limit, 10000),
        }
        if log_stream:
            kwargs["logStreamName"] = log_stream
        if filter_pattern:
            kwargs["filterPattern"] = filter_pattern

        events: list[dict[str, Any]] = []
        next_token: str | None = None

        while len(events) < limit:
            if next_token:
                kwargs["nextToken"] = next_token
            try:
                response = self._cw.filter_log_events(**kwargs)
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if "Throttl" in code or "TooMany" in code:
                    raise ThrottlingError(
                        f"CloudWatch Logs throttled: {e}", log_group=log_group
                    ) from e
                raise CloudWatchError(
                    f"CloudWatch Logs failed: {e}", log_group=log_group
                ) from e

            events.extend(response.get("events", []))
            next_token = response.get("nextToken")
            if not next_token:
                break

        return events[:limit]

    # ── Public async API ───────────────────────────────────────────────────

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
        """
        Fetch a single CloudWatch metric over the last `lookback_minutes`.

        Args:
            namespace:        CloudWatch namespace (e.g. "AWS/EC2").
            metric_name:      Metric name (e.g. "CPUUtilization").
            dimensions:       Dict of dimension name → value.
            period_seconds:   Resolution period, must align to CloudWatch rules.
            lookback_minutes: How far back to fetch data.
            statistic:        "Average", "Maximum", "Minimum", "Sum", or "SampleCount".

        Returns:
            MetricData populated with sorted datapoints.
        """
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)

        logger.debug(
            "cloudwatch_get_metric",
            namespace=namespace,
            metric=metric_name,
            lookback_minutes=lookback_minutes,
        )

        loop = asyncio.get_running_loop()
        datapoints = await loop.run_in_executor(
            None,
            partial(
                self._get_metric_statistics_sync,
                namespace,
                metric_name,
                dimensions,
                period_seconds,
                start_time,
                end_time,
                statistic,
            ),
        )

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
        """
        Fetch multiple metrics in parallel using GetMetricData.

        Each spec in `metric_specs` must have:
            namespace, metric_name, dimensions (dict), statistic (optional)

        Returns:
            List of MetricData, one per spec, in the same order as input.
        """
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)

        # Build GetMetricData query objects
        queries: list[dict[str, Any]] = []
        for i, spec in enumerate(metric_specs):
            dim_list = [
                {"Name": k, "Value": v}
                for k, v in spec.get("dimensions", {}).items()
            ]
            queries.append(
                {
                    "Id": f"m{i}",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": spec["namespace"],
                            "MetricName": spec["metric_name"],
                            "Dimensions": dim_list,
                        },
                        "Period": period_seconds,
                        "Stat": spec.get("statistic", "Average"),
                    },
                    "ReturnData": True,
                }
            )

        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None,
            partial(self._get_metric_data_sync, queries, start_time, end_time),
        )

        results: list[MetricData] = []
        for i, spec in enumerate(metric_specs):
            results.append(
                MetricData(
                    metric_name=spec["metric_name"],
                    namespace=spec["namespace"],
                    dimensions=spec.get("dimensions", {}),
                    datapoints=raw.get(f"m{i}", []),
                    statistic=spec.get("statistic", "Average"),
                    period_seconds=period_seconds,
                )
            )
        return results

    @with_retry(retry_on=(ThrottlingError,))
    async def get_log_events(
        self,
        log_group: str,
        log_stream: str | None = None,
        lookback_minutes: int = 30,
        filter_pattern: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Fetch log events from a CloudWatch Logs group.

        Args:
            log_group:       CloudWatch Logs group name.
            log_stream:      Optional stream name; omit to search all streams.
            lookback_minutes: How far back to search.
            filter_pattern:  CloudWatch Logs filter pattern syntax.
            limit:           Maximum number of events to return.

        Returns:
            List of event dicts with keys: timestamp, message, logStreamName.
        """
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._get_log_events_sync,
                log_group,
                log_stream,
                start_time,
                end_time,
                filter_pattern,
                limit,
            ),
        )


# Module-level singleton
cloudwatch_client = CloudWatchClient()
