"""
integrations/aws/billing.py — AWS Cost Explorer client.

Provides:
  get_cost_by_service(start, end)        -> dict[str, float]  (USD per service)
  get_cost_by_resource(start, end, tag)  -> dict[str, float]  (USD per resource tag)
  get_daily_costs(lookback_days)         -> list[dict]         (daily total USD)
  get_rightsizing_recommendations()      -> list[dict]         (raw AWS recommendations)

All monetary amounts are returned as USD floats.  Conversion to INR is
done in the Cost Agent using settings.usd_to_inr_rate — never here.

Raises BillingError on API failures, ThrottlingError on throttling.
"""

from __future__ import annotations

import asyncio
import csv
import io
from datetime import date, datetime, timedelta
from functools import partial
from typing import Any

import boto3
from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import BillingError, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry

logger = get_logger(__name__)


def _make_client(service: str) -> Any:
    kwargs: dict[str, Any] = {"region_name": "us-east-1"}  # CE only works in us-east-1
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client(service, **kwargs)
    return boto3.client(service, **kwargs)


class BillingClient:
    """
    Async wrapper around AWS Cost Explorer.

    Note: Cost Explorer data has a ~24-hour lag and is billed per API call.
    Consumers should cache results where possible.
    """

    def __init__(self) -> None:
        self._ce = _make_client("ce")

    # ── Sync helpers ───────────────────────────────────────────────────────

    def _get_cost_and_usage_sync(
        self,
        start: str,  # YYYY-MM-DD
        end: str,    # YYYY-MM-DD
        granularity: str,
        group_by: list[dict[str, str]],
        filter_expr: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Paginated Cost Explorer GetCostAndUsage call."""
        kwargs: dict[str, Any] = {
            "TimePeriod": {"Start": start, "End": end},
            "Granularity": granularity,
            "Metrics": ["UnblendedCost"],
            "GroupBy": group_by,
        }
        if filter_expr:
            kwargs["Filter"] = filter_expr

        results: list[dict[str, Any]] = []
        next_token: str | None = None

        while True:
            if next_token:
                kwargs["NextPageToken"] = next_token
            try:
                response = self._ce.get_cost_and_usage(**kwargs)
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if "Throttl" in code or "TooMany" in code or "LimitExceeded" in code:
                    raise ThrottlingError(f"Cost Explorer throttled: {e}") from e
                raise BillingError(f"Cost Explorer GetCostAndUsage failed: {e}") from e

            results.extend(response.get("ResultsByTime", []))
            next_token = response.get("NextPageToken")
            if not next_token:
                break

        return results

    def _get_rightsizing_sync(self) -> list[dict[str, Any]]:
        """Fetch AWS Compute Optimizer / CE right-sizing recommendations."""
        try:
            response = self._ce.get_rightsizing_recommendation(
                Service="AmazonEC2",
                Configuration={"RecommendationTarget": "SAME_INSTANCE_FAMILY"},
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"Cost Explorer rightsizing throttled: {e}") from e
            raise BillingError(
                f"Cost Explorer get_rightsizing_recommendation failed: {e}"
            ) from e

        return response.get("RightsizingRecommendations", [])

    # ── Public async API ───────────────────────────────────────────────────

    @with_retry(retry_on=(ThrottlingError,))
    async def get_cost_by_service(
        self, lookback_days: int = 30
    ) -> dict[str, float]:
        """
        Return total USD cost per AWS service for the last `lookback_days`.

        Returns:
            dict mapping service name → total USD cost (e.g. {"Amazon EC2": 250.40})
        """
        end = date.today()
        start = end - timedelta(days=lookback_days)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            partial(
                self._get_cost_and_usage_sync,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                "MONTHLY",
                [{"Type": "DIMENSION", "Key": "SERVICE"}],
            ),
        )

        totals: dict[str, float] = {}
        for period in results:
            for group in period.get("Groups", []):
                service = group["Keys"][0]
                amount = float(
                    group["Metrics"]["UnblendedCost"]["Amount"]
                )
                totals[service] = totals.get(service, 0.0) + amount

        return totals

    @with_retry(retry_on=(ThrottlingError,))
    async def get_daily_costs(
        self, lookback_days: int = 30
    ) -> list[dict[str, Any]]:
        """
        Return daily total USD cost for the last `lookback_days`.

        Returns:
            List of dicts: [{"date": "2025-01-01", "total_usd": 12.34}, ...]
        """
        end = date.today()
        start = end - timedelta(days=lookback_days)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            partial(
                self._get_cost_and_usage_sync,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                "DAILY",
                [],  # no grouping — just daily totals
            ),
        )

        daily: list[dict[str, Any]] = []
        for period in results:
            total = sum(
                float(m["Amount"])
                for m in period.get("Total", {}).values()
            )
            daily.append(
                {
                    "date": period["TimePeriod"]["Start"],
                    "total_usd": total,
                }
            )

        return sorted(daily, key=lambda x: x["date"])

    @with_retry(retry_on=(ThrottlingError,))
    async def get_cost_by_tag(
        self, tag_key: str, lookback_days: int = 30
    ) -> dict[str, float]:
        """
        Return USD cost grouped by a specific resource tag (e.g. "Environment").

        Returns:
            dict mapping tag value → total USD cost.
        """
        end = date.today()
        start = end - timedelta(days=lookback_days)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            partial(
                self._get_cost_and_usage_sync,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                "MONTHLY",
                [{"Type": "TAG", "Key": tag_key}],
            ),
        )

        totals: dict[str, float] = {}
        for period in results:
            for group in period.get("Groups", []):
                tag_value = group["Keys"][0].removeprefix(f"{tag_key}$")
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                totals[tag_value] = totals.get(tag_value, 0.0) + amount

        return totals

    @with_retry(retry_on=(ThrottlingError,))
    async def get_rightsizing_recommendations(self) -> list[dict[str, Any]]:
        """
        Fetch EC2 right-sizing recommendations from Cost Explorer.

        Returns raw recommendation dicts from AWS for the Cost Agent
        to interpret and convert into INR savings estimates.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_rightsizing_sync)

    @staticmethod
    def parse_billing_csv(csv_content: str) -> list[dict[str, str]]:
        """
        Parse an AWS detailed billing CSV (from S3) into a list of row dicts.

        Used by KnowledgeBase.index_metrics() to store billing data for RAG.
        This is a synchronous utility — no network call.

        Args:
            csv_content: Raw CSV string from the billing file.

        Returns:
            List of dicts, one per billing line item.
        """
        reader = csv.DictReader(io.StringIO(csv_content))
        return [row for row in reader]


# Module-level singleton
billing_client = BillingClient()
