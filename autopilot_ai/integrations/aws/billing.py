"""integrations/aws/billing.py — AWS Cost Explorer client using aws_api.

Routes Cost Explorer and related billing APIs through `aws_api` so boto3
usage stays centralised.
"""

from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from typing import Any

from botocore.exceptions import ClientError

from autopilot_ai.core.exceptions import BillingError, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.integrations.aws.tool import aws_api

logger = get_logger(__name__)


class BillingClient:
    def __init__(self) -> None:
        pass

    async def _get_cost_and_usage(self, start: str, end: str, granularity: str, group_by: list[dict[str, str]], filter_expr: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "TimePeriod": {"Start": start, "End": end},
            "Granularity": granularity,
            "Metrics": ["UnblendedCost"],
            "GroupBy": group_by,
        }
        if filter_expr:
            params["Filter"] = filter_expr

        results: list[dict[str, Any]] = []
        next_token = None
        try:
            while True:
                if next_token:
                    params["NextPageToken"] = next_token
                resp = await aws_api("ce", "get_cost_and_usage", params)
                results.extend(resp.get("ResultsByTime", []))
                next_token = resp.get("NextPageToken")
                if not next_token:
                    break
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code or "LimitExceeded" in code:
                raise ThrottlingError(f"Cost Explorer throttled: {e}") from e
            raise BillingError(f"Cost Explorer GetCostAndUsage failed: {e}") from e

        return results

    @with_retry(retry_on=(ThrottlingError,))
    async def get_cost_by_service(self, lookback_days: int = 30) -> dict[str, float]:
        end = date.today()
        start = end - timedelta(days=lookback_days)
        results = await self._get_cost_and_usage(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "MONTHLY", [{"Type": "DIMENSION", "Key": "SERVICE"}])

        totals: dict[str, float] = {}
        for period in results:
            for group in period.get("Groups", []):
                service = group["Keys"][0]
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                totals[service] = totals.get(service, 0.0) + amount

        return totals

    @with_retry(retry_on=(ThrottlingError,))
    async def get_daily_costs(self, lookback_days: int = 30) -> list[dict[str, Any]]:
        end = date.today()
        start = end - timedelta(days=lookback_days)
        results = await self._get_cost_and_usage(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "DAILY", [])

        daily: list[dict[str, Any]] = []
        for period in results:
            total = sum(float(m["Amount"]) for m in period.get("Total", {}).values())
            daily.append({"date": period["TimePeriod"]["Start"], "total_usd": total})

        return sorted(daily, key=lambda x: x["date"])

    @with_retry(retry_on=(ThrottlingError,))
    async def get_cost_by_tag(self, tag_key: str, lookback_days: int = 30) -> dict[str, float]:
        end = date.today()
        start = end - timedelta(days=lookback_days)
        results = await self._get_cost_and_usage(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "MONTHLY", [{"Type": "TAG", "Key": tag_key}])

        totals: dict[str, float] = {}
        for period in results:
            for group in period.get("Groups", []):
                tag_value = group["Keys"][0].removeprefix(f"{tag_key}$")
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                totals[tag_value] = totals.get(tag_value, 0.0) + amount

        return totals

    @with_retry(retry_on=(ThrottlingError,))
    async def get_rightsizing_recommendations(self) -> list[dict[str, Any]]:
        try:
            resp = await aws_api("ce", "get_rightsizing_recommendation", {"Service": "AmazonEC2", "Configuration": {"RecommendationTarget": "SAME_INSTANCE_FAMILY"}})
            return resp.get("RightsizingRecommendations", [])
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"Cost Explorer rightsizing throttled: {e}") from e
            raise BillingError(f"Cost Explorer get_rightsizing_recommendation failed: {e}") from e

    @staticmethod
    def parse_billing_csv(csv_content: str) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(csv_content))
        return [row for row in reader]


# Module-level singleton
billing_client = BillingClient()
