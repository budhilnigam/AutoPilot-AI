"""
agents/cost.py — Cost Optimisation Agent.

Handles:
  ANALYZE_COSTS              → break down current spend by service
  IDENTIFY_OPTIMIZATION      → flag resources with <30% utilisation over 7 days
  FORECAST_COSTS             → extrapolate spend trend and forecast next 30 days

Task parameters:
  ANALYZE_COSTS:
    cost_by_service: dict[str, float]   — USD spend per service name
    daily_costs: list[dict]             — [{date, amount}] in USD
    period_days: int                    — lookback period

  IDENTIFY_OPTIMIZATION:
    cost_by_service: dict[str, float]   — USD spend per service
    utilisation: dict[str, float]       — service → utilisation pct (0-100)
    rightsizing: list[dict]             — from billing_client.get_rightsizing_recommendations()

  FORECAST_COSTS:
    daily_costs: list[dict]             — [{date, amount}] in USD

ALL monetary values are surfaced in INR (multiplied by settings.usd_to_inr_rate).
Validates Properties 14, 15, 16.
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import date, timedelta

from autopilot_ai.agents.base import BaseAgent
from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.aws.bedrock import bedrock_client
from autopilot_ai.models.insights import (
    CostImpact,
    Insight,
    InsightCategory,
    Recommendation,
    Urgency,
)
from autopilot_ai.models.responses import AgentResponse
from autopilot_ai.models.tasks import AgentType, Task, TaskType
from autopilot_ai.services.knowledge_base import knowledge_base

logger = get_logger(__name__)


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    return re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)


def _usd_to_inr(usd: float) -> float:
    return round(usd * settings.usd_to_inr_rate, 2)


def _parse_insights(raw: str) -> list[Insight]:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        logger.warning("cost_agent_json_parse_error", raw=raw[:200])
        return []

    urgency_map = {u.value: u for u in Urgency}
    results: list[Insight] = []

    for item in data.get("insights", []):
        try:
            # Build CostImpact if provided
            cost_impact: CostImpact | None = None
            ci = item.get("cost_impact")
            if ci:
                monthly_inr = float(ci.get("monthly_inr", 0))
                if monthly_inr > 0:
                    cost_impact = CostImpact(
                        monthly_inr=monthly_inr,
                        annual_inr=round(monthly_inr * 12, 2),
                        description=ci.get("description", ""),
                    )

            recs = [
                Recommendation(
                    action=r.get("action", "Review"),
                    rationale=r.get("rationale", "See details"),
                    steps=r.get("steps", ["Investigate"]),
                    expected_benefit=r.get("expected_benefit", "Cost reduction"),
                )
                for r in item.get("recommendations", [])
            ] or [Recommendation(
                action="Review resource utilisation",
                rationale="Cost optimisation opportunity detected",
                steps=["Review resource config"],
                expected_benefit="Reduced cloud spend",
            )]

            results.append(Insight(
                category=InsightCategory.COST,
                component=item.get("component", "aws"),
                title=item.get("title", "Cost finding"),
                business_context=item.get("business_context", ""),
                urgency=urgency_map.get(item.get("urgency", "medium"), Urgency.MEDIUM),
                confidence=float(item.get("confidence", 0.7)),
                recommendations=recs,
                cost_impact=cost_impact,
                attribution=item.get("attribution"),
            ))
        except Exception as e:
            logger.warning("cost_insight_parse_error", error=str(e))

    return results


def _simple_linear_forecast(daily_amounts: list[float], days_ahead: int = 30) -> float:
    """
    Fit a simple linear trend over the last N days and extrapolate.
    Returns total forecasted spend for the next `days_ahead` days.
    """
    n = len(daily_amounts)
    if n < 3:
        return sum(daily_amounts) / n * days_ahead if n > 0 else 0.0

    x_mean = (n - 1) / 2
    y_mean = statistics.mean(daily_amounts)
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(daily_amounts))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0.0
    intercept = y_mean - slope * x_mean

    # Forecast from day n onward
    future_daily = [intercept + slope * (n + i) for i in range(days_ahead)]
    return max(sum(future_daily), 0.0)


class CostAgent(BaseAgent):
    """
    Cost Optimisation Agent.

    All monetary values exposed to callers and in Insights are in INR.
    The agent receives USD values from the billing client and converts
    them using settings.usd_to_inr_rate internally.
    """

    agent_type = AgentType.COST

    async def execute(self, task: Task) -> AgentResponse:
        if task.task_type == TaskType.ANALYZE_COSTS:
            return await self._analyze_costs(task)
        if task.task_type == TaskType.IDENTIFY_OPTIMIZATION:
            return await self._identify_optimization(task)
        if task.task_type == TaskType.FORECAST_COSTS:
            return await self._forecast_costs(task)

        return self._partial(
            task,
            error_message=f"CostAgent does not handle task_type={task.task_type.value}",
        )

    # ── ANALYZE_COSTS ─────────────────────────────────────────────────────

    async def _analyze_costs(self, task: Task) -> AgentResponse:
        """
        Summarise current spend by service in INR with trend context.
        Validates Property 14.
        """
        cost_by_service: dict = task.parameters.get("cost_by_service", {})  # type: ignore[assignment]
        daily_costs: list = task.parameters.get("daily_costs", [])  # type: ignore[assignment]
        period_days: int = int(task.parameters.get("period_days", 30))

        if not cost_by_service:
            return self._partial(task, error_message="No cost_by_service data provided")

        # Convert to INR
        inr_by_service = {svc: _usd_to_inr(usd) for svc, usd in cost_by_service.items()}
        total_inr = sum(inr_by_service.values())

        # Daily totals in INR
        daily_inr = [
            {"date": d.get("date", ""), "amount_inr": _usd_to_inr(float(d.get("amount", 0)))}
            for d in daily_costs
        ]

        services_text = "\n".join(
            f"- {s}: ₹{inr:,.0f}" for s, inr in sorted(inr_by_service.items(), key=lambda x: -x[1])
        )

        kb_results = await knowledge_base.query_context(
            "aws cost optimization cloud spend reduction", max_results=3
        )
        kb_context = "\n\n".join(r.content for r in kb_results) or "No historical context."

        prompt = f"""You are an expert FinOps engineer analysing AWS costs for an Indian startup.

## Spend by Service (last {period_days} days, in INR)
{services_text}
Total: ₹{total_inr:,.0f}

## Historical Context
{kb_context}

Analyse the cost breakdown and identify top opportunities for savings.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "<aws service>",
      "title": "<finding>",
      "business_context": "<business impact>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "cost_impact": {{
        "monthly_inr": <monthly savings in INR as number>,
        "description": "<what drives this saving>"
      }},
      "recommendations": [
        {{
          "action": "<action>",
          "rationale": "<why>",
          "steps": ["<step>"],
          "expected_benefit": "<₹ saving>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            model_id=settings.get_agent_model_id("cost"),
            system_prompt="You are a FinOps AI. All monetary values must be in INR. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw)
        return self._success(
            task,
            insights=insights,
            data={
                "total_spend_inr": round(total_inr, 2),
                "inr_by_service": {k: round(v, 2) for k, v in inr_by_service.items()},
                "daily_inr": daily_inr,
            },
            model_used=settings.get_agent_model_id("cost"),
        )

    # ── IDENTIFY_OPTIMIZATION ─────────────────────────────────────────────

    async def _identify_optimization(self, task: Task) -> AgentResponse:
        """
        Flag resources with <30% utilisation over 7 days for rightsizing.
        Validates Property 16: <30% utilisation triggers recommendation.
        """
        cost_by_service: dict = task.parameters.get("cost_by_service", {})  # type: ignore[assignment]
        utilisation: dict = task.parameters.get("utilisation", {})  # type: ignore[assignment]
        rightsizing: list = task.parameters.get("rightsizing", [])  # type: ignore[assignment]

        if not cost_by_service and not rightsizing:
            return self._partial(task, error_message="No cost or rightsizing data provided")

        # Identify under-utilised services (Property 16)
        underutilised: list[dict] = []
        for service, util_pct in utilisation.items():
            if util_pct < 30.0:
                usd_spend = float(cost_by_service.get(service, 0))
                underutilised.append({
                    "service": service,
                    "utilisation_pct": util_pct,
                    "spend_inr": _usd_to_inr(usd_spend),
                })
        underutilised.sort(key=lambda x: -x["spend_inr"])

        util_text = "\n".join(
            f"- {u['service']}: {u['utilisation_pct']:.0f}% utilised, ₹{u['spend_inr']:,.0f}/period spend"
            for u in underutilised
        ) or "No services below 30% utilisation threshold."

        rs_text = "\n".join(
            f"- {r.get('resource_id','?')}: {r.get('recommendation','')} (est. saving: ${r.get('estimated_monthly_savings_usd', 0):.2f}/month)"
            for r in rightsizing[:10]
        ) or "No rightsizing recommendations from AWS."

        prompt = f"""You are an expert FinOps engineer identifying cost optimisation opportunities for an Indian startup.

## Under-Utilised Resources (<30% over 7 days)
{util_text}

## AWS Rightsizing Recommendations
{rs_text}

## Full Spend Context (INR)
{chr(10).join(f"- {s}: ₹{_usd_to_inr(float(v)):,.0f}" for s, v in sorted(cost_by_service.items(), key=lambda x: -float(x[1]))[:15])}

For each under-utilised resource, recommend specific rightsizing actions with INR savings.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "<service or resource>",
      "title": "Over-provisioned <resource>",
      "business_context": "<waste and opportunity cost>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<utilisation metric used>",
      "cost_impact": {{
        "monthly_inr": <savings as number>,
        "description": "<rightsizing action>"
      }},
      "recommendations": [
        {{
          "action": "<resize or delete action>",
          "rationale": "<utilisation data>",
          "steps": ["<step>"],
          "expected_benefit": "<₹ monthly saving>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            model_id=settings.get_agent_model_id("cost"),
            system_prompt="You are a FinOps AI specialising in rightsizing. All monetary values in INR. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw)
        return self._success(
            task,
            insights=insights,
            data={
                "underutilised_resources": underutilised,
                "rightsizing_count": len(rightsizing),
            },
            model_used=settings.get_agent_model_id("cost"),
        )

    # ── FORECAST_COSTS ────────────────────────────────────────────────────

    async def _forecast_costs(self, task: Task) -> AgentResponse:
        """
        Project next 30 days of cloud spend using linear trend + LLM context.
        Validates Property 15.
        """
        daily_costs: list = task.parameters.get("daily_costs", [])  # type: ignore[assignment]

        if not daily_costs:
            return self._partial(task, error_message="No daily_costs provided for forecasting")

        amounts_usd = [float(d.get("amount", 0)) for d in daily_costs if d.get("amount") is not None]
        if len(amounts_usd) < 3:
            return self._partial(task, error_message="Need at least 3 data points for forecast")

        forecast_usd = _simple_linear_forecast(amounts_usd, days_ahead=30)
        forecast_inr = _usd_to_inr(forecast_usd)
        current_30d_inr = _usd_to_inr(sum(amounts_usd[-30:]) if len(amounts_usd) >= 30 else sum(amounts_usd) * 30 / len(amounts_usd))

        trend = "increasing" if amounts_usd[-1] > amounts_usd[0] else "decreasing"
        avg_daily_inr = _usd_to_inr(statistics.mean(amounts_usd))

        prompt = f"""You are an expert FinOps engineer forecasting AWS costs for an Indian startup.

## Historical Spend (last {len(amounts_usd)} days)
- Average daily spend: ₹{avg_daily_inr:,.0f}
- Trend: {trend}
- Last 7-day average: ₹{_usd_to_inr(statistics.mean(amounts_usd[-7:])):,.0f}/day

## Statistical Forecast (next 30 days)
- Forecasted total: ₹{forecast_inr:,.0f}
- Current 30-day actual: ₹{current_30d_inr:,.0f}
- Change: {(forecast_inr - current_30d_inr) / current_30d_inr * 100:+.1f}%

Interpret this forecast and recommend actions to manage the projected spend.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "aws-total",
      "title": "<forecast summary>",
      "business_context": "<budget impact and cash flow consideration>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "cost_impact": {{
        "monthly_inr": <forecasted monthly spend as number>,
        "description": "30-day spend forecast"
      }},
      "recommendations": [
        {{
          "action": "<spend management action>",
          "rationale": "<trend analysis>",
          "steps": ["<step>"],
          "expected_benefit": "<₹ reduction>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            model_id=settings.get_agent_model_id("cost"),
            system_prompt="You are a FinOps forecasting AI. All monetary values in INR. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw)
        return self._success(
            task,
            insights=insights,
            data={
                "forecast_30d_inr": round(forecast_inr, 2),
                "current_30d_inr": round(current_30d_inr, 2),
                "trend": trend,
                "avg_daily_inr": round(avg_daily_inr, 2),
            },
            model_used=settings.get_agent_model_id("cost"),
        )
