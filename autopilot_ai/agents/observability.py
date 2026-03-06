"""
agents/observability.py — Observability Agent.

Handles:
  ANALYZE_METRICS       → CloudWatch metrics → semantic Insight list
  DETECT_ANOMALIES      → 2-sigma statistical check, then Claude for root cause
  ATTRIBUTE_BOTTLENECK  → correlates a latency spike to a specific component

Task parameters expected per task_type:
  ANALYZE_METRICS:
    metrics: list[dict]   — serialised MetricData objects
    timerange_minutes: int (optional, default 60)

  DETECT_ANOMALIES:
    metrics: list[dict]   — serialised MetricData objects
    sigma_threshold: float (optional, default from settings)

  ATTRIBUTE_BOTTLENECK:
    timeseries: dict      — serialised TimeSeries object
    symptom: str          — plain-English description of the observed problem

Validates Properties 4, 5, 6, 31.
"""

from __future__ import annotations

import statistics
from typing import Any

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
from autopilot_ai.models.metrics import Anomaly, MetricData, MetricDataPoint, TimeSeries
from autopilot_ai.models.responses import AgentResponse
from autopilot_ai.models.tasks import AgentType, Task, TaskType
from autopilot_ai.services.knowledge_base import knowledge_base

logger = get_logger(__name__)


def _parse_metrics(raw: list[dict[str, Any]]) -> list[MetricData]:
    """Deserialise list of MetricData parameter dicts from a Task."""
    result = []
    for item in raw:
        # Rebuild MetricDataPoint objects from raw dicts
        datapoints = [
            MetricDataPoint(timestamp=dp["timestamp"], value=dp["value"])
            for dp in item.get("datapoints", [])
        ]
        result.append(
            MetricData(
                metric_name=item["metric_name"],
                namespace=item.get("namespace", "Custom"),
                dimensions=item.get("dimensions", {}),
                datapoints=datapoints,
                statistic=item.get("statistic", "Average"),
                period_seconds=item.get("period_seconds", 300),
            )
        )
    return result


def _detect_anomalies_statistical(
    metrics: list[MetricData], sigma_threshold: float
) -> list[Anomaly]:
    """
    Pure statistical anomaly detection — no LLM needed.

    For each metric, computes mean and std over all-but-last datapoint,
    then checks if the latest value exceeds `sigma_threshold` standard
    deviations from the mean.

    Validates Property 5: detects deviation > sigma_threshold with confidence
    proportional to the magnitude of deviation.
    """
    anomalies: list[Anomaly] = []

    for metric in metrics:
        values = metric.values
        if len(values) < 4:
            continue  # need at least 4 points for meaningful statistics

        baseline = values[:-1]  # all but latest
        latest = values[-1]
        latest_ts = metric.timestamps[-1]

        mean = statistics.mean(baseline)
        try:
            std = statistics.stdev(baseline)
        except statistics.StatisticsError:
            continue

        if std == 0:
            continue  # constant metric — no anomaly possible

        sigma = abs(latest - mean) / std

        if sigma >= sigma_threshold:
            # Confidence scales from 0.7 at threshold up to 0.99 at 4x threshold
            raw_confidence = 0.7 + (sigma - sigma_threshold) / (4 * sigma_threshold) * 0.29
            confidence = min(raw_confidence, 0.99)

            anomalies.append(
                Anomaly(
                    metric_name=metric.metric_name,
                    timestamp=latest_ts,
                    observed_value=latest,
                    baseline_mean=round(mean, 4),
                    baseline_std=round(std, 4),
                    sigma_deviation=round(sigma, 2),
                    confidence=round(confidence, 3),
                    component=metric.dimensions.get(
                        "ServiceName",
                        metric.dimensions.get("InstanceId", "unknown"),
                    ),
                )
            )

    return anomalies


class ObservabilityAgent(BaseAgent):
    """
    Observability Agent — metric interpretation, anomaly detection, bottleneck attribution.

    All three task types follow the same pattern:
      1. Deserialise input from task.parameters
      2. Statistical pre-processing (no LLM)
      3. Query KB for historical context
      4. Call Claude via BedrockClient for semantic reasoning
      5. Return structured Insight list
    """

    agent_type = AgentType.OBSERVABILITY

    # ── Task dispatch ──────────────────────────────────────────────────────

    async def execute(self, task: Task) -> AgentResponse:
        if task.task_type == TaskType.ANALYZE_METRICS:
            return await self._analyze_metrics(task)
        if task.task_type == TaskType.DETECT_ANOMALIES:
            return await self._detect_anomalies(task)
        if task.task_type == TaskType.ATTRIBUTE_BOTTLENECK:
            return await self._attribute_bottleneck(task)

        return self._partial(
            task,
            error_message=f"ObservabilityAgent does not handle task_type={task.task_type.value}",
        )

    # ── ANALYZE_METRICS ────────────────────────────────────────────────────

    async def _analyze_metrics(self, task: Task) -> AgentResponse:
        """
        Converts raw CloudWatch metric data into semantic Insight objects.
        Validates Property 4: insights include business context, severity, recommendations.
        """
        raw_metrics: list[dict] = task.parameters.get("metrics", [])  # type: ignore[assignment]
        metrics = _parse_metrics(raw_metrics)

        if not metrics:
            return self._partial(task, error_message="No metrics provided")

        # Build a compact metric summary for the prompt
        metric_lines: list[str] = []
        for m in metrics:
            if m.latest_value is None:
                continue
            mean_text = f"{m.mean:.2f}" if m.mean is not None else "N/A"
            metric_lines.append(
                f"- {m.metric_name} [{m.namespace}]: "
                f"latest={m.latest_value:.2f} {m.unit}, "
                f"mean={mean_text}, "
                f"dimensions={m.dimensions}"
            )
        metric_summary = "\n".join(metric_lines)

        # KB context retrieval
        kb_results = await knowledge_base.query_context(
            f"infrastructure metrics analysis {' '.join(m.metric_name for m in metrics)}",
            max_results=3,
        )
        kb_context = (
            "\n\n".join(r.content for r in kb_results)
            if kb_results
            else "No historical context available."
        )

        prompt = f"""You are an expert SRE analyzing infrastructure metrics for an Indian startup.

## Current Metrics
{metric_summary}

## Historical Context from Knowledge Base
{kb_context}

## Task
Analyze these metrics and produce a JSON response with this exact structure:
{{
  "insights": [
    {{
      "component": "<service/resource name>",
      "title": "<one-line finding>",
      "business_context": "<plain-English business impact, mention INR cost if relevant>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "recommendations": [
        {{
          "action": "<imperative action>",
          "rationale": "<why>",
          "steps": ["<step 1>", "<step 2>"],
          "expected_benefit": "<concrete outcome>"
        }}
      ]
    }}
  ]
}}

Rules:
- urgency=immediate only for issues actively degrading user experience
- Include INR cost impact in business_context when relevant
- Each insight MUST have at least one recommendation
- Respond with JSON only, no prose."""

        raw_response = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are an SRE AI assistant. Respond only with valid JSON.",
        )

        insights = self._parse_insights_from_response(raw_response, InsightCategory.PERFORMANCE)

        return self._success(
            task,
            insights=insights,
            data={"metrics_analyzed": len(metrics), "kb_results": len(kb_results)},
            model_used=settings.bedrock_model_id,
        )

    # ── DETECT_ANOMALIES ───────────────────────────────────────────────────

    async def _detect_anomalies(self, task: Task) -> AgentResponse:
        """
        Two-phase anomaly detection:
          Phase 1 — statistical (2-sigma), no LLM
          Phase 2 — Claude provides root cause for detected anomalies
        Validates Property 5.
        """
        raw_metrics: list[dict] = task.parameters.get("metrics", [])  # type: ignore[assignment]
        sigma_threshold: float = float(
            task.parameters.get("sigma_threshold", settings.anomaly_sigma_threshold)  # type: ignore[arg-type]
        )
        metrics = _parse_metrics(raw_metrics)

        # Phase 1: statistical detection
        anomalies = _detect_anomalies_statistical(metrics, sigma_threshold)

        if not anomalies:
            return self._success(
                task,
                insights=[],
                data={"anomalies_detected": 0, "metrics_checked": len(metrics)},
            )

        # Phase 2: Claude root-cause analysis for detected anomalies
        anomaly_text = "\n".join(
            f"- {a.metric_name}: observed={a.observed_value:.2f}, "
            f"baseline_mean={a.baseline_mean:.2f}, "
            f"sigma={a.sigma_deviation:.1f}σ, "
            f"component={a.component}"
            for a in anomalies
        )

        kb_results = await knowledge_base.query_context(
            f"anomaly root cause {' '.join(a.metric_name for a in anomalies)}",
            max_results=3,
        )
        kb_context = (
            "\n\n".join(r.content for r in kb_results)
            if kb_results
            else "No historical context available."
        )

        prompt = f"""You are an expert SRE. The following statistical anomalies were detected:

{anomaly_text}

Historical context:
{kb_context}

For each anomaly, provide root cause analysis as JSON:
{{
  "insights": [
    {{
      "component": "<component name>",
      "title": "<anomaly summary>",
      "business_context": "<user-facing impact and business consequence>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<probable root cause: commit SHA, config change, traffic spike, etc.>",
      "recommendations": [
        {{
          "action": "<fix action>",
          "rationale": "<why this fixes it>",
          "steps": ["<step>"],
          "expected_benefit": "<outcome>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw_response = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are an SRE AI assistant. Respond only with valid JSON.",
        )

        insights = self._parse_insights_from_response(raw_response, InsightCategory.ANOMALY)

        return self._success(
            task,
            insights=insights,
            data={
                "anomalies_detected": len(anomalies),
                "metrics_checked": len(metrics),
                "sigma_threshold": sigma_threshold,
                "anomaly_details": [
                    {
                        "metric": a.metric_name,
                        "sigma": a.sigma_deviation,
                        "confidence": a.confidence,
                    }
                    for a in anomalies
                ],
            },
            model_used=settings.bedrock_model_id,
        )

    # ── ATTRIBUTE_BOTTLENECK ───────────────────────────────────────────────

    async def _attribute_bottleneck(self, task: Task) -> AgentResponse:
        """
        Correlates an observed symptom to a specific infrastructure component.
        Validates Property 31 (attribution completeness).
        """
        symptom: str = str(task.parameters.get("symptom", "Unknown performance issue"))
        raw_ts: dict = task.parameters.get("timeseries", {})  # type: ignore[assignment]

        # Build a readable correlation analysis prompt
        metrics_summary = ""
        if raw_ts:
            for m_dict in raw_ts.get("metrics", []):
                m = MetricData(**{
                    k: v for k, v in m_dict.items() if k != "datapoints"
                } | {"datapoints": [
                    MetricDataPoint(**dp) for dp in m_dict.get("datapoints", [])
                ]})
                metrics_summary += (
                    f"- {m.metric_name}: latest={m.latest_value}, mean={m.mean:.2f if m.mean else 'N/A'}\n"
                )

        kb_results = await knowledge_base.query_context(
            f"bottleneck attribution {symptom}",
            max_results=3,
        )
        kb_context = "\n\n".join(r.content for r in kb_results) or "No context available."

        prompt = f"""You are an expert SRE performing bottleneck attribution.

Observed symptom: {symptom}

Current metric readings:
{metrics_summary or 'No metrics provided.'}

Historical context:
{kb_context}

Identify which infrastructure component is the root cause. Respond as JSON:
{{
  "insights": [
    {{
      "component": "<specific component: e.g. celery-worker, rds-postgres, redis>",
      "title": "<one-line bottleneck statement>",
      "business_context": "<business impact explanation>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<specific evidence: metric correlation, commit SHA, config change>",
      "recommendations": [
        {{
          "action": "<remediation action>",
          "rationale": "<why>",
          "steps": ["<step>"],
          "expected_benefit": "<outcome>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw_response = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are an SRE AI assistant. Respond only with valid JSON.",
        )

        insights = self._parse_insights_from_response(raw_response, InsightCategory.PERFORMANCE)
        return self._success(task, insights=insights, model_used=settings.bedrock_model_id)

    # ── JSON parsing helper ────────────────────────────────────────────────

    def _parse_insights_from_response(
        self, raw: str, default_category: InsightCategory
    ) -> list[Insight]:
        """
        Parse Claude's JSON response into a list of Insight objects.

        Handles malformed JSON gracefully — returns a single PARTIAL insight
        rather than crashing the agent.
        """
        import json
        import re

        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("observability_parse_error", error=str(e), raw=raw[:200])
            return []

        insights: list[Insight] = []
        urgency_map = {u.value: u for u in Urgency}

        for item in data.get("insights", []):
            try:
                recs = [
                    Recommendation(
                        action=r.get("action", "Review configuration"),
                        rationale=r.get("rationale", "See insight details"),
                        steps=r.get("steps", ["Investigate"]),
                        expected_benefit=r.get("expected_benefit", "Improved performance"),
                    )
                    for r in item.get("recommendations", [])
                ]
                if not recs:
                    recs = [Recommendation(
                        action="Investigate further",
                        rationale="Anomaly detected — root cause unclear",
                        steps=["Review dashboards", "Check recent deployments"],
                        expected_benefit="Identify and resolve root cause",
                    )]

                insights.append(
                    Insight(
                        category=default_category,
                        component=item.get("component", "unknown"),
                        title=item.get("title", "Unnamed insight"),
                        business_context=item.get("business_context", ""),
                        urgency=urgency_map.get(
                            item.get("urgency", "medium"), Urgency.MEDIUM
                        ),
                        confidence=float(item.get("confidence", 0.7)),
                        recommendations=recs,
                        attribution=item.get("attribution"),
                    )
                )
            except Exception as e:
                logger.warning("observability_insight_parse_error", error=str(e))
                continue

        return insights
