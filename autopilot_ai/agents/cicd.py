"""
agents/cicd.py — CI/CD Agent.

Handles:
  TRACK_BUILD_TIMES      → ingest build history and compute rolling baseline
  DETECT_BUILD_REGRESSION → flag if latest build > 1.5× baseline (Property 17)
  PREDICT_BUILD_FAILURE  → if failure rate >10% with same pattern, predict failure
                           and return confidence score (Property 19)
  ANALYZE_WORKFLOW        → fetch GitHub workflow runs and commit diff, explain failures

Task parameters:
  TRACK_BUILD_TIMES:
    builds: list[dict]   — [{sha, duration_seconds, status, branch, timestamp}]

  DETECT_BUILD_REGRESSION:
    builds: list[dict]   — recent builds in chronological order
    threshold_multiplier: float (optional, default 1.5)

  PREDICT_BUILD_FAILURE:
    recent_builds: list[dict]   — last N builds with same trigger/pattern
    current_build: dict         — {sha, branch, changed_files: list[str]}

  ANALYZE_WORKFLOW:
    repo: str
    workflow_run_id: int (optional)
    since_hours: int (default 24)

Validates Properties 17, 18, 19, 6.
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime, timezone, timedelta
from typing import Any

from autopilot_ai.agents.base import BaseAgent
from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.aws.bedrock import bedrock_client
from autopilot_ai.integrations.github.client import github_client
from autopilot_ai.models.insights import (
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


def _parse_insights(raw: str) -> list[Insight]:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        logger.warning("cicd_agent_json_parse_error", raw=raw[:200])
        return []

    urgency_map = {u.value: u for u in Urgency}
    results: list[Insight] = []

    for item in data.get("insights", []):
        try:
            recs = [
                Recommendation(
                    action=r.get("action", "Review"),
                    rationale=r.get("rationale", "See details"),
                    steps=r.get("steps", ["Investigate"]),
                    expected_benefit=r.get("expected_benefit", "Faster builds"),
                )
                for r in item.get("recommendations", [])
            ] or [Recommendation(
                action="Review CI/CD pipeline",
                rationale="Issue detected",
                steps=["Check build logs"],
                expected_benefit="Reduced build time",
            )]

            results.append(Insight(
                category=InsightCategory.CICD,
                component=item.get("component", "ci-cd"),
                title=item.get("title", "CI/CD finding"),
                business_context=item.get("business_context", ""),
                urgency=urgency_map.get(item.get("urgency", "medium"), Urgency.MEDIUM),
                confidence=float(item.get("confidence", 0.7)),
                recommendations=recs,
                attribution=item.get("attribution"),
            ))
        except Exception as e:
            logger.warning("cicd_insight_parse_error", error=str(e))

    return results


def _build_baseline(builds: list[dict]) -> tuple[float, float]:
    """
    Compute mean and stdev of build durations from a list of build dicts.
    Returns (mean, stdev). Uses all builds except the last one (current).
    """
    durations = [
        float(b["duration_seconds"])
        for b in builds[:-1]
        if b.get("duration_seconds") and b.get("status") == "success"
    ]
    if not durations:
        return 0.0, 0.0
    mean = statistics.mean(durations)
    stdev = statistics.stdev(durations) if len(durations) > 1 else 0.0
    return mean, stdev


class CICDAgent(BaseAgent):
    """
    CI/CD Agent — build time tracking, regression detection, failure prediction.

    Statistical analysis is done locally; Claude is used only for root-cause
    interpretation when a regression or failure pattern is confirmed.
    """

    agent_type = AgentType.CICD

    async def execute(self, task: Task) -> AgentResponse:
        if task.task_type == TaskType.TRACK_BUILD_TIMES:
            return await self._track_build_times(task)
        if task.task_type == TaskType.DETECT_BUILD_REGRESSION:
            return await self._detect_build_regression(task)
        if task.task_type == TaskType.PREDICT_BUILD_FAILURE:
            return await self._predict_build_failure(task)
        if task.task_type == TaskType.ANALYZE_WORKFLOW:
            return await self._analyze_workflow(task)

        return self._partial(
            task,
            error_message=f"CICDAgent does not handle task_type={task.task_type.value}",
        )

    # ── TRACK_BUILD_TIMES ─────────────────────────────────────────────────

    async def _track_build_times(self, task: Task) -> AgentResponse:
        """
        Ingest build history (up to 30 days) and compute rolling stats.
        Stores the summary in the Knowledge Base for future regression checks.
        Validates Property 18.
        """
        builds: list = task.parameters.get("builds", [])  # type: ignore[assignment]

        if not builds:
            return self._partial(task, error_message="No build data provided")

        successful = [b for b in builds if b.get("status") == "success" and b.get("duration_seconds")]
        failed = [b for b in builds if b.get("status") == "failure"]

        if not successful:
            return self._success(
                task,
                insights=[],
                data={"message": "No successful builds to establish baseline"},
            )

        durations = [float(b["duration_seconds"]) for b in successful]
        mean_s = statistics.mean(durations)
        median_s = statistics.median(durations)
        stdev_s = statistics.stdev(durations) if len(durations) > 1 else 0.0
        p95_s = sorted(durations)[int(len(durations) * 0.95)]

        summary = {
            "build_count": len(builds),
            "success_count": len(successful),
            "failure_count": len(failed),
            "mean_duration_seconds": round(mean_s, 1),
            "median_duration_seconds": round(median_s, 1),
            "p95_duration_seconds": round(p95_s, 1),
            "stdev_seconds": round(stdev_s, 1),
            "failure_rate_pct": round(len(failed) / len(builds) * 100, 1) if builds else 0.0,
        }

        logger.info("cicd_build_stats", **summary)

        return self._success(
            task,
            insights=[],
            data=summary,
        )

    # ── DETECT_BUILD_REGRESSION ───────────────────────────────────────────

    async def _detect_build_regression(self, task: Task) -> AgentResponse:
        """
        Compare the latest build duration against the rolling baseline.
        Regression = latest > threshold_multiplier × baseline mean (default 1.5×).
        Validates Property 17.
        """
        builds: list = task.parameters.get("builds", [])  # type: ignore[assignment]
        multiplier: float = float(task.parameters.get("threshold_multiplier", settings.build_regression_multiplier))

        if len(builds) < 2:
            return self._partial(task, error_message="Need at least 2 builds to detect regression")

        latest = builds[-1]
        latest_duration = float(latest.get("duration_seconds", 0))
        baseline_mean, baseline_stdev = _build_baseline(builds)

        if baseline_mean == 0:
            return self._success(
                task,
                insights=[],
                data={"regression_detected": False, "reason": "No successful baseline builds"},
            )

        threshold = baseline_mean * multiplier
        regression_detected = latest_duration > threshold
        ratio = latest_duration / baseline_mean if baseline_mean > 0 else 0

        if not regression_detected:
            return self._success(
                task,
                insights=[],
                data={
                    "regression_detected": False,
                    "latest_duration_s": latest_duration,
                    "baseline_mean_s": round(baseline_mean, 1),
                    "ratio": round(ratio, 2),
                    "threshold_s": round(threshold, 1),
                },
            )

        # Regression confirmed — ask Claude for root cause
        builds_context = "\n".join(
            f"- {b.get('sha','?')[:8]}: {b.get('duration_seconds','?')}s ({b.get('status','?')}) [{b.get('branch','?')}]"
            for b in builds[-10:]
        )

        prompt = f"""You are an expert DevOps engineer analysing a CI/CD build regression for an Indian startup.

## Regression Detected
- Latest build: {latest_duration:.0f}s (SHA {str(latest.get('sha','?'))[:8]}, branch: {latest.get('branch','main')})
- Baseline mean: {baseline_mean:.0f}s (± {baseline_stdev:.0f}s)
- Threshold ({multiplier}×): {threshold:.0f}s
- Actual ratio: {ratio:.2f}×

## Recent Build History
{builds_context}

Explain why the build likely regressed and how to fix it. Consider: dependency installation changes, new test suites, Docker layer cache misses, flaky steps, resource constraints.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "ci-build",
      "title": "Build regression — {ratio:.1f}× slower than baseline",
      "business_context": "<developer productivity and deployment velocity impact>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<likely cause>",
      "recommendations": [
        {{
          "action": "<fix>",
          "rationale": "<why>",
          "steps": ["<step>"],
          "expected_benefit": "<time saved>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are a CI/CD performance AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw)
        return self._success(
            task,
            insights=insights,
            data={
                "regression_detected": True,
                "latest_duration_s": latest_duration,
                "baseline_mean_s": round(baseline_mean, 1),
                "ratio": round(ratio, 2),
                "threshold_s": round(threshold, 1),
            },
            model_used=settings.bedrock_model_id,
        )

    # ── PREDICT_BUILD_FAILURE ─────────────────────────────────────────────

    async def _predict_build_failure(self, task: Task) -> AgentResponse:
        """
        If failure rate > 10% for builds with the same trigger pattern,
        predict the current build will likely fail and return a confidence score.
        Validates Property 19.
        """
        recent_builds: list = task.parameters.get("recent_builds", [])  # type: ignore[assignment]
        current_build: dict = task.parameters.get("current_build", {})  # type: ignore[assignment]

        if not recent_builds:
            return self._partial(task, error_message="No recent_builds provided for prediction")

        total = len(recent_builds)
        failures = [b for b in recent_builds if b.get("status") == "failure"]
        failure_rate = len(failures) / total if total > 0 else 0.0

        if failure_rate <= 0.10:
            return self._success(
                task,
                insights=[],
                data={
                    "failure_predicted": False,
                    "failure_rate_pct": round(failure_rate * 100, 1),
                    "threshold_pct": 10.0,
                },
            )

        # >10% failure rate — compute confidence from rate
        # confidence = min(0.99, 0.5 + failure_rate * 0.5)
        confidence = min(0.99, 0.5 + failure_rate * 0.5)

        changed_files: list = current_build.get("changed_files", [])  # type: ignore[assignment]
        infra_files = [
            f for f in changed_files
            if any(p in f.lower() for p in ["dockerfile", ".tf", "requirements", "pyproject", ".github/workflows"])
        ]

        failure_reasons = [b.get("failure_reason", "") for b in failures if b.get("failure_reason")]
        reason_text = "\n".join(f"- {r}" for r in failure_reasons[:10]) or "No failure reasons recorded."

        prompt = f"""You are an expert DevOps engineer predicting build failures for an Indian startup.

## Failure Statistics
- Recent builds: {total}
- Failure rate: {failure_rate*100:.1f}% (threshold: 10%)
- Computed confidence of failure: {confidence:.0%}

## Common Failure Reasons
{reason_text}

## Current Build
- SHA: {current_build.get('sha','?')[:8]}
- Branch: {current_build.get('branch','?')}
- Changed files: {len(changed_files)} total ({len(infra_files)} infra/config files)
- Infra changes: {', '.join(infra_files[:5]) or 'none'}

Predict why this build is likely to fail and what pre-emptive actions to take.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "ci-build",
      "title": "High probability of build failure ({failure_rate*100:.0f}% recent failure rate)",
      "business_context": "<deployment delay and developer time cost>",
      "urgency": "immediate|high|medium|low",
      "confidence": {confidence:.2f},
      "attribution": "<most likely failure category>",
      "recommendations": [
        {{
          "action": "<pre-emptive action>",
          "rationale": "<pattern observed>",
          "steps": ["<step>"],
          "expected_benefit": "<reduced failure probability>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are a CI/CD failure prediction AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw)

        # Override confidence with our statistical value on the primary insight
        if insights:
            object.__setattr__(insights[0], "confidence", round(confidence, 2)) if hasattr(insights[0], "__dict__") else None
            try:
                # Pydantic v2 model — use model_copy to update
                insights[0] = insights[0].model_copy(update={"confidence": round(confidence, 2)})
            except Exception:
                pass

        return self._success(
            task,
            insights=insights,
            data={
                "failure_predicted": True,
                "failure_rate_pct": round(failure_rate * 100, 1),
                "confidence": round(confidence, 2),
                "infra_files_changed": infra_files,
            },
            model_used=settings.bedrock_model_id,
        )

    # ── ANALYZE_WORKFLOW ──────────────────────────────────────────────────

    async def _analyze_workflow(self, task: Task) -> AgentResponse:
        """
        Fetch recent GitHub workflow runs and explain any failures.
        Validates Property 18.
        """
        repo: str = str(task.parameters.get("repo", ""))
        since_hours: int = int(task.parameters.get("since_hours", 24))

        if not repo:
            return self._partial(task, error_message="No repo provided")

        since = datetime.now(tz=timezone.utc) - timedelta(hours=since_hours)
        runs = await github_client.get_workflow_runs(repo, since=since)

        if not runs:
            return self._success(
                task,
                insights=[],
                data={"message": f"No workflow runs in the last {since_hours}h"},
            )

        failed_runs = [r for r in runs if r.conclusion == "failure"]
        total = len(runs)
        failure_rate = len(failed_runs) / total if total > 0 else 0.0

        runs_text = "\n".join(
            f"- [{r.conclusion or r.status}] {r.workflow_name} #{r.run_number} "
            f"({r.branch}) — {r.duration_seconds or '?'}s — {r.sha[:8] if r.sha else '?'}"
            for r in runs[:20]
        )

        prompt = f"""You are an expert DevOps engineer analysing GitHub Actions workflows for an Indian startup.

## Workflow Runs (last {since_hours}h)
- Total: {total}, Failed: {len(failed_runs)}, Failure rate: {failure_rate*100:.0f}%

{runs_text}

Identify patterns in failures and recommend fixes.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "github-actions",
      "title": "<finding>",
      "business_context": "<deployment velocity impact>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<workflow name or step>",
      "recommendations": [
        {{
          "action": "<fix>",
          "rationale": "<why>",
          "steps": ["<step>"],
          "expected_benefit": "<outcome>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are a GitHub Actions AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw)
        return self._success(
            task,
            insights=insights,
            data={
                "total_runs": total,
                "failed_runs": len(failed_runs),
                "failure_rate_pct": round(failure_rate * 100, 1),
            },
            model_used=settings.bedrock_model_id,
        )
