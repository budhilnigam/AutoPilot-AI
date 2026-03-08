"""
agents/infra.py — Infrastructure Agent.

Handles:
  ANALYZE_DOCKERFILE    → layer cache, base image, multi-stage build opportunities
  DETECT_DRIFT          → compare current config vs stored baseline
  ANALYZE_WORKER_SIZING → worker count / concurrency recommendations

Task parameters:
  ANALYZE_DOCKERFILE:
    dockerfile: str         — raw Dockerfile text
    build_history: list     — optional list of {duration_seconds, commit_sha}

  DETECT_DRIFT:
    current_config: dict    — serialised Configuration object
    baseline_s3_uri: str    — S3 URI of the stored baseline config

  ANALYZE_WORKER_SIZING:
    worker_config: dict     — serialised WorkerConfig object
    metrics: list[dict]     — optional MetricData for queue depth / throughput

Validates Properties 7, 8, 9, 10, 6.
"""

from __future__ import annotations

import json
import re
from typing import Any

from autopilot_ai.agents.base import BaseAgent
from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.aws.bedrock import bedrock_client
from autopilot_ai.models.domain import Configuration, ConfigType, WorkerConfig
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
    cleaned = re.sub(r"^```(?:json|hcl|yaml|dockerfile)?\s*", "", text.strip(), flags=re.MULTILINE)
    return re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)


def _parse_insights(raw: str, category: InsightCategory) -> list[Insight]:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        logger.warning("infra_agent_json_parse_error", raw=raw[:200])
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
                    expected_benefit=r.get("expected_benefit", "Improved reliability"),
                )
                for r in item.get("recommendations", [])
            ] or [Recommendation(
                action="Review configuration",
                rationale="Issue detected",
                steps=["Review the configuration"],
                expected_benefit="Reduced risk",
            )]

            results.append(Insight(
                category=category,
                component=item.get("component", "infrastructure"),
                title=item.get("title", "Infrastructure finding"),
                business_context=item.get("business_context", ""),
                urgency=urgency_map.get(item.get("urgency", "medium"), Urgency.MEDIUM),
                confidence=float(item.get("confidence", 0.7)),
                recommendations=recs,
                attribution=item.get("attribution"),
            ))
        except Exception as e:
            logger.warning("infra_insight_parse_error", error=str(e))

    return results


class InfraAgent(BaseAgent):
    """
    Infrastructure Agent — Dockerfile analysis, drift detection, worker sizing.

    Parsing of Dockerfile, docker-compose YAML, and Terraform HCL is done
    with regex/heuristics here. Claude is used only for reasoning, not parsing.
    """

    agent_type = AgentType.INFRA

    async def execute(self, task: Task) -> AgentResponse:
        if task.task_type == TaskType.ANALYZE_DOCKERFILE:
            return await self._analyze_dockerfile(task)
        if task.task_type == TaskType.DETECT_DRIFT:
            return await self._detect_drift(task)
        if task.task_type == TaskType.ANALYZE_WORKER_SIZING:
            return await self._analyze_worker_sizing(task)

        return self._partial(
            task,
            error_message=f"InfraAgent does not handle task_type={task.task_type.value}",
        )

    # ── ANALYZE_DOCKERFILE ────────────────────────────────────────────────

    async def _analyze_dockerfile(self, task: Task) -> AgentResponse:
        """
        Identify layer caching, base image, multi-stage build, and dependency
        management optimization opportunities.
        Validates Property 7: at least one optimization category returned.
        """
        dockerfile: str = str(task.parameters.get("dockerfile", ""))
        build_history: list = task.parameters.get("build_history", [])  # type: ignore[assignment]

        if not dockerfile.strip():
            return self._partial(task, error_message="No Dockerfile content provided")

        # Heuristic pre-analysis (no LLM)
        lines = dockerfile.splitlines()
        has_multistage = sum(1 for l in lines if l.strip().upper().startswith("FROM")) > 1
        has_apt_clean = "apt-get clean" in dockerfile or "rm -rf /var/lib/apt" in dockerfile
        copies_before_install = False
        last_copy_idx = last_run_idx = -1
        for i, line in enumerate(lines):
            s = line.strip().upper()
            if s.startswith("COPY") or s.startswith("ADD"):
                last_copy_idx = i
            if s.startswith("RUN"):
                last_run_idx = i
        if last_copy_idx < last_run_idx and last_copy_idx != -1:
            copies_before_install = True

        avg_build = ""
        if build_history:
            durations = [b.get("duration_seconds", 0) for b in build_history if b.get("duration_seconds")]
            if durations:
                avg_build = f"\nAverage recent build time: {sum(durations)/len(durations):.0f}s over {len(durations)} runs."

        kb_results = await knowledge_base.query_context(
            "dockerfile optimization layer caching multi-stage build",
            max_results=3,
        )
        kb_context = "\n\n".join(r.content for r in kb_results) or "No historical context."

        prompt = f"""You are an expert DevOps engineer analyzing a Dockerfile for an Indian startup.

## Dockerfile
```
{dockerfile[:3000]}
```

## Pre-Analysis
- Multi-stage build: {"YES" if has_multistage else "NO — consider adding"}
- APT cache cleanup: {"YES" if has_apt_clean else "NO — missing"}
- COPY before install (good for layer cache): {"YES" if copies_before_install else "NO — COPY is after RUN, poor cache"}
{avg_build}

## Historical Context
{kb_context}

## Task
Identify optimization opportunities. You MUST cover at least one of: layer caching, base image selection, multi-stage builds, dependency management.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "dockerfile",
      "title": "<optimization opportunity>",
      "business_context": "<impact: build time, image size, cost in INR if relevant>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "recommendations": [
        {{
          "action": "<specific change>",
          "rationale": "<why it helps>",
          "steps": ["<step>"],
          "expected_benefit": "<e.g. '40% faster builds, saves ₹6,200/month in CI costs'>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            model_id=settings.get_agent_model_id("infra"),
            system_prompt="You are an infrastructure optimization AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw, InsightCategory.CONFIGURATION)
        return self._success(
            task,
            insights=insights,
            data={"has_multistage": has_multistage, "has_apt_clean": has_apt_clean},
            model_used=settings.get_agent_model_id("infra"),
        )

    # ── DETECT_DRIFT ──────────────────────────────────────────────────────

    async def _detect_drift(self, task: Task) -> AgentResponse:
        """
        Compare current config against stored baseline and report drift.
        Validates Property 8.
        """
        current_config_dict: dict = task.parameters.get("current_config", {})  # type: ignore[assignment]
        baseline_s3_uri: str = str(task.parameters.get("baseline_s3_uri", ""))

        current_content: str = str(current_config_dict.get("content", ""))
        config_type: str = str(current_config_dict.get("config_type", "unknown"))

        if not current_content:
            return self._partial(task, error_message="No current configuration content provided")

        # Fetch baseline from S3/KB
        baseline_content = ""
        if baseline_s3_uri:
            try:
                baseline_content = await knowledge_base.get_document(baseline_s3_uri)
            except Exception as e:
                logger.warning("infra_drift_baseline_fetch_failed", error=str(e))

        if not baseline_content:
            # No baseline — store current as the new baseline and return no drift
            config = Configuration(
                config_type=ConfigType(config_type) if config_type in ConfigType._value2member_map_ else ConfigType.CUSTOM,
                content=current_content,
                source_path=str(current_config_dict.get("source_path", "unknown")),
                commit_sha=str(current_config_dict.get("commit_sha", "")),
            )
            await knowledge_base.store_configuration(config)
            return self._success(
                task,
                insights=[],
                data={"drift_detected": False, "reason": "No baseline — current stored as baseline"},
            )

        if current_content.strip() == baseline_content.strip():
            return self._success(
                task,
                insights=[],
                data={"drift_detected": False},
            )

        # There is drift — ask Claude to interpret it
        prompt = f"""You are an expert DevOps engineer performing infrastructure drift detection.

## Baseline Configuration ({config_type})
```
{baseline_content[:2000]}
```

## Current Configuration ({config_type})
```
{current_content[:2000]}
```

Identify all differences and their impact. Respond as JSON:
{{
  "insights": [
    {{
      "component": "<resource name>",
      "title": "<drift summary>",
      "business_context": "<production risk and business impact>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<specific changed parameter or section>",
      "recommendations": [
        {{
          "action": "<remediation>",
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
            model_id=settings.get_agent_model_id("infra"),
            system_prompt="You are an infrastructure drift analysis AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw, InsightCategory.CONFIGURATION)
        return self._success(
            task,
            insights=insights,
            data={"drift_detected": len(insights) > 0},
            model_used=settings.get_agent_model_id("infra"),
        )

    # ── ANALYZE_WORKER_SIZING ─────────────────────────────────────────────

    async def _analyze_worker_sizing(self, task: Task) -> AgentResponse:
        """
        Recommend optimal worker count and concurrency values.
        Validates Property 10: specific concurrency values recommended when
        saturation or under-utilisation is detected.
        """
        wc_dict: dict = task.parameters.get("worker_config", {})  # type: ignore[assignment]
        metrics_raw: list = task.parameters.get("metrics", [])  # type: ignore[assignment]

        if not wc_dict:
            return self._partial(task, error_message="No worker_config provided")

        wc = WorkerConfig(**wc_dict)

        # Heuristic saturation / under-utilisation check
        throughput_needed = wc.peak_tasks_per_minute or 0
        avg_duration = wc.avg_task_duration_seconds or 0
        current_capacity = 0.0
        saturation_note = ""
        if throughput_needed > 0 and avg_duration > 0:
            current_capacity = (wc.worker_count * wc.concurrency * 60) / avg_duration
            utilisation = throughput_needed / current_capacity if current_capacity > 0 else 0
            if utilisation > 0.85:
                saturation_note = (
                    f"SATURATED: current capacity {current_capacity:.0f} tasks/min "
                    f"vs demand {throughput_needed:.0f} tasks/min ({utilisation*100:.0f}% utilisation). "
                    f"Recommended workers = {int(throughput_needed * avg_duration / (wc.concurrency * 60)) + 1}."
                )
            elif utilisation < 0.30:
                recommended_workers = max(1, int(throughput_needed * avg_duration / (wc.concurrency * 60)) + 1)
                saturation_note = (
                    f"UNDER-UTILISED: only {utilisation*100:.0f}% utilisation. "
                    f"Recommended workers = {recommended_workers} (currently {wc.worker_count})."
                )

        metrics_text = ""
        for m_dict in metrics_raw:
            metrics_text += f"- {m_dict.get('metric_name','?')}: latest={m_dict.get('latest_value','?')}\n"

        prompt = f"""You are an expert SRE sizing worker pools for an Indian startup.

## Current Worker Configuration
Service: {wc.service_name}
Workers: {wc.worker_count}, Concurrency: {wc.concurrency}
Memory: {wc.memory_mb}MB, CPU: {wc.cpu_units} units
Queue size: {wc.queue_size}
Avg task duration: {wc.avg_task_duration_seconds}s
Peak tasks/min: {wc.peak_tasks_per_minute}

## Heuristic Analysis
{saturation_note or "Utilisation within normal range."}

## Related Metrics
{metrics_text or "No live metrics provided."}

## Task
Recommend exact worker count and concurrency values. Be specific with numbers.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "{wc.service_name}",
      "title": "<worker sizing finding>",
      "business_context": "<throughput impact and cost in INR>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "recommendations": [
        {{
          "action": "Set worker_count=<N> concurrency=<M>",
          "rationale": "<capacity math>",
          "steps": ["Update ECS task definition", "Deploy with --desired-count <N>"],
          "expected_benefit": "<throughput and cost outcome>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            model_id=settings.get_agent_model_id("infra"),
            system_prompt="You are an SRE capacity planning AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw, InsightCategory.PERFORMANCE)
        return self._success(
            task,
            insights=insights,
            data={
                "saturation_note": saturation_note,
                "current_capacity_per_min": round(current_capacity, 1),
            },
            model_used=settings.get_agent_model_id("infra"),
        )
