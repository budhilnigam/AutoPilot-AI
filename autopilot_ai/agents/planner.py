"""
agents/planner.py — Planner Agent (Orchestrator).

The Planner is the single entry point for every user interaction:
  - Natural language queries  ("why is checkout slow?")
  - Dashboard refresh         ("give me a full system health snapshot")
  - Alert triage              ("ECS memory alarm just fired — what's happening?")

Flow for PLAN_QUERY
───────────────────
1. Use Claude (fast model) to analyse the query and decide which specialized
   agents to call, with what task_type and parameters.
2. Detect dependency cycles (raises CircularDependencyError).
3. Execute agents in topological order; independent agents run in parallel
   via asyncio.gather.
4. Collect all AgentResponses.
5. Use Claude (primary model) to synthesise a plain-English narrative from
   all insights.
6. Return a QueryResponse with narrative + all agent responses.

Task parameters for PLAN_QUERY:
  query: str               — natural-language question or alert description
  context: dict            — optional extra context (e.g. alert metadata)
  mode: "query"|"alert"|"dashboard"  (default "query")

Task parameters for SYNTHESIZE_RESPONSES:
  responses: list[dict]    — serialised AgentResponse objects
  original_query: str      — the original question for synthesis context
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

from autopilot_ai.agents.base import BaseAgent
from autopilot_ai.agents.cicd import CICDAgent
from autopilot_ai.agents.cost import CostAgent
from autopilot_ai.agents.db import DBAgent
from autopilot_ai.agents.infra import InfraAgent
from autopilot_ai.agents.observability import ObservabilityAgent
from autopilot_ai.agents.tool_generator import ToolGeneratorAgent
from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import CircularDependencyError, PlannerError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.aws.bedrock import bedrock_client
from autopilot_ai.models.responses import AgentResponse, QueryResponse, ResponseStatus
from autopilot_ai.models.tasks import AgentType, Priority, Task, TaskType
from autopilot_ai.services.metrics_service import metrics_service

logger = get_logger(__name__)

# ── Agent registry ────────────────────────────────────────────────────────────
# Instantiated once; each agent is stateless beyond its circuit breaker state.
_AGENT_REGISTRY: dict[AgentType, BaseAgent] = {
    AgentType.OBSERVABILITY: ObservabilityAgent(),
    AgentType.INFRA: InfraAgent(),
    AgentType.DB: DBAgent(),
    AgentType.COST: CostAgent(),
    AgentType.CICD: CICDAgent(),
    AgentType.TOOL_GENERATOR: ToolGeneratorAgent(),
}

# ── Routing prompt ────────────────────────────────────────────────────────────
_ROUTING_SYSTEM = (
    "You are the planning layer of AutoPilot-AI, an SRE assistant for an Indian startup. "
    "You decompose user queries into structured agent task plans. "
    "Respond only with valid JSON."
)

_ROUTING_CHOICES = """
Available agent types and task types:
  observability: analyze_metrics | detect_anomalies | attribute_bottleneck
  infra:         analyze_dockerfile | detect_drift | analyze_worker_sizing
  db:            analyze_query_plan | recommend_indices | analyze_redis
  cost:          analyze_costs | identify_optimization | forecast_costs
  cicd:          track_build_times | detect_build_regression | predict_build_failure | analyze_workflow
  tool_generator: generate_tool | validate_tool
"""


def _strip_fences(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    return re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)


def _topological_sort(
    plans: list[dict],
) -> list[list[dict]]:
    """
    Group plans into execution waves respecting `depends_on` keys.
    Each wave contains plans that can run in parallel.

    Raises CircularDependencyError if a cycle is detected.

    `plans` is a list of dicts with keys:
      id: str
      agent_type: str
      task_type: str
      parameters: dict
      depends_on: list[str]   — list of plan IDs this plan waits for
      priority: str
    """
    remaining = {p["id"]: p for p in plans}
    completed: set[str] = set()
    waves: list[list[dict]] = []

    max_iterations = len(plans) + 1
    iteration = 0

    while remaining:
        iteration += 1
        if iteration > max_iterations:
            raise CircularDependencyError(
                "Circular dependency detected in planner task graph",
                unresolved=list(remaining.keys()),
            )

        ready = [
            p for p in remaining.values()
            if all(dep in completed for dep in p.get("depends_on", []))
        ]
        if not ready:
            raise CircularDependencyError(
                "No progress possible — circular dependency in task graph",
                unresolved=list(remaining.keys()),
            )

        waves.append(ready)
        for p in ready:
            completed.add(p["id"])
            del remaining[p["id"]]

    return waves


class PlannerAgent(BaseAgent):
    """
    Orchestrating Planner Agent.

    Uses Claude to decompose queries into agent tasks, executes them in
    topological order (independent tasks in parallel), and synthesises
    a final narrative response.
    """

    agent_type = AgentType.PLANNER

    async def execute(self, task: Task) -> AgentResponse:
        if task.task_type == TaskType.PLAN_QUERY:
            query_response = await self._plan_and_execute(task)
            # Wrap QueryResponse in an AgentResponse for the BaseAgent contract
            return AgentResponse(
                agent_type=AgentType.PLANNER,
                task_id=task.id,
                status=ResponseStatus.SUCCESS,
                execution_time_ms=0.0,  # will be overwritten by BaseAgent timer
                insights=query_response.all_insights,
                data={"query_response": query_response.model_dump(mode="json")},
                model_used=settings.bedrock_model_id,
            )

        if task.task_type == TaskType.SYNTHESIZE_RESPONSES:
            narrative = await self._synthesize(
                responses_data=task.parameters.get("responses", []),  # type: ignore[arg-type]
                original_query=str(task.parameters.get("original_query", "")),
            )
            return self._success(task, insights=[], data={"narrative": narrative})

        return self._partial(
            task,
            error_message=f"PlannerAgent does not handle task_type={task.task_type.value}",
        )

    # ── Main orchestration ────────────────────────────────────────────────

    async def _plan_and_execute(self, task: Task) -> QueryResponse:
        query: str = str(task.parameters.get("query", ""))
        context: dict = task.parameters.get("context", {})  # type: ignore[assignment]
        mode: str = str(task.parameters.get("mode", "query"))
        query_id: str = task.correlation_id or str(uuid.uuid4())

        if not query.strip():
            raise PlannerError("Empty query", task_id=task.id)

        t_start = time.perf_counter()
        logger.info("planner_start", query_id=query_id, mode=mode, query=query[:100])

        # ── Step 1: Route query to agent tasks ───────────────────────────
        plans = await self._route_query(query, context, mode)
        if not plans:
            # No agents needed — return direct narrative from Claude
            narrative = await self._direct_answer(query)
            return QueryResponse(
                query_id=query_id,
                narrative=narrative,
                agent_responses=[],
                total_insights=0,
                execution_time_ms=(time.perf_counter() - t_start) * 1000,
            )

        # ── Step 2: Build execution waves ────────────────────────────────
        try:
            waves = _topological_sort(plans)
        except CircularDependencyError as e:
            logger.error("planner_circular_dependency", error=str(e))
            raise

        logger.info("planner_execution_plan", waves=len(waves), total_tasks=len(plans))

        # ── Step 3: Execute waves ─────────────────────────────────────────
        all_responses: list[AgentResponse] = []

        for wave_idx, wave in enumerate(waves):
            logger.info("planner_wave_start", wave=wave_idx, tasks=len(wave))
            wave_tasks = [self._execute_plan(p, task, query_id) for p in wave]
            results = await asyncio.gather(*wave_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error("planner_wave_task_error", error=str(result))
                    # Continue — don't let one agent failure abort the whole query
                else:
                    all_responses.append(result)  # type: ignore[arg-type]

        # ── Step 4: Synthesise narrative ──────────────────────────────────
        narrative = await self._synthesize_from_responses(query, all_responses)

        total_insights = sum(len(r.insights) for r in all_responses)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "planner_complete",
            query_id=query_id,
            total_agents=len(all_responses),
            total_insights=total_insights,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return QueryResponse(
            query_id=query_id,
            narrative=narrative,
            agent_responses=all_responses,
            total_insights=total_insights,
            execution_time_ms=elapsed_ms,
        )

    # ── Routing ───────────────────────────────────────────────────────────

    async def _route_query(
        self,
        query: str,
        context: dict,
        mode: str,
    ) -> list[dict]:
        """
        Ask Claude (fast model) to decide which agents to invoke.
        Returns a list of plan dicts ready for topological sort.
        """
        context_text = json.dumps(context, default=str)[:1000] if context else "None"

        prompt = f"""You are the planner for AutoPilot-AI. A user has asked:

"{query}"

Mode: {mode}
Context: {context_text}

{_ROUTING_CHOICES}

Decide which agents to invoke. Return a JSON array of tasks:
[
  {{
    "id": "t1",
    "agent_type": "<agent>",
    "task_type": "<task_type>",
    "parameters": {{}},
    "depends_on": [],
    "priority": "high|medium|low"
  }}
]

Rules:
- Only invoke agents that are relevant to the query.
- Set depends_on if one task needs another's output (e.g. attribution_bottleneck depends on detect_anomalies).
- For "dashboard" mode, invoke observability + cost + cicd at minimum.
- For "alert" mode, invoke observability first, then infra or db as needed.
- For simple factual questions with no data needed, return an empty array [].
- Keep parameters as an object with keys relevant to the task_type.
- For any task that needs repo analysis, set parameters.repo to the repo name from context if available.

Respond with JSON array only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            model_id=settings.bedrock_fast_model_id,
            system_prompt=_ROUTING_SYSTEM,
        )

        cleaned = _strip_fences(raw).strip()
        # Handle case where Claude wraps in {"tasks": [...]}
        if cleaned.startswith("{"):
            try:
                obj = json.loads(cleaned)
                cleaned = json.dumps(obj.get("tasks", obj.get("plan", [])))
            except json.JSONDecodeError:
                pass

        try:
            plans: list[dict] = json.loads(cleaned)
            if not isinstance(plans, list):
                plans = []
        except json.JSONDecodeError:
            logger.warning("planner_routing_parse_error", raw=raw[:300])
            plans = []

        # Assign UUIDs if Claude didn't provide stable IDs
        for i, p in enumerate(plans):
            if not p.get("id"):
                p["id"] = f"t{i+1}"
            if "parameters" not in p or not isinstance(p.get("parameters"), dict):
                p["parameters"] = {}
            if "depends_on" not in p:
                p["depends_on"] = []

        logger.info("planner_routing_result", tasks=[p.get("id") for p in plans])
        return plans

    # ── Task execution ────────────────────────────────────────────────────

    async def _enrich_plan_with_metrics(
        self,
        plan: dict,
        parent_task: Task,
    ) -> None:
        """
        Enrich observability agent plans with real CloudWatch metrics.
        
        Modifies plan["parameters"] in-place to include fetched metrics.
        """
        agent_type_str = plan.get("agent_type", "")
        task_type_str = plan.get("task_type", "")
        
        # Only enrich observability tasks that need metrics
        if agent_type_str != "observability":
            return
        
        if task_type_str not in ["analyze_metrics", "detect_anomalies"]:
            return
        
        # Extract query and context for metric fetching
        query: str = str(parent_task.parameters.get("query", ""))
        context: dict = parent_task.parameters.get("context", {})  # type: ignore[assignment]
        
        # Get lookback window from plan parameters or use default
        lookback_minutes = int(plan.get("parameters", {}).get("timerange_minutes", 60))
        
        logger.info(
            "planner_fetching_metrics",
            task_type=task_type_str,
            lookback_minutes=lookback_minutes,
        )
        
        try:
            # Fetch metrics from CloudWatch
            metrics = await metrics_service.fetch_metrics_for_query(
                query=query,
                context=context,
                lookback_minutes=lookback_minutes,
            )
            
            if metrics:
                # Serialize MetricData objects to dicts for the agent
                metrics_serialized = [
                    {
                        "metric_name": m.metric_name,
                        "namespace": m.namespace,
                        "dimensions": m.dimensions,
                        "datapoints": [
                            {"timestamp": dp.timestamp, "value": dp.value}
                            for dp in m.datapoints
                        ],
                        "statistic": m.statistic,
                        "period_seconds": m.period_seconds,
                        "unit": m.unit,
                        "latest_value": m.latest_value,
                        "mean": m.mean,
                    }
                    for m in metrics
                ]
                
                # Add metrics to plan parameters
                if "parameters" not in plan:
                    plan["parameters"] = {}
                plan["parameters"]["metrics"] = metrics_serialized
                
                logger.info(
                    "planner_metrics_enriched",
                    count=len(metrics),
                    task_type=task_type_str,
                )
            else:
                logger.warning(
                    "planner_no_metrics_found",
                    query=query[:100],
                    task_type=task_type_str,
                )
        except Exception as e:
            logger.error(
                "planner_metric_fetch_failed",
                error=str(e),
                task_type=task_type_str,
            )
            # Continue without metrics rather than failing the whole query

    async def _execute_plan(
        self,
        plan: dict,
        parent_task: Task,
        query_id: str,
    ) -> AgentResponse:
        """Build a Task from a plan dict and dispatch it to the right agent."""
        # Enrich observability plans with real metrics before execution
        await self._enrich_plan_with_metrics(plan, parent_task)
        
        agent_type_str: str = plan.get("agent_type", "")
        task_type_str: str = plan.get("task_type", "")

        try:
            agent_type = AgentType(agent_type_str)
        except ValueError:
            logger.warning("planner_unknown_agent_type", agent_type=agent_type_str)
            return AgentResponse(
                agent_type=AgentType.PLANNER,
                task_id=plan.get("id", "unknown"),
                status=ResponseStatus.FAILED,
                execution_time_ms=0.0,
                error_message=f"Unknown agent_type: {agent_type_str}",
            )

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            logger.warning("planner_unknown_task_type", task_type=task_type_str)
            return AgentResponse(
                agent_type=agent_type,
                task_id=plan.get("id", "unknown"),
                status=ResponseStatus.FAILED,
                execution_time_ms=0.0,
                error_message=f"Unknown task_type: {task_type_str}",
            )

        agent = _AGENT_REGISTRY.get(agent_type)
        if agent is None:
            return AgentResponse(
                agent_type=agent_type,
                task_id=plan.get("id", "unknown"),
                status=ResponseStatus.FAILED,
                execution_time_ms=0.0,
                error_message=f"No agent registered for {agent_type.value}",
            )

        priority_str = plan.get("priority", "medium")
        try:
            priority = Priority(priority_str)
        except ValueError:
            priority = Priority.MEDIUM

        sub_task = Task(
            agent_type=agent_type,
            task_type=task_type,
            parameters=plan.get("parameters", {}),
            context=parent_task.context,
            priority=priority,
            correlation_id=query_id,
        )

        logger.info(
            "planner_dispatching",
            agent=agent_type.value,
            task_type=task_type.value,
            task_id=sub_task.id,
        )

        return await agent(sub_task)

    # ── Synthesis ─────────────────────────────────────────────────────────

    async def _synthesize_from_responses(
        self,
        query: str,
        responses: list[AgentResponse],
    ) -> str:
        """Use primary Claude model to write the final narrative."""
        if not responses:
            return await self._direct_answer(query)

        all_insights = [
            {
                "agent": r.agent_type.value,
                "title": ins.title,
                "component": ins.component,
                "urgency": ins.urgency.value,
                "business_context": ins.business_context,
                "recommendations": [
                    {"action": rec.action, "benefit": rec.expected_benefit}
                    for rec in ins.recommendations
                ],
                "cost_impact_inr": ins.cost_impact.monthly_inr if ins.cost_impact else None,
            }
            for r in responses
            for ins in r.insights
        ]

        failed = [r for r in responses if r.status in (ResponseStatus.FAILED, ResponseStatus.TIMEOUT)]
        failed_text = (
            "\n".join(f"- {r.agent_type.value}: {r.error_message}" for r in failed)
            if failed else "All agents completed successfully."
        )

        prompt = f"""You are an SRE AI assistant responding to an engineering team at an Indian startup.

## Original Query
{query}

## Findings from {len(responses)} agents ({sum(len(r.insights) for r in responses)} total insights)
{json.dumps(all_insights, indent=2, default=str)[:6000]}

## Agent Status
{failed_text}

Write a clear, concise, actionable response in plain English.
- Lead with the most urgent finding.
- Group related issues.
- Include specific INR cost figures where available (format as ₹X,XXX).
- End with a prioritised action list (numbered, most urgent first).
- Be direct — this is for an engineer, not a manager.
- Maximum 400 words."""

        return await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are an expert SRE AI assistant. Be concise and actionable.",
        )

    async def _synthesize(
        self,
        responses_data: list[dict],
        original_query: str,
    ) -> str:
        """Synthesise narrative from pre-serialised AgentResponse dicts."""
        responses = []
        for d in responses_data:
            try:
                responses.append(AgentResponse.model_validate(d))
            except Exception as e:
                logger.warning("planner_response_deserialize_error", error=str(e))
        return await self._synthesize_from_responses(original_query, responses)

    async def _direct_answer(self, query: str) -> str:
        """Answer a simple factual question directly without agents."""
        return await bedrock_client.invoke(
            prompt=f"Answer this question from an SRE team at an Indian startup: {query}",
            model_id=settings.bedrock_fast_model_id,
            system_prompt="You are a concise SRE assistant. Be helpful and brief.",
        )


# Module-level singleton
planner = PlannerAgent()
