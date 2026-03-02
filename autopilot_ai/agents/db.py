"""
agents/db.py — Database Agent.

Handles:
  ANALYZE_QUERY_PLAN  → parse EXPLAIN ANALYZE output, identify issues, suggest fixes
  RECOMMEND_INDICES   → generate SQL DDL for missing indices
  ANALYZE_REDIS       → parse Redis INFO output, flag evictions, high memory, etc.

Task parameters:
  ANALYZE_QUERY_PLAN:
    query: str               — the SQL query
    explain_output: str      — raw text of EXPLAIN (ANALYZE, BUFFERS) output

  RECOMMEND_INDICES:
    slow_queries: list[dict] — each: {query: str, avg_ms: float, calls: int}
    table_schema: str        — optional CREATE TABLE DDL for context

  ANALYZE_REDIS:
    redis_info: str          — raw output of Redis INFO ALL command

Validates Properties 11, 12, 13.
"""

from __future__ import annotations

import json
import re

from autopilot_ai.agents.base import BaseAgent
from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.aws.bedrock import bedrock_client
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
    cleaned = re.sub(r"^```(?:json|sql)?\s*", "", text.strip(), flags=re.MULTILINE)
    return re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)


def _parse_insights(raw: str, category: InsightCategory) -> list[Insight]:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        logger.warning("db_agent_json_parse_error", raw=raw[:200])
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
                    expected_benefit=r.get("expected_benefit", "Improved performance"),
                )
                for r in item.get("recommendations", [])
            ] or [Recommendation(
                action="Review query",
                rationale="Issue detected",
                steps=["Review the slow query"],
                expected_benefit="Reduced latency",
            )]

            results.append(Insight(
                category=category,
                component=item.get("component", "database"),
                title=item.get("title", "Database finding"),
                business_context=item.get("business_context", ""),
                urgency=urgency_map.get(item.get("urgency", "medium"), Urgency.MEDIUM),
                confidence=float(item.get("confidence", 0.7)),
                recommendations=recs,
                attribution=item.get("attribution"),
            ))
        except Exception as e:
            logger.warning("db_insight_parse_error", error=str(e))

    return results


def _parse_redis_info(info_text: str) -> dict[str, str]:
    """Parse Redis INFO output into a flat dict of key→value."""
    result: dict[str, str] = {}
    for line in info_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _extract_explain_stats(explain_text: str) -> dict[str, str]:
    """
    Extract key stats from PostgreSQL EXPLAIN ANALYZE output:
    actual total time, rows, seq scans, index scans, sort operations.
    Pure text parsing — no LLM involved.
    """
    stats: dict[str, str] = {}

    # Execution time
    m = re.search(r"Execution Time:\s*([\d.]+)\s*ms", explain_text)
    if m:
        stats["execution_time_ms"] = m.group(1)

    # Planning time
    m = re.search(r"Planning Time:\s*([\d.]+)\s*ms", explain_text)
    if m:
        stats["planning_time_ms"] = m.group(1)

    # Seq Scan count
    stats["seq_scans"] = str(len(re.findall(r"Seq Scan on", explain_text)))

    # Index Scan count
    stats["index_scans"] = str(len(re.findall(r"Index (?:Only )?Scan", explain_text)))

    # Sort operations
    stats["sort_ops"] = str(len(re.findall(r"Sort  \(", explain_text)))

    # Hash join operations
    stats["hash_joins"] = str(len(re.findall(r"Hash Join|Hash Semi Join|Hash Anti Join", explain_text)))

    # Nested loop with large rows
    nested = re.findall(r"Nested Loop.*?rows=(\d+)", explain_text)
    stats["nested_loop_max_rows"] = max((int(r) for r in nested), default=0).__str__()

    return stats


class DBAgent(BaseAgent):
    """
    Database Agent — query plan analysis, index recommendations, Redis analysis.

    Heuristic parsing is used for EXPLAIN output and Redis INFO.
    Claude is invoked for reasoning and DDL generation, not raw text parsing.
    """

    agent_type = AgentType.DB

    async def execute(self, task: Task) -> AgentResponse:
        if task.task_type == TaskType.ANALYZE_QUERY_PLAN:
            return await self._analyze_query_plan(task)
        if task.task_type == TaskType.RECOMMEND_INDICES:
            return await self._recommend_indices(task)
        if task.task_type == TaskType.ANALYZE_REDIS:
            return await self._analyze_redis(task)

        return self._partial(
            task,
            error_message=f"DBAgent does not handle task_type={task.task_type.value}",
        )

    # ── ANALYZE_QUERY_PLAN ────────────────────────────────────────────────

    async def _analyze_query_plan(self, task: Task) -> AgentResponse:
        """
        Parse EXPLAIN ANALYZE text (no LLM), then call Claude for reasoning.
        Validates Property 11.
        """
        query: str = str(task.parameters.get("query", ""))
        explain: str = str(task.parameters.get("explain_output", ""))

        if not explain.strip():
            return self._partial(task, error_message="No EXPLAIN output provided")

        stats = _extract_explain_stats(explain)

        kb_results = await knowledge_base.query_context(
            "postgresql slow query optimization sequential scan index",
            max_results=3,
        )
        kb_context = "\n\n".join(r.content for r in kb_results) or "No historical context."

        prompt = f"""You are an expert PostgreSQL DBA analyzing a slow query for an Indian startup.

## SQL Query
```sql
{query[:1000] or "Not provided"}
```

## EXPLAIN ANALYZE Output
```
{explain[:3000]}
```

## Pre-Parsed Stats
- Execution time: {stats.get('execution_time_ms','?')}ms
- Planning time: {stats.get('planning_time_ms','?')}ms
- Sequential scans: {stats.get('seq_scans','?')}
- Index scans: {stats.get('index_scans','?')}
- Sort operations: {stats.get('sort_ops','?')}
- Hash joins: {stats.get('hash_joins','?')}
- Max nested-loop rows: {stats.get('nested_loop_max_rows','?')}

## Historical Context
{kb_context}

## Task
Identify performance bottlenecks and actionable optimisations. Focus on: index usage, join strategy, sort elimination, covering indices.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "postgresql",
      "title": "<bottleneck name>",
      "business_context": "<impact: query latency, user experience, cost>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<node type or operation causing the bottleneck>",
      "recommendations": [
        {{
          "action": "<specific SQL or config change>",
          "rationale": "<why this works>",
          "steps": ["<step>"],
          "expected_benefit": "<e.g. '10× speedup for this query pattern'>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are a PostgreSQL performance AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw, InsightCategory.PERFORMANCE)
        return self._success(
            task,
            insights=insights,
            data={"explain_stats": stats},
            model_used=settings.bedrock_model_id,
        )

    # ── RECOMMEND_INDICES ─────────────────────────────────────────────────

    async def _recommend_indices(self, task: Task) -> AgentResponse:
        """
        Generate SQL DDL for missing indices based on slow query list.
        Validates Property 12: SQL DDL returned in steps of recommendations.
        """
        slow_queries: list = task.parameters.get("slow_queries", [])  # type: ignore[assignment]
        table_schema: str = str(task.parameters.get("table_schema", ""))

        if not slow_queries:
            return self._partial(task, error_message="No slow_queries provided")

        # Format slow query list
        queries_text = "\n".join(
            f"{i+1}. [{q.get('calls', 1)} calls, avg {q.get('avg_ms', 0):.0f}ms] {q.get('query', '')[:300]}"
            for i, q in enumerate(slow_queries[:20])
        )

        schema_section = f"\n## Table Schema\n```sql\n{table_schema[:2000]}\n```" if table_schema else ""

        prompt = f"""You are an expert PostgreSQL DBA recommending indices for an Indian startup.

## Slow Queries (sorted by total cost)
{queries_text}
{schema_section}

## Task
For each query pattern that would benefit from an index, provide exact SQL DDL (CREATE INDEX CONCURRENTLY).
Consider: composite indices, partial indices, covering indices (INCLUDE), and expression indices.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "postgresql",
      "title": "Missing index on <table>(<columns>)",
      "business_context": "<queries affected and expected latency reduction>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "recommendations": [
        {{
          "action": "Create index <index_name>",
          "rationale": "<which queries benefit and why>",
          "steps": [
            "CREATE INDEX CONCURRENTLY <name> ON <table>(<col1>, <col2>) [WHERE ...] [INCLUDE (...)];",
            "ANALYZE <table>;"
          ],
          "expected_benefit": "<estimated speedup for affected queries>"
        }}
      ]
    }}
  ]
}}

Respond with JSON only."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are a PostgreSQL index advisor. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw, InsightCategory.PERFORMANCE)
        return self._success(
            task,
            insights=insights,
            data={"slow_query_count": len(slow_queries)},
            model_used=settings.bedrock_model_id,
        )

    # ── ANALYZE_REDIS ─────────────────────────────────────────────────────

    async def _analyze_redis(self, task: Task) -> AgentResponse:
        """
        Parse Redis INFO output and identify evictions, fragmentation,
        slow commands, and memory concerns.
        Validates Property 13.
        """
        redis_info: str = str(task.parameters.get("redis_info", ""))

        if not redis_info.strip():
            return self._partial(task, error_message="No redis_info provided")

        parsed = _parse_redis_info(redis_info)

        # Key heuristic metrics
        evicted_keys = int(parsed.get("evicted_keys", 0))
        used_memory_mb = round(int(parsed.get("used_memory", 0)) / 1024 / 1024, 1)
        max_memory_mb = round(int(parsed.get("maxmemory", 0)) / 1024 / 1024, 1)
        frag_ratio = float(parsed.get("mem_fragmentation_ratio", 1.0))
        hit_rate = 0.0
        hits = int(parsed.get("keyspace_hits", 0))
        misses = int(parsed.get("keyspace_misses", 0))
        if hits + misses > 0:
            hit_rate = hits / (hits + misses) * 100

        heuristic_notes: list[str] = []
        if evicted_keys > 0:
            heuristic_notes.append(f"EVICTIONS: {evicted_keys} keys evicted — data loss occurring")
        if frag_ratio > 1.5:
            heuristic_notes.append(f"HIGH FRAGMENTATION: ratio {frag_ratio:.2f} (>1.5 is problematic)")
        if max_memory_mb > 0 and used_memory_mb / max_memory_mb > 0.85:
            heuristic_notes.append(f"MEMORY PRESSURE: {used_memory_mb}MB used of {max_memory_mb}MB ({used_memory_mb/max_memory_mb*100:.0f}%)")
        if hit_rate < 80 and (hits + misses) > 100:
            heuristic_notes.append(f"LOW HIT RATE: {hit_rate:.1f}% (target ≥80%)")

        heuristic_summary = "\n".join(f"- {n}" for n in heuristic_notes) or "No critical issues detected by heuristics."

        prompt = f"""You are an expert Redis SRE analyzing a production Redis instance for an Indian startup.

## Key Metrics
- Used memory: {used_memory_mb}MB / {max_memory_mb}MB max
- Cache hit rate: {hit_rate:.1f}%
- Evicted keys: {evicted_keys}
- Fragmentation ratio: {frag_ratio:.2f}
- Connected clients: {parsed.get('connected_clients', '?')}
- Blocked clients: {parsed.get('blocked_clients', '?')}
- Maxmemory policy: {parsed.get('maxmemory_policy', 'unknown')}
- Redis version: {parsed.get('redis_version', '?')}

## Heuristic Findings
{heuristic_summary}

## Task
Analyse the Redis statistics and provide recommendations. Cover: eviction tuning, maxmemory-policy, fragmentation, TTL strategy, connection pool sizing.

Respond as JSON:
{{
  "insights": [
    {{
      "component": "redis",
      "title": "<issue name>",
      "business_context": "<user-facing impact and cost>",
      "urgency": "immediate|high|medium|low",
      "confidence": <0.0-1.0>,
      "attribution": "<specific Redis metric or config parameter>",
      "recommendations": [
        {{
          "action": "<config change or Redis command>",
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
            system_prompt="You are a Redis performance AI. Respond only with valid JSON.",
        )

        insights = _parse_insights(raw, InsightCategory.PERFORMANCE)
        return self._success(
            task,
            insights=insights,
            data={
                "used_memory_mb": used_memory_mb,
                "hit_rate_pct": round(hit_rate, 1),
                "evicted_keys": evicted_keys,
                "frag_ratio": frag_ratio,
                "heuristic_notes": heuristic_notes,
            },
            model_used=settings.bedrock_model_id,
        )
