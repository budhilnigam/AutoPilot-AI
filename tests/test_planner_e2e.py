"""
test_planner_e2e.py — End-to-end test of the planner routing + agent execution.

Tests the exact path a real user query takes:
  POST /api/query → PlannerAgent → routing → enrichment → agent → response

Run with: python test_planner_e2e.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Load .env first
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


def _sep(label: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print('-' * 60)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"         {msg}")


async def main():
    from autopilot_ai.agents.planner import PlannerAgent
    from autopilot_ai.models.tasks import AgentType, Priority, Task, TaskType

    p = PlannerAgent()

    # ── Test 1: Keyword fallback routing ──────────────────────────────────
    _sep("1. Keyword fallback routing")
    test_queries = {
        "How much am I spending on AWS?": "cost",
        "List my repos": "github",
        "What is my infrastructure?": "infra",
        "CPU usage is high": "observability",
    }
    for q, expected in test_queries.items():
        plans = p._keyword_fallback_route(q, "query")
        agents = [p_item["agent_type"] for p_item in plans]
        if expected in agents:
            _ok(f'"{q}" → {agents}')
        else:
            _fail(f'"{q}" → {agents} (expected {expected})')

    # ── Test 2: Route via LLM ─────────────────────────────────────────────
    _sep("2. LLM-based routing (fast model)")
    try:
        plans = await p._route_query("How much am I spending on AWS this month?", {}, "query")
        if plans:
            agents = [pl["agent_type"] for pl in plans]
            _ok(f"LLM routed to: {agents}")
            for pl in plans:
                _info(f"  {pl['agent_type']}:{pl['task_type']} params={list(pl.get('parameters',{}).keys())}")
        else:
            _fail("LLM returned empty plan [] — keyword fallback will save this")
    except Exception as e:
        _fail(f"LLM routing failed: {type(e).__name__}: {e}")

    # ── Test 3: Full cost query (planner → enrichment → cost agent) ───────
    _sep("3. Full cost query end-to-end")
    try:
        task = Task(
            task_type=TaskType.PLAN_QUERY,
            agent_type=AgentType.PLANNER,
            priority=Priority.HIGH,
            parameters={
                "query": "What are my current AWS costs?",
                "context": {},
                "mode": "query",
            },
        )
        result = await p.execute(task)
        _ok(f"Status: {result.status.value}")
        qr = result.data.get("query_response", {})
        narrative = qr.get("narrative", result.error_message or "No narrative")
        _info(f"Agents called: {len(qr.get('agent_responses', []))}")
        _info(f"Total insights: {qr.get('total_insights', 0)}")
        # Show first 300 chars of narrative
        narr_preview = narrative[:300] if narrative else "(empty)"
        _info(f"Narrative (preview): {narr_preview}...")
    except Exception as e:
        _fail(f"Cost query failed: {type(e).__name__}: {e}")

    # ── Test 4: GitHub query ──────────────────────────────────────────────
    _sep("4. Full GitHub query end-to-end")
    try:
        task = Task(
            task_type=TaskType.PLAN_QUERY,
            agent_type=AgentType.PLANNER,
            priority=Priority.MEDIUM,
            parameters={
                "query": "List my repositories",
                "context": {},
                "mode": "query",
            },
        )
        result = await p.execute(task)
        _ok(f"Status: {result.status.value}")
        qr = result.data.get("query_response", {})
        narrative = qr.get("narrative", result.error_message or "No narrative")
        _info(f"Agents called: {len(qr.get('agent_responses', []))}")
        narr_preview = narrative[:300] if narrative else "(empty)"
        _info(f"Narrative (preview): {narr_preview}...")
    except Exception as e:
        _fail(f"GitHub query failed: {type(e).__name__}: {e}")

    # ── Test 5: Observability query ───────────────────────────────────────
    _sep("5. Full observability query end-to-end")
    try:
        task = Task(
            task_type=TaskType.PLAN_QUERY,
            agent_type=AgentType.PLANNER,
            priority=Priority.HIGH,
            parameters={
                "query": "Are there any anomalies in my metrics right now?",
                "context": {},
                "mode": "query",
            },
        )
        result = await p.execute(task)
        _ok(f"Status: {result.status.value}")
        qr = result.data.get("query_response", {})
        narrative = qr.get("narrative", result.error_message or "No narrative")
        _info(f"Agents called: {len(qr.get('agent_responses', []))}")
        narr_preview = narrative[:300] if narrative else "(empty)"
        _info(f"Narrative (preview): {narr_preview}...")
    except Exception as e:
        _fail(f"Observability query failed: {type(e).__name__}: {e}")

    _sep("Done")


if __name__ == "__main__":
    asyncio.run(main())
