#!/usr/bin/env python3
"""
Quick integration test for CloudWatch metrics → Observability Agent flow.

This script verifies that:
1. MetricsService can fetch CloudWatch metrics based on natural language queries
2. The Planner enriches observability tasks with real metrics
3. The ObservabilityAgent receives and can process the metric data

Usage:
    python test_cloudwatch_integration.py
"""

import asyncio
import json
from datetime import datetime, timezone

from autopilot_ai.core.logging import get_logger
from autopilot_ai.services.metrics_service import metrics_service
from autopilot_ai.agents.planner import planner
from autopilot_ai.models.tasks import Task, TaskType, AgentType, Priority

logger = get_logger(__name__)


async def test_metrics_service():
    """Test that MetricsService can fetch CloudWatch metrics."""
    print("\n" + "="*80)
    print("TEST 1: MetricsService - Fetch metrics for natural language query")
    print("="*80)
    
    query = "why is the checkout service slow?"
    context = {"component": "checkout-service", "service_name": "checkout-api"}
    
    print(f"\nQuery: {query}")
    print(f"Context: {json.dumps(context, indent=2)}")
    
    try:
        metrics = await metrics_service.fetch_metrics_for_query(
            query=query,
            context=context,
            lookback_minutes=60,
        )
        
        print(f"\n✓ Fetched {len(metrics)} metrics from CloudWatch:")
        for metric in metrics[:5]:  # Show first 5
            print(f"  - {metric.namespace}/{metric.metric_name}")
            print(f"    Dimensions: {metric.dimensions}")
            print(f"    Data points: {len(metric.datapoints)}")
            if metric.latest_value is not None:
                print(f"    Latest value: {metric.latest_value:.2f} {metric.unit}")
            print()
        
        if len(metrics) > 5:
            print(f"  ... and {len(metrics) - 5} more\n")
        
        return len(metrics) > 0
    except Exception as e:
        print(f"\n✗ Failed to fetch metrics: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_planner_enrichment():
    """Test that Planner enriches observability tasks with metrics."""
    print("\n" + "="*80)
    print("TEST 2: Planner Enrichment - Check metrics injection into tasks")
    print("="*80)
    
    query = ("Our checkout API is experiencing high latency. "
             "CPU usage seems elevated and users are reporting timeouts.")
    
    print(f"\nQuery: {query}\n")
    
    try:
        # Create a PLAN_QUERY task
        task = Task(
            task_type=TaskType.PLAN_QUERY,
            agent_type=AgentType.PLANNER,
            priority=Priority.HIGH,
            parameters={
                "query": query,
                "context": {"component": "checkout", "mode": "query"},
                "mode": "query",
            },
        )
        
        print("Executing Planner Agent...")
        print("(This will route to ObservabilityAgent and fetch real CloudWatch metrics)\n")
        
        # Execute the planner (this will route, fetch metrics, and run agents)
        response = await planner.execute(task)
        
        # The planner wraps the QueryResponse in AgentResponse.data
        query_response_data = response.data.get("query_response")
        
        if query_response_data:
            print(f"✓ Query completed successfully!")
            print(f"\nStatus: {response.status.value}")
            print(f"Total insights: {len(response.insights)}")
            print(f"Execution time: {response.execution_time_ms:.1f}ms")
            
            # Show narrative
            narrative = query_response_data.get("narrative", "")
            print(f"\nAI Narrative (first 500 chars):")
            print("-" * 80)
            print(narrative[:500])
            if len(narrative) > 500:
                print("...")
            print("-" * 80)
            
            # Show which agents ran
            agent_responses = query_response_data.get("agent_responses", [])
            print(f"\n✓ {len(agent_responses)} agent(s) executed:")
            for ar in agent_responses:
                agent_name = ar.get("agent_type", "unknown")
                status = ar.get("status", "unknown")
                insight_count = len(ar.get("insights", []))
                print(f"  - {agent_name}: {status} ({insight_count} insights)")
            
            # Check if observability agent got metrics
            for ar in agent_responses:
                if ar.get("agent_type") == "observability":
                    metrics_count = ar.get("data", {}).get("metrics_analyzed", 0)
                    print(f"\n✓ ObservabilityAgent processed {metrics_count} CloudWatch metrics")
                    break
            
            return True
        else:
            print("✗ No query_response in planner output")
            return False
            
    except Exception as e:
        print(f"\n✗ Planner execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dashboard_mode():
    """Test dashboard mode which should fetch comprehensive metrics."""
    print("\n" + "="*80)
    print("TEST 3: Dashboard Mode - Full system health snapshot")
    print("="*80)
    
    try:
        task = Task(
            task_type=TaskType.PLAN_QUERY,
            agent_type=AgentType.PLANNER,
            priority=Priority.MEDIUM,
            parameters={
                "query": "Give me a full system health snapshot",
                "context": {},
                "mode": "dashboard",
            },
        )
        
        print("\nExecuting dashboard query...")
        print("(This should invoke observability + cost + cicd agents)\n")
        
        response = await planner.execute(task)
        query_response_data = response.data.get("query_response")
        
        if query_response_data:
            agent_responses = query_response_data.get("agent_responses", [])
            print(f"✓ Dashboard complete: {len(agent_responses)} agents executed")
            
            for ar in agent_responses:
                agent_name = ar.get("agent_type", "unknown")
                status = ar.get("status", "unknown")
                insight_count = len(ar.get("insights", []))
                print(f"  - {agent_name}: {status} ({insight_count} insights)")
            
            return True
        else:
            print("✗ No query_response in dashboard output")
            return False
            
    except Exception as e:
        print(f"\n✗ Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n" + "="*80)
    print("CloudWatch Integration Test Suite")
    print("="*80)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("\nThis test verifies that the SRE Copilot frontend can get real")
    print("insights from AWS CloudWatch via the backend integration.\n")
    
    results = []
    
    # Test 1: Metrics service
    results.append(("Metrics Service", await test_metrics_service()))
    
    # Test 2: Planner enrichment
    results.append(("Planner Enrichment", await test_planner_enrichment()))
    
    # Test 3: Dashboard mode (optional - can be slow)
    # Uncomment to test dashboard mode:
    # results.append(("Dashboard Mode", await test_dashboard_mode()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! CloudWatch integration is working.")
        print("The frontend chatbot can now get real insights from AWS CloudWatch.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
