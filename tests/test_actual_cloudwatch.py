#!/usr/bin/env python3
"""
Test actual CloudWatch connectivity and metric fetching.
This will tell us exactly what's failing.
"""

import asyncio
from datetime import datetime, timezone

from autopilot_ai.integrations.aws.cloudwatch import cloudwatch_client
from autopilot_ai.services.metrics_service import metrics_service


async def test_cloudwatch_connection():
    """Test if we can actually connect to CloudWatch."""
    print("\n" + "="*60)
    print("CloudWatch Connection Test")
    print("="*60)
    
    try:
        # Try to fetch a simple metric
        print("\nAttempting to fetch ECS CPU metrics...")
        print("(This will try to connect to AWS CloudWatch)")
        
        metric = await cloudwatch_client.get_metric_statistics(
            namespace="AWS/ECS",
            metric_name="CPUUtilization",
            dimensions={},  # No specific dimensions - get any ECS metrics
            period_seconds=300,
            lookback_minutes=60,
            statistic="Average",
        )
        
        if metric and metric.datapoints:
            print(f"✓ SUCCESS: Retrieved {len(metric.datapoints)} data points")
            print(f"  Latest value: {metric.latest_value}")
            return True
        else:
            print("⚠ Connected but no data points found")
            print("  (This is OK if you don't have ECS running)")
            return True
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        print(f"\nError type: {type(e).__name__}")
        
        if "credentials" in str(e).lower():
            print("\n💡 This is a CREDENTIALS issue")
            print("   Your AWS credentials are not configured or invalid")
        elif "unauthorized" in str(e).lower() or "forbidden" in str(e).lower():
            print("\n💡 This is a PERMISSIONS issue")
            print("   Your AWS credentials don't have CloudWatch permissions")
        elif "timeout" in str(e).lower():
            print("\n💡 This is a NETWORK issue")
            print("   Cannot reach AWS CloudWatch")
        else:
            print("\n💡 This is an UNKNOWN issue")
            
        import traceback
        traceback.print_exc()
        return False


async def test_metrics_service():
    """Test the metrics service query logic."""
    print("\n" + "="*60)
    print("Metrics Service Query Test")
    print("="*60)
    
    try:
        print("\nFetching metrics for query: 'show me system health'")
        
        metrics = await metrics_service.fetch_metrics_for_query(
            query="show me system health",
            context={},
            lookback_minutes=60,
        )
        
        if metrics:
            print(f"✓ Retrieved {len(metrics)} metrics")
            for m in metrics[:3]:
                print(f"  - {m.namespace}/{m.metric_name}")
            return True
        else:
            print("⚠ No metrics retrieved (but no error)")
            print("  This could mean:")
            print("  - AWS credentials not configured")
            print("  - No resources running in your AWS account")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n" + "="*60)
    print("Actual CloudWatch Connectivity Test")
    print("="*60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("\nThis will attempt REAL AWS CloudWatch API calls\n")
    
    results = []
    
    print("This test will tell you WHY the chatbot says it can't connect.\n")
    
    results.append(("CloudWatch Connection", await test_cloudwatch_connection()))
    results.append(("Metrics Service Query", await test_metrics_service()))
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    for name, passed in results:
        status = "✓ WORKING" if passed else "✗ NOT WORKING"
        print(f"{status}: {name}")
    
    if all(r[1] for r in results):
        print("\n✓ CloudWatch is accessible!")
        print("  The chatbot should be able to fetch metrics.")
        print("\n  If it's still saying it can't connect, the issue is:")
        print("  1. Backend server not running with new code, OR")
        print("  2. Claude is misinterpreting your question")
    else:
        print("\n✗ CloudWatch is NOT accessible")
        print("\n  Fix required:")
        print("  1. Configure AWS credentials in .env file")
        print("  2. Ensure credentials have CloudWatch:GetMetricStatistics permission")
        print("  3. Restart the backend server")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
