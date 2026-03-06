#!/usr/bin/env python3
"""Quick health check tests for GitHub fix and CloudWatch integration."""

import asyncio
import sys

from autopilot_ai.core.logging import get_logger
from autopilot_ai.api.routes.health import _check_github, _check_bedrock

logger = get_logger(__name__)


async def test_github_check():
    """Test GitHub health check."""
    print("\n" + "="*60)
    print("Test 1: GitHub Health Check")
    print("="*60)
    
    try:
        result = await _check_github()
        print(f"Name: {result.name}")
        print(f"Healthy: {result.healthy}")
        print(f"Latency: {result.latency_ms}ms")
        print(f"Detail: {result.detail}")
        
        if result.healthy or "no token configured" in (result.detail or ""):
            print("✓ PASS: GitHub check working correctly")
            return True
        else:
            print(f"✗ FAIL: GitHub check unhealthy - {result.detail}")
            return False
    except Exception as e:
        print(f"✗ FAIL: Exception - {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_bedrock_check():
    """Test Bedrock health check."""
    print("\n" + "="*60)
    print("Test 2: Bedrock Health Check")
    print("="*60)
    
    try:
        result = await _check_bedrock()
        print(f"Name: {result.name}")
        print(f"Healthy: {result.healthy}")
        print(f"Latency: {result.latency_ms}ms")
        print(f"Detail: {result.detail or 'OK'}")
        
        if result.healthy:
            print("✓ PASS: Bedrock connection healthy")
            return True
        else:
            print(f"⚠ WARNING: Bedrock unhealthy - {result.detail}")
            print("(This may be expected if AWS credentials are not configured)")
            return False
    except Exception as e:
        print(f"✗ FAIL: Exception - {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_metrics_import():
    """Test that metrics service can be imported."""
    print("\n" + "="*60)
    print("Test 3: Metrics Service Import")
    print("="*60)
    
    try:
        from autopilot_ai.services.metrics_service import metrics_service
        print(f"✓ Metrics service imported successfully")
        print(f"  Type: {type(metrics_service).__name__}")
        
        # Try to access the client
        if hasattr(metrics_service, '_client'):
            print(f"  CloudWatch client: {type(metrics_service._client).__name__}")
        
        print("✓ PASS: Metrics service ready")
        return True
    except Exception as e:
        print(f"✗ FAIL: Could not import metrics service - {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("\n" + "="*60)
    print("Quick Health & Integration Tests")
    print("="*60)
    
    results = []
    
    # Run 3 tests max as requested
    results.append(("GitHub Check", await test_github_check()))
    results.append(("Bedrock Check", await test_bedrock_check()))
    results.append(("Metrics Service", await test_metrics_import()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")
    print("="*60 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
