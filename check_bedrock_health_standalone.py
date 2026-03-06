"""
Standalone Bedrock dependency health check.

Run:
    python check_bedrock_health_standalone.py

Environment:
    - Reads AWS_REGION from process env or .env file (if present)
    - Falls back to ap-south-1 if AWS_REGION is not set
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import boto3

_DEFAULT_CHECK_TIMEOUT = 8.0


@dataclass
class DependencyStatus:
    name: str
    healthy: bool
    latency_ms: float
    detail: str | None = None


def _load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    """Minimal .env loader to keep this file self-contained."""
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


async def check_bedrock() -> DependencyStatus:
    """Verify we can reach Bedrock (ListFoundationModels free call)."""
    t0 = time.monotonic()
    region = os.getenv("AWS_REGION", "ap-south-1")
    check_timeout = float(os.getenv("BEDROCK_HEALTH_TIMEOUT_SECONDS", str(_DEFAULT_CHECK_TIMEOUT)))

    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("aws_access_key_id")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("aws_secret_access_key")
    profile = os.getenv("AWS_PROFILE") or os.getenv("aws_profile")

    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _call_bedrock(region, access_key, secret_key, profile),
            ),
            timeout=check_timeout,
        )
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(name="bedrock", healthy=True, latency_ms=round(latency, 1))
    
    except asyncio.TimeoutError:
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="bedrock",
            healthy=False,
            latency_ms=round(latency, 1),
            detail=f"timed out after {check_timeout:.1f}s",
        )
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return DependencyStatus(
            name="bedrock",
            healthy=False,
            latency_ms=round(latency, 1),
            detail=str(e)[:120],
        )


def _call_bedrock(
    region: str,
    access_key: str | None,
    secret_key: str | None,
    profile: str | None,
) -> None:
    """Build a boto session and call Bedrock using deterministic credential precedence."""
    session_kwargs: dict[str, str] = {}

    # Prefer explicit access keys over profile to avoid "profile not found" failures.
    if access_key and secret_key:
        session_kwargs["aws_access_key_id"] = access_key
        session_kwargs["aws_secret_access_key"] = secret_key
    elif profile:
        session_kwargs["profile_name"] = profile

    # Botocore can still honor AWS_PROFILE from process env. When explicit keys
    # are provided, temporarily clear profile env vars for this call.
    restore_aws_profile = os.environ.pop("AWS_PROFILE", None) if access_key and secret_key else None
    restore_aws_profile_lower = (
        os.environ.pop("aws_profile", None) if access_key and secret_key else None
    )

    try:
        session = boto3.Session(**session_kwargs)
        client = session.client("bedrock", region_name=region)
        client.list_foundation_models(byOutputModality="TEXT")
    finally:
        if restore_aws_profile is not None:
            os.environ["AWS_PROFILE"] = restore_aws_profile
        if restore_aws_profile_lower is not None:
            os.environ["aws_profile"] = restore_aws_profile_lower


async def main() -> None:
    _load_dotenv_if_present()
    status = await check_bedrock()
    print(json.dumps(asdict(status), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
