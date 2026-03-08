"""AWS helper utilities for the application.

Provides two async helpers used across the integrations:

- `aws_cli(command: str) -> Any` — runs `aws <command> --output json` using
  an async subprocess and returns parsed JSON (or raw text on parse failure).
- `aws_api(service, operation, params) -> Any` — runs the equivalent boto3
  client operation inside a threadpool to avoid blocking the event loop.

Centralising boto3 usage here keeps the rest of the codebase free of
hardcoded clients while still using native Python AWS SDK.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any

import boto3
from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings


async def aws_cli(command: str | list[str]) -> Any:
    """Run `aws <command> --output json` asynchronously and return parsed JSON.

    Accepts either a single string (shell-like) or a list of argv-style tokens.

    Example: await aws_cli("s3api list-objects-v2 --bucket my-bucket --prefix foo/")
             await aws_cli(["s3api", "list-objects-v2", "--bucket", "my-bucket"])
    """
    # Build argv list when possible; if shlex fails due to unmatched quotes,
    # fall back to running the whole command in a shell so embedded JSON is
    # preserved (this avoids `No closing quotation` errors from shlex).
    if isinstance(command, list):
        parts = ["aws"] + command + ["--output", "json"]
        proc = await asyncio.create_subprocess_exec(
            *parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        try:
            parts = ["aws"] + shlex.split(command) + ["--output", "json"]
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except ValueError as e:
            # Likely 'No closing quotation' from shlex; run with shell=True instead
            shell_cmd = f"aws {command} --output json"
            proc = await asyncio.create_subprocess_shell(
                shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="ignore") or "aws cli failed")
    out = stdout.decode("utf-8")
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def _make_boto_client(service: str):
    # Prefer explicit access keys when present; otherwise fall back to profile.
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        return boto3.client(service, **kwargs)
    if settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client(service, **kwargs)
    return boto3.client(service, **kwargs)


async def aws_api(service: str, operation: str, params: dict | None = None) -> Any:
    """Run a boto3 client operation in a threadpool and return the result.

    Args:
        service: boto3 service name (e.g. 's3', 'cloudwatch').
        operation: client method name (e.g. 'get_metric_data').
        params: keyword args to pass to the client method.
    """
    loop = asyncio.get_running_loop()

    def _call():
        client = _make_boto_client(service)
        func = getattr(client, operation)
        # Special-case reading streaming bodies so callers get plain bytes
        resp = func(**(params or {}))
        # Normalize common streaming keys to plain bytes for callers.
        for key in ("Body", "body"):
            if isinstance(resp, dict) and key in resp:
                val = resp[key]
                # If it's a StreamingBody-like object, read it once here.
                if hasattr(val, "read"):
                    resp[key] = val.read()
        return resp
        return func(**(params or {}))

    try:
        return await loop.run_in_executor(None, _call)
    except ClientError:
        # Re-raise botocore errors to let callers handle specific codes
        raise
