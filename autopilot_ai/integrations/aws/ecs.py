"""
integrations/aws/ecs.py — ECS client for task definitions and service configs.

Provides:
  get_service_config(cluster, service)      -> ECSServiceConfig
  list_services(cluster)                    -> list[str]
  get_task_definition(task_def_arn)         -> dict
  update_service_desired_count(...)         -> None  (used by Infra Agent recommendations)

All methods are async via run_in_executor.
Raises ECSError on API failures, ThrottlingError on throttling.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import boto3
from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import ECSError, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.models.domain import ECSServiceConfig

logger = get_logger(__name__)


def _make_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client("ecs", **kwargs)
    return boto3.client("ecs", **kwargs)


class ECSClient:
    """
    Async wrapper around boto3 ECS client.

    Usage:
        ecs = ECSClient()
        config = await ecs.get_service_config("my-cluster", "my-service")
    """

    def __init__(self) -> None:
        self._client = _make_client()

    # ── Sync helpers ───────────────────────────────────────────────────────

    def _describe_services_sync(
        self, cluster: str, services: list[str]
    ) -> list[dict[str, Any]]:
        try:
            response = self._client.describe_services(
                cluster=cluster, services=services, include=["TAGS"]
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(
                    f"ECS throttled: {e}", cluster=cluster
                ) from e
            raise ECSError(
                f"ECS describe_services failed: {e}", cluster=cluster
            ) from e

        failures = response.get("failures", [])
        if failures:
            reasons = [f["reason"] for f in failures]
            raise ECSError(
                f"ECS describe_services partial failure: {reasons}",
                cluster=cluster,
                services=services,
            )

        return response.get("services", [])

    def _describe_task_definition_sync(self, task_def: str) -> dict[str, Any]:
        try:
            response = self._client.describe_task_definition(
                taskDefinition=task_def, include=["TAGS"]
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(
                    f"ECS throttled on task def: {e}", task_def=task_def
                ) from e
            raise ECSError(
                f"ECS describe_task_definition failed: {e}", task_def=task_def
            ) from e
        return response.get("taskDefinition", {})

    def _list_services_sync(self, cluster: str) -> list[str]:
        arns: list[str] = []
        paginator = self._client.get_paginator("list_services")
        try:
            for page in paginator.paginate(cluster=cluster):
                arns.extend(page.get("serviceArns", []))
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"ECS list_services throttled: {e}") from e
            raise ECSError(f"ECS list_services failed: {e}", cluster=cluster) from e
        return arns

    def _update_service_sync(
        self, cluster: str, service: str, desired_count: int
    ) -> None:
        try:
            self._client.update_service(
                cluster=cluster, service=service, desiredCount=desired_count
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(
                    f"ECS throttled on update: {e}", cluster=cluster, service=service
                ) from e
            raise ECSError(
                f"ECS update_service failed: {e}", cluster=cluster, service=service
            ) from e

    # ── Public async API ───────────────────────────────────────────────────

    @with_retry(retry_on=(ThrottlingError,))
    async def get_service_config(
        self, cluster: str, service_name: str
    ) -> ECSServiceConfig:
        """
        Fetch ECS service configuration and its active task definition.

        Returns an ECSServiceConfig with container definitions, CPU/memory
        allocations, and current running/pending counts.
        """
        loop = asyncio.get_running_loop()
        services = await loop.run_in_executor(
            None,
            partial(self._describe_services_sync, cluster, [service_name]),
        )

        if not services:
            raise ECSError(
                f"Service '{service_name}' not found in cluster '{cluster}'",
                cluster=cluster,
                service=service_name,
            )

        svc = services[0]
        task_def_arn: str = svc.get("taskDefinition", "")

        # Fetch task definition to get container specs
        task_def = {}
        if task_def_arn:
            task_def = await loop.run_in_executor(
                None,
                partial(self._describe_task_definition_sync, task_def_arn),
            )

        containers = task_def.get("containerDefinitions", [])
        # Collect all environment variables from all containers
        env_vars: dict[str, str] = {}
        for container in containers:
            for env in container.get("environment", []):
                name = env.get("name", "")
                value = env.get("value", "")
                if name:
                    # Do NOT log values — they may contain secrets
                    env_vars[name] = value

        return ECSServiceConfig(
            cluster_name=cluster,
            service_name=service_name,
            task_definition_arn=task_def_arn,
            desired_count=svc.get("desiredCount", 0),
            running_count=svc.get("runningCount", 0),
            pending_count=svc.get("pendingCount", 0),
            cpu_units=int(task_def.get("cpu", 0) or 0),
            memory_mb=int(task_def.get("memory", 0) or 0),
            container_definitions=containers,
            environment_variables=env_vars,
        )

    @with_retry(retry_on=(ThrottlingError,))
    async def list_services(self, cluster: str) -> list[str]:
        """Return all service ARNs in the given ECS cluster (paginated)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._list_services_sync, cluster)
        )

    @with_retry(retry_on=(ThrottlingError,))
    async def get_task_definition(self, task_definition: str) -> dict[str, Any]:
        """
        Fetch raw task definition JSON dict.
        Useful for Infrastructure drift detection (Property 8).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._describe_task_definition_sync, task_definition)
        )

    @with_retry(retry_on=(ThrottlingError,))
    async def update_service_desired_count(
        self, cluster: str, service: str, desired_count: int
    ) -> None:
        """
        Update a service's desired task count.

        Only called when the Infra Agent recommendation is explicitly
        approved — this method modifies live infrastructure.
        """
        logger.info(
            "ecs_update_desired_count",
            cluster=cluster,
            service=service,
            desired_count=desired_count,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._update_service_sync, cluster, service, desired_count),
        )


# Module-level singleton
ecs_client = ECSClient()
