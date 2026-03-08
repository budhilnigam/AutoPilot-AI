"""integrations/aws/ecs.py — ECS client using aws_api shim.

Provides async helpers to describe services, list services, fetch task definitions,
and update desired counts. Uses centralized `aws_api` to access boto3.
"""

from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from autopilot_ai.core.exceptions import ECSError, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.integrations.aws.tool import aws_api
from autopilot_ai.models.domain import ECSServiceConfig

logger = get_logger(__name__)


class ECSClient:
    def __init__(self) -> None:
        pass

    @with_retry(retry_on=(ThrottlingError,))
    async def _describe_services(self, cluster: str, services: list[str]) -> list[dict[str, Any]]:
        try:
            resp = await aws_api("ecs", "describe_services", {"cluster": cluster, "services": services, "include": ["TAGS"]})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"ECS throttled: {e}", cluster=cluster) from e
            raise ECSError(f"ECS describe_services failed: {e}", cluster=cluster) from e

        failures = resp.get("failures", [])
        if failures:
            reasons = [f.get("reason") for f in failures]
            raise ECSError(f"ECS describe_services partial failure: {reasons}", cluster=cluster, services=services)

        return resp.get("services", [])

    @with_retry(retry_on=(ThrottlingError,))
    async def _describe_task_definition(self, task_def: str) -> dict[str, Any]:
        try:
            resp = await aws_api("ecs", "describe_task_definition", {"taskDefinition": task_def, "include": ["TAGS"]})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"ECS throttled on task def: {e}", task_def=task_def) from e
            raise ECSError(f"ECS describe_task_definition failed: {e}", task_def=task_def) from e
        return resp.get("taskDefinition", {})

    @with_retry(retry_on=(ThrottlingError,))
    async def _list_services(self, cluster: str) -> list[str]:
        # Use list_services with pagination
        arns: list[str] = []
        params = {"cluster": cluster, "maxResults": 100}
        try:
            while True:
                resp = await aws_api("ecs", "list_services", params)
                arns.extend(resp.get("serviceArns", []))
                token = resp.get("nextToken") or resp.get("nextToken")
                if not token:
                    break
                params["nextToken"] = token
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"ECS list_services throttled: {e}") from e
            raise ECSError(f"ECS list_services failed: {e}", cluster=cluster) from e
        return arns

    @with_retry(retry_on=(ThrottlingError,))
    async def get_service_config(self, cluster: str, service_name: str) -> ECSServiceConfig:
        services = await self._describe_services(cluster, [service_name])

        if not services:
            raise ECSError(f"Service '{service_name}' not found in cluster '{cluster}'", cluster=cluster, service=service_name)

        svc = services[0]
        task_def_arn: str = svc.get("taskDefinition", "")

        task_def = {}
        if task_def_arn:
            task_def = await self._describe_task_definition(task_def_arn)

        containers = task_def.get("containerDefinitions", [])
        env_vars: dict[str, str] = {}
        for container in containers:
            for env in container.get("environment", []):
                name = env.get("name", "")
                value = env.get("value", "")
                if name:
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
        return await self._list_services(cluster)

    @with_retry(retry_on=(ThrottlingError,))
    async def get_task_definition(self, task_definition: str) -> dict[str, Any]:
        return await self._describe_task_definition(task_definition)

    @with_retry(retry_on=(ThrottlingError,))
    async def update_service_desired_count(self, cluster: str, service: str, desired_count: int) -> None:
        logger.info("ecs_update_desired_count", cluster=cluster, service=service, desired_count=desired_count)
        try:
            await aws_api("ecs", "update_service", {"cluster": cluster, "service": service, "desiredCount": desired_count})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(f"ECS throttled on update: {e}", cluster=cluster, service=service) from e
            raise ECSError(f"ECS update_service failed: {e}", cluster=cluster, service=service) from e


# Module-level singleton
ecs_client = ECSClient()
