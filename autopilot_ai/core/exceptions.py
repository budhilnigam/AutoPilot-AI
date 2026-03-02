"""
core/exceptions.py — Custom exception hierarchy for AutoPilot-AI.

Using typed exceptions instead of bare Exception means:
  - Retry logic can target ThrottlingError specifically
  - Circuit breakers can open on BedrockError only
  - API layer can produce correct HTTP status codes per exception type
  - Logs always include structured context (agent_type, task_id, etc.)
"""

from __future__ import annotations


class AutoPilotError(Exception):
    """
    Base exception for all AutoPilot-AI errors.

    Always carry a human-readable message and optional structured context
    so structlog can emit them as key-value pairs.
    """

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


# ── Bedrock / LLM Errors ───────────────────────────────────────────────────


class BedrockError(AutoPilotError):
    """Raised when a Bedrock model invocation fails for any reason."""


class BedrockThrottlingError(BedrockError):
    """
    Raised on Bedrock ThrottlingException / TooManyRequestsException.
    The retry decorator specifically targets this subclass.
    """


class BedrockModelError(BedrockError):
    """Raised when the model returns an error response (e.g. content filter)."""


class BedrockParseError(BedrockError):
    """
    Raised when Claude's response cannot be parsed into the expected
    Pydantic model. Indicates a prompt engineering issue.
    """

    def __init__(self, message: str, raw_response: str, **context: object) -> None:
        super().__init__(message, **context)
        self.raw_response = raw_response


# ── Integration / External Service Errors ─────────────────────────────────


class IntegrationError(AutoPilotError):
    """Base for all external service integration failures."""


class AWSError(IntegrationError):
    """Raised when any AWS API call fails (non-Bedrock services)."""


class ThrottlingError(AWSError):
    """AWS API throttling — triggers retry logic."""


class CloudWatchError(AWSError):
    """CloudWatch-specific errors (missing metrics, malformed namespaces)."""


class ECSError(AWSError):
    """ECS API errors (missing task definitions, service not found)."""


class BillingError(AWSError):
    """Cost Explorer API errors."""


class S3Error(AWSError):
    """S3 read/write errors."""


class GitHubError(IntegrationError):
    """GitHub API errors — rate limits, auth failures, not found."""


class GitHubRateLimitError(GitHubError):
    """GitHub API rate limit exhausted. Poller should back off."""


# ── Agent Errors ───────────────────────────────────────────────────────────


class AgentError(AutoPilotError):
    """Base for agent execution failures."""

    def __init__(
        self,
        message: str,
        agent_type: str | None = None,
        task_id: str | None = None,
        **context: object,
    ) -> None:
        super().__init__(message, agent_type=agent_type, task_id=task_id, **context)
        self.agent_type = agent_type
        self.task_id = task_id


class AgentTimeoutError(AgentError):
    """Agent did not complete within the allotted deadline."""


class AgentUnavailableError(AgentError):
    """Agent is in circuit-breaker OPEN state or unregistered."""


class CircularDependencyError(AgentError):
    """Planner detected a circular dependency in agent execution order."""


class PlannerError(AgentError):
    """Planner Agent-specific errors (routing, synthesis)."""


# ── Tool Generation Errors ─────────────────────────────────────────────────


class ToolGenerationError(AutoPilotError):
    """Raised when Claude fails to generate valid tool code."""


class ToolValidationError(ToolGenerationError):
    """
    Raised when generated code fails AST validation.
    Carries the offending code for logging.
    """

    def __init__(self, message: str, generated_code: str, **context: object) -> None:
        super().__init__(message, **context)
        self.generated_code = generated_code


class ToolExecutionError(ToolGenerationError):
    """Raised when a generated tool fails during subprocess execution."""

    def __init__(
        self,
        message: str,
        returncode: int,
        stderr: str,
        **context: object,
    ) -> None:
        super().__init__(message, returncode=returncode, **context)
        self.returncode = returncode
        self.stderr = stderr


# ── Knowledge Base Errors ──────────────────────────────────────────────────


class KnowledgeBaseError(AutoPilotError):
    """Knowledge Base retrieval or storage failures."""


class KBNotConfiguredError(KnowledgeBaseError):
    """
    Raised when a KB operation is attempted but KNOWLEDGE_BASE_ID /
    S3_BUCKET_NAME are not set in config. Agents should fall back to
    operating without RAG context.
    """


# ── Validation Errors ──────────────────────────────────────────────────────


class DataValidationError(AutoPilotError):
    """Input data failed schema validation before reaching an agent."""


class ConfigurationError(AutoPilotError):
    """Missing or invalid application configuration."""
