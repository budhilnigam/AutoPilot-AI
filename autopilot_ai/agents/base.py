"""
agents/base.py — Abstract base class for all specialized agents.

Every agent (Observability, Infra, DB, Cost, CICD, ToolGenerator) inherits
from BaseAgent and implements a single method: execute(task) -> AgentResponse.

The __call__ template method handles everything that should happen exactly once,
regardless of which agent is running:

  1. Timing — records wall-clock execution time in ms
  2. Correlation ID logging — binds agent_type + task_id so all log lines
     inside execute() carry those fields automatically
  3. Deadline enforcement — raises AgentTimeoutError if task.deadline is exceeded
  4. Circuit breaker — each agent instance carries its own CircuitBreaker;
     if the breaker is OPEN the call is short-circuited to _fallback immediately
  5. Error wrapping — any bare exception from execute() is wrapped into an
     AgentResponse with status=FAILED so the Planner always gets a typed response
  6. Fallback — on error returns PARTIAL status with a heuristic explanation
     so the Planner can still synthesize a partial answer for the user

Agents MUST NOT re-implement timing, logging, or error wrapping.
All of that lives here, once.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import ClassVar

from autopilot_ai.core.exceptions import (
    AgentError,
    AgentTimeoutError,
    AgentUnavailableError,
    AutoPilotError,
)
from autopilot_ai.core.logging import (
    bind_agent_context,
    clear_agent_context,
    get_logger,
)
from autopilot_ai.core.retry import CircuitBreaker
from autopilot_ai.models.insights import Insight, InsightCategory, Urgency
from autopilot_ai.models.responses import AgentResponse, ResponseStatus
from autopilot_ai.models.tasks import AgentType, Task

logger = get_logger(__name__)

# How long (seconds) an agent task is allowed to run before timeout,
# if the task has no explicit deadline set.
_DEFAULT_TIMEOUT_SECONDS = 60.0


class BaseAgent(ABC):
    """
    Abstract base for all AutoPilot-AI specialized agents.

    Subclasses implement:
        agent_type: AgentType          — class-level declaration
        execute(task) -> AgentResponse — the actual domain logic

    Usage:
        class ObservabilityAgent(BaseAgent):
            agent_type = AgentType.OBSERVABILITY

            async def execute(self, task: Task) -> AgentResponse:
                ...

        agent = ObservabilityAgent()
        response = await agent(task)       # goes through __call__ template
    """

    # Every subclass MUST declare this at class level
    agent_type: ClassVar[AgentType]

    def __init__(self) -> None:
        # Each agent gets its own circuit breaker so a failing DB agent
        # does not affect the Cost agent's breaker state.
        self._circuit_breaker = CircuitBreaker(name=self._agent_type_name())

    # ── Helpers ────────────────────────────────────────────────────────────

    def _agent_type_name(self) -> str:
        """Return the agent type string, safe to call from __init__."""
        try:
            return self.agent_type.value
        except AttributeError:
            return type(self).__name__.lower()

    def _make_failed_response(
        self,
        task: Task,
        error: Exception,
        execution_time_ms: float,
        status: ResponseStatus = ResponseStatus.FAILED,
    ) -> AgentResponse:
        """Build a FAILED AgentResponse from an exception."""
        return AgentResponse(
            agent_type=task.agent_type,
            task_id=task.id,
            status=status,
            execution_time_ms=execution_time_ms,
            error_message=str(error),
        )

    # ── Fallback ───────────────────────────────────────────────────────────

    async def _fallback(
        self, task: Task, error: Exception, execution_time_ms: float
    ) -> AgentResponse:
        """
        Called when execute() raises any exception.

        Returns a PARTIAL AgentResponse with a single heuristic Insight so
        the Planner can still include this agent's perspective in its synthesis.
        The Planner's synthesis prompt will see status=PARTIAL and adjust
        its confidence accordingly.

        Subclasses can override to provide domain-specific fallback logic,
        but the returned AgentResponse MUST have status=PARTIAL.
        """
        fallback_insight = Insight(
            category=InsightCategory.PERFORMANCE,
            component=self._agent_type_name(),
            title=f"{self._agent_type_name().capitalize()} agent could not complete analysis",
            business_context=(
                f"The {self._agent_type_name()} agent encountered an error "
                f"and returned a partial result. Manual review recommended. "
                f"Error: {type(error).__name__}: {error}"
            ),
            urgency=Urgency.MEDIUM,
            confidence=0.0,
        )
        return AgentResponse(
            agent_type=task.agent_type,
            task_id=task.id,
            status=ResponseStatus.PARTIAL,
            execution_time_ms=execution_time_ms,
            insights=[fallback_insight],
            error_message=str(error),
        )

    # ── Template method ────────────────────────────────────────────────────

    async def __call__(self, task: Task) -> AgentResponse:
        """
        Invoke the agent with full instrumentation.

        This is the only entry point for agent execution.
        Never call execute() directly — always use agent(task).

        Flow:
          circuit breaker check
            → bind logging context
              → deadline enforcement (asyncio.wait_for)
                → execute(task)
              → timing
            → unbind logging context
          → return AgentResponse
        """
        start = time.monotonic()

        # ── Circuit breaker guard ──────────────────────────────────────────
        if self._circuit_breaker.is_open:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            logger.warning(
                "agent_circuit_open",
                agent=self._agent_type_name(),
                task_id=task.id,
            )
            return AgentResponse(
                agent_type=task.agent_type,
                task_id=task.id,
                status=ResponseStatus.FAILED,
                execution_time_ms=elapsed,
                error_message=f"Circuit breaker for {self._agent_type_name()} is OPEN",
            )

        # ── Log context binding ────────────────────────────────────────────
        bind_agent_context(agent_type=self._agent_type_name(), task_id=task.id)

        try:
            logger.info(
                "agent_started",
                agent=self._agent_type_name(),
                task_type=task.task_type.value,
                priority=task.priority.value,
            )

            # ── Deadline enforcement ───────────────────────────────────────
            # Use explicit deadline if set, otherwise fall back to default timeout
            if task.deadline:
                timeout = (task.deadline - __import__("datetime").datetime.utcnow()).total_seconds()
                if timeout <= 0:
                    raise AgentTimeoutError(
                        f"Task deadline already passed",
                        agent_type=self._agent_type_name(),
                        task_id=task.id,
                    )
            else:
                timeout = _DEFAULT_TIMEOUT_SECONDS

            # ── Execute with circuit breaker and timeout ───────────────────
            async with self._circuit_breaker:
                response = await asyncio.wait_for(
                    self.execute(task),
                    timeout=timeout,
                )

            execution_time_ms = round((time.monotonic() - start) * 1000, 2)
            response.execution_time_ms = execution_time_ms

            logger.info(
                "agent_completed",
                agent=self._agent_type_name(),
                task_type=task.task_type.value,
                status=response.status.value,
                execution_time_ms=execution_time_ms,
                insights_count=len(response.insights),
            )
            return response

        except asyncio.TimeoutError:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            err = AgentTimeoutError(
                f"Agent '{self._agent_type_name()}' timed out after {timeout:.1f}s",
                agent_type=self._agent_type_name(),
                task_id=task.id,
            )
            logger.error(
                "agent_timeout",
                agent=self._agent_type_name(),
                task_id=task.id,
                timeout_seconds=timeout,
            )
            return AgentResponse(
                agent_type=task.agent_type,
                task_id=task.id,
                status=ResponseStatus.TIMEOUT,
                execution_time_ms=elapsed,
                error_message=str(err),
            )

        except AgentUnavailableError as e:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            logger.error(
                "agent_unavailable",
                agent=self._agent_type_name(),
                task_id=task.id,
                error=str(e),
            )
            return self._make_failed_response(task, e, elapsed)

        except AutoPilotError as e:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            logger.error(
                "agent_error",
                agent=self._agent_type_name(),
                task_id=task.id,
                error_type=type(e).__name__,
                error=str(e),
            )
            return await self._fallback(task, e, elapsed)

        except Exception as e:
            elapsed = round((time.monotonic() - start) * 1000, 2)
            logger.exception(
                "agent_unexpected_error",
                agent=self._agent_type_name(),
                task_id=task.id,
                error_type=type(e).__name__,
            )
            return await self._fallback(task, e, elapsed)

        finally:
            clear_agent_context()

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    async def execute(self, task: Task) -> AgentResponse:
        """
        Domain-specific agent logic.  Implement this in every subclass.

        Contract:
          - Receives a Task with agent_type, task_type, parameters, context
          - Returns an AgentResponse with status SUCCESS (or PARTIAL if
            best-effort results are available despite partial failure)
          - MUST NOT catch and swallow AutoPilotError — let it propagate
            to __call__ for proper fallback and circuit-breaker accounting
          - SHOULD query knowledge_base.query_context() for relevant context
            before invoking Bedrock
          - timing, logging, and error wrapping are handled by __call__

        Args:
            task: The task dispatched by the Planner Agent.

        Returns:
            AgentResponse — do NOT set execution_time_ms; __call__ sets it.
        """
        ...

    # ── Utilities for subclasses ───────────────────────────────────────────

    def _success(
        self,
        task: Task,
        insights: list[Insight] | None = None,
        data: dict | None = None,
        model_used: str | None = None,
    ) -> AgentResponse:
        """
        Convenience method to build a SUCCESS response.

        Subclasses should use this instead of constructing AgentResponse manually.
        execution_time_ms is set to 0 here and overwritten by __call__.
        """
        return AgentResponse(
            agent_type=task.agent_type,
            task_id=task.id,
            status=ResponseStatus.SUCCESS,
            execution_time_ms=0.0,  # overwritten by __call__
            insights=insights or [],
            data=data or {},
            model_used=model_used,
        )

    def _partial(
        self,
        task: Task,
        insights: list[Insight] | None = None,
        data: dict | None = None,
        error_message: str = "",
    ) -> AgentResponse:
        """Convenience method to build a PARTIAL response."""
        return AgentResponse(
            agent_type=task.agent_type,
            task_id=task.id,
            status=ResponseStatus.PARTIAL,
            execution_time_ms=0.0,
            insights=insights or [],
            data=data or {},
            error_message=error_message,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"agent_type={self._agent_type_name()!r}, "
            f"circuit={self._circuit_breaker.state.value})"
        )
