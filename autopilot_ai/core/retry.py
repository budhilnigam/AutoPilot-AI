"""
core/retry.py — Retry decorator and Circuit Breaker for external API calls.

Two complementary resilience mechanisms:

1. @with_retry decorator (tenacity-based)
   - Retries on transient errors (throttling, timeouts)
   - Exponential backoff with jitter
   - Configurable via Settings

2. CircuitBreaker class
   - Prevents cascading failures when Bedrock is degraded
   - States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (probing)
   - Thread-safe via asyncio.Lock

Usage:
    # Retry on throttling:
    @with_retry(retry_on=(BedrockThrottlingError, ThrottlingError))
    async def call_bedrock(...): ...

    # Circuit breaker for an entire service:
    breaker = CircuitBreaker(name="bedrock")
    async with breaker:
        result = await bedrock_client.invoke(...)
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    wait_combine,
)

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import (
    AgentUnavailableError,
    AutoPilotError,
    BedrockThrottlingError,
    ThrottlingError,
)
from autopilot_ai.core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

# Default exception types that trigger a retry
_DEFAULT_RETRY_ON = (BedrockThrottlingError, ThrottlingError)


def with_retry(
    retry_on: tuple[type[Exception], ...] = _DEFAULT_RETRY_ON,
    max_attempts: int | None = None,
    min_wait: float | None = None,
    max_wait: float | None = None,
) -> Callable[[F], F]:
    """
    Decorator factory that adds exponential-backoff retry to an async function.

    Args:
        retry_on:     Exception types that trigger a retry. Defaults to
                      throttling errors only — do NOT retry on AgentError,
                      BedrockModelError etc. as those won't self-resolve.
        max_attempts: Override Settings.retry_max_attempts.
        min_wait:     Override Settings.retry_min_wait_seconds.
        max_wait:     Override Settings.retry_max_wait_seconds.

    Example:
        @with_retry(retry_on=(BedrockThrottlingError,))
        async def invoke_model(self, prompt: str) -> str: ...
    """
    attempts = max_attempts or settings.retry_max_attempts
    min_w = min_wait or settings.retry_min_wait_seconds
    max_w = max_wait or settings.retry_max_wait_seconds

    def decorator(func: F) -> F:
        @retry(
            retry=retry_if_exception_type(retry_on),
            stop=stop_after_attempt(attempts),
            wait=wait_combine(
                wait_exponential(multiplier=1, min=min_w, max=max_w),
                wait_random(min=0, max=1),  # jitter to avoid thundering herd
            ),
            reraise=True,  # re-raise original exception after all attempts
        )
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except RetryError as e:
                # tenacity wraps the last exception — unwrap it for clarity
                raise e.last_attempt.exception() from e

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Circuit Breaker ────────────────────────────────────────────────────────


class CircuitState(enum.Enum):
    CLOSED = "closed"      # Normal operation, calls flow through
    OPEN = "open"          # Failing, calls rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Async circuit breaker for protecting calls to external services.

    The circuit opens (stops calls) after `failure_threshold` consecutive
    failures. After `timeout_seconds` it enters HALF_OPEN and allows a
    single probe call. If the probe succeeds the circuit closes; if it
    fails the circuit re-opens.

    Usage as async context manager:
        breaker = CircuitBreaker(name="bedrock")

        async with breaker:
            result = await bedrock.invoke(...)

    Usage as decorator:
        @breaker.protect
        async def call_bedrock(): ...
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.name = name
        self.failure_threshold = (
            failure_threshold or settings.circuit_breaker_failure_threshold
        )
        self.timeout_seconds = (
            timeout_seconds or settings.circuit_breaker_timeout_seconds
        )

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def _check_and_transition(self) -> None:
        """Transition OPEN → HALF_OPEN if timeout has elapsed."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._last_failure_time or 0)
            if elapsed >= self.timeout_seconds:
                async with self._lock:
                    if self._state == CircuitState.OPEN:  # double-check
                        self._state = CircuitState.HALF_OPEN
                        logger.info(
                            "circuit_breaker_half_open",
                            breaker=self.name,
                            elapsed_seconds=round(elapsed, 1),
                        )

    async def __aenter__(self) -> "CircuitBreaker":
        await self._check_and_transition()
        if self._state == CircuitState.OPEN:
            raise AgentUnavailableError(
                f"Circuit breaker '{self.name}' is OPEN — service unavailable",
                breaker=self.name,
                last_failure_ago=round(
                    time.monotonic() - (self._last_failure_time or 0), 1
                ),
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        async with self._lock:
            if exc_type is not None and issubclass(exc_type, AutoPilotError):
                # Failure
                self._failure_count += 1
                self._last_failure_time = time.monotonic()

                if self._state == CircuitState.HALF_OPEN:
                    # Probe failed — re-open
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "circuit_breaker_reopened",
                        breaker=self.name,
                        failure_count=self._failure_count,
                    )
                elif self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        "circuit_breaker_opened",
                        breaker=self.name,
                        failure_count=self._failure_count,
                        threshold=self.failure_threshold,
                    )
                return False  # don't suppress the exception

            # Success
            if self._state == CircuitState.HALF_OPEN:
                # Probe succeeded — close circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("circuit_breaker_closed", breaker=self.name)
            elif self._state == CircuitState.CLOSED:
                # Reset failure streak on any success
                self._failure_count = 0

        return False

    def protect(self, func: F) -> F:
        """Decorator form: @breaker.protect"""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    def reset(self) -> None:
        """Manually reset to CLOSED state (useful in tests)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self._state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )
