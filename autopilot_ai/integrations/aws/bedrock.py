"""
integrations/aws/bedrock.py — Async wrapper around Amazon Bedrock Runtime.

Two public methods:
  invoke(prompt, model_id)                  -> str
  invoke_with_schema(prompt, schema)        -> BaseModel subclass

Uses boto3's synchronous client via asyncio.get_event_loop().run_in_executor
so the event loop is never blocked.

All calls are protected by:
  - @with_retry for ThrottlingError / BedrockThrottlingError (3 attempts, exp backoff)
  - A shared CircuitBreaker so a degraded Bedrock endpoint doesn't cascade

The Converse API is used instead of InvokeModel because it handles
provider-specific message formatting automatically and works across
Claude, Titan, and other models with a single call shape.
"""

from __future__ import annotations

import asyncio
import json
import re
from functools import partial
from typing import TYPE_CHECKING, Any, Type, TypeVar

import boto3
from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import (
    BedrockError,
    BedrockModelError,
    BedrockParseError,
    BedrockThrottlingError,
)
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import CircuitBreaker, with_retry

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = get_logger(__name__)

T = TypeVar("T", bound="BaseModel")

# One shared circuit breaker for the entire Bedrock service.
# Opened after 3 consecutive AutoPilotErrors; resets after 60s.
_bedrock_circuit_breaker = CircuitBreaker(name="bedrock")


def _make_client() -> Any:
    """Build a boto3 bedrock-runtime client from config."""
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client(
            "bedrock-runtime", **kwargs
        )
    return boto3.client("bedrock-runtime", **kwargs)


class BedrockClient:
    """
    Async Bedrock client.  Instantiate once and reuse (it is not thread-unsafe;
    boto3 clients are safe to share across threads when used via run_in_executor).

    Example:
        client = BedrockClient()
        answer = await client.invoke("What is CPU utilization?")
        model = await client.invoke_with_schema("Extract data:", MyModel)
    """

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id or settings.bedrock_model_id
        self._client = _make_client()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily get the running event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.get_event_loop()

    def _converse_sync(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None,
        model_id: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Synchronous Bedrock Converse call — runs inside run_in_executor.

        Returns the plain text content of the first assistant message.
        Raises BedrockThrottlingError, BedrockModelError, or BedrockError.
        """
        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        try:
            response = self._client.converse(**kwargs)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "TooManyRequestsException"):
                raise BedrockThrottlingError(
                    f"Bedrock throttled: {e}", model_id=model_id
                ) from e
            if code in ("ModelErrorException", "ModelNotReadyException"):
                raise BedrockModelError(
                    f"Model error: {e}", model_id=model_id
                ) from e
            raise BedrockError(f"Bedrock ClientError: {e}", model_id=model_id) from e
        except Exception as e:
            raise BedrockError(
                f"Unexpected Bedrock error: {e}", model_id=model_id
            ) from e

        # Extract text from Converse response shape
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text" or "text" in block:
                return block["text"]

        raise BedrockModelError(
            "Bedrock response contained no text content",
            model_id=model_id,
            response_keys=list(response.keys()),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @with_retry(retry_on=(BedrockThrottlingError,))
    async def invoke(
        self,
        prompt: str,
        model_id: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Send a prompt to Bedrock and return the plain-text response.

        Args:
            prompt:        User message content.
            model_id:      Override the default model.
            system_prompt: Optional system instruction for the model.
            max_tokens:    Override Settings.bedrock_max_tokens.
            temperature:   Override Settings.bedrock_temperature.

        Returns:
            The assistant's text response as a plain string.
        """
        mid = model_id or self._model_id
        mt = max_tokens or settings.bedrock_max_tokens
        temp = temperature if temperature is not None else settings.bedrock_temperature

        messages = [{"role": "user", "content": [{"text": prompt}]}]

        logger.debug("bedrock_invoke", model_id=mid, prompt_len=len(prompt))

        async with _bedrock_circuit_breaker:
            loop = self._get_loop()
            result = await loop.run_in_executor(
                None,
                partial(
                    self._converse_sync,
                    messages,
                    system_prompt,
                    mid,
                    mt,
                    temp,
                ),
            )

        logger.debug("bedrock_invoke_done", model_id=mid, response_len=len(result))
        return result

    @with_retry(retry_on=(BedrockThrottlingError,))
    async def invoke_with_schema(
        self,
        prompt: str,
        schema: Type[T],
        model_id: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """
        Send a prompt and parse Claude's JSON response directly into a Pydantic model.

        The system prompt instructs the model to respond ONLY with valid JSON
        matching the schema.  If parsing fails, BedrockParseError is raised
        (not retried — it's a prompt engineering issue, not a transient error).

        Args:
            prompt:        User message content.
            schema:        Pydantic BaseModel subclass to parse into.
            model_id:      Override the default model.
            system_prompt: Additional system instruction appended to the JSON directive.
            max_tokens:    Override Settings.bedrock_max_tokens.

        Returns:
            An instance of `schema` populated from Claude's JSON output.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        json_system = (
            "You are a precise data extraction assistant. "
            "Respond ONLY with valid JSON that conforms to this schema — no markdown, "
            "no explanation, no code fences:\n\n"
            f"{schema_json}"
        )
        if system_prompt:
            json_system = f"{json_system}\n\nAdditional instructions: {system_prompt}"

        raw = await self.invoke(
            prompt=prompt,
            model_id=model_id,
            system_prompt=json_system,
            max_tokens=max_tokens,
        )

        # Strip accidental markdown code fences if the model added them
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise BedrockParseError(
                f"Claude response is not valid JSON: {e}",
                raw_response=raw,
                schema=schema.__name__,
            ) from e

        try:
            return schema.model_validate(data)
        except Exception as e:
            raise BedrockParseError(
                f"JSON does not match schema {schema.__name__}: {e}",
                raw_response=raw,
                schema=schema.__name__,
            ) from e

    async def invoke_stream(
        self,
        prompt: str,
        model_id: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ):
        """
        Async generator that yields text chunks as they stream from Bedrock.

        Used by the /api/query route for SSE streaming (Property: real-time UX).
        Falls back to a single non-streaming call on models that don't support streaming.

        Yields:
            str  — each text chunk from the model
        """
        mid = model_id or self._model_id
        mt = max_tokens or settings.bedrock_max_tokens
        temp = settings.bedrock_temperature
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        kwargs: dict[str, Any] = {
            "modelId": mid,
            "messages": messages,
            "inferenceConfig": {"maxTokens": mt, "temperature": temp},
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        loop = self._get_loop()

        def _stream_sync():
            try:
                return self._client.converse_stream(**kwargs)
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("ThrottlingException", "TooManyRequestsException"):
                    raise BedrockThrottlingError(
                        f"Bedrock throttled: {e}", model_id=mid
                    ) from e
                raise BedrockError(f"Bedrock stream error: {e}", model_id=mid) from e

        async with _bedrock_circuit_breaker:
            response = await loop.run_in_executor(None, _stream_sync)

        stream = response.get("stream")
        if stream is None:
            # Fallback to non-streaming
            text = await self.invoke(prompt, model_id=mid, system_prompt=system_prompt)
            yield text
            return

        for event in stream:
            chunk = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
            if chunk:
                yield chunk


# Module-level singleton — import and reuse this throughout the application.
bedrock_client = BedrockClient()


# ── Standalone smoke test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _smoke() -> None:
        client = BedrockClient()
        print("Invoking Bedrock with a simple prompt...")
        result = await client.invoke(
            "Reply with exactly one sentence: What is Amazon Bedrock?"
        )
        print(f"Response: {result}")

    asyncio.run(_smoke())
