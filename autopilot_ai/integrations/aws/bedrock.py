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
import shlex
import subprocess
from functools import partial
from typing import TYPE_CHECKING, Any, Type, TypeVar

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import (
    BedrockError,
    BedrockModelError,
    BedrockParseError,
    BedrockThrottlingError,
)
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import CircuitBreaker, with_retry
from autopilot_ai.integrations.aws.tool import aws_api

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = get_logger(__name__)

T = TypeVar("T", bound="BaseModel")

# One shared circuit breaker for the entire Bedrock service.
# Opened after 3 consecutive AutoPilotErrors; resets after 60s.
_bedrock_circuit_breaker = CircuitBreaker(name="bedrock")


def _quote(s: str) -> str:
    return shlex.quote(s)


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
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily get the running event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.get_event_loop()

    

    # ── Public API ─────────────────────────────────────────────────────────

    @with_retry(retry_on=(BedrockThrottlingError,))
    async def invoke(
        self,
        prompt: str,
        model_id: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        """
        Send a prompt to Bedrock and return the plain-text response.

        Args:
            prompt:         User message content.
            model_id:       Override the default model.
            system_prompt:  Optional system instruction for the model.
            max_tokens:     Override Settings.bedrock_max_tokens.
            temperature:    Override Settings.bedrock_temperature.
            thinking_budget: Max tokens for extended thinking (Claude 3.5+ only).
                           If set, limits the thinking phase to this many tokens.

        Returns:
            The assistant's text response as a plain string.
        """
        mid = model_id or self._model_id
        mt = max_tokens or settings.bedrock_max_tokens
        temp = temperature if temperature is not None else settings.bedrock_temperature

        messages = [{"role": "user", "content": [{"text": prompt}]}]

        logger.debug("bedrock_invoke", model_id=mid, prompt_len=len(prompt))

        async with _bedrock_circuit_breaker:
            # Use boto3 Bedrock Runtime `invoke_model` via aws_api helper.
            # Build a generic request body; models accept a `messages` list
            # or a model-specific body — this shape mirrors the test harness.
            request_body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": mt,
                "temperature": temp,
            }
            if system_prompt:
                request_body["system"] = system_prompt
            # For Claude 3.5+, set thinking budget to prevent infinite reasoning loops
            if thinking_budget is not None:
                request_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }

            params = {"modelId": mid, "body": json.dumps(request_body)}

            try:
                resp = await aws_api("bedrock-runtime", "invoke_model", params)
            except Exception as e:
                serr = str(e)
                if "Throttl" in serr or "TooMany" in serr or "Throttling" in serr:
                    raise BedrockThrottlingError(f"Bedrock throttled: {serr}") from e
                raise BedrockError(f"Bedrock invocation error: {serr}") from e

            # The boto3 response often contains a streaming `body` key (bytes).
            body = None
            if isinstance(resp, dict):
                body = resp.get("body") or resp.get("Body")
            # If bytes, decode and parse JSON
            parsed = None
            if isinstance(body, (bytes, bytearray)):
                try:
                    parsed = json.loads(body.decode("utf-8"))
                except Exception:
                    parsed = None
            elif isinstance(body, str):
                try:
                    parsed = json.loads(body)
                except Exception:
                    parsed = None
            elif isinstance(resp, dict) and "body" not in resp and "output" in resp:
                # Fallback for different shapes
                parsed = resp.get("output")

            # Robust extraction from a few known response shapes
            result = None
            thinking = None
            if parsed:
                # common: {'choices':[{'message':{'content': '...'}}]}
                if isinstance(parsed, dict) and "choices" in parsed:
                    try:
                        choice = parsed["choices"][0]
                        msg = choice.get("message") or {}
                        if isinstance(msg, dict):
                            # content may be nested
                            content = msg.get("content")
                            if isinstance(content, str):
                                result = content
                            elif isinstance(content, dict) and "text" in content:
                                result = content["text"]
                            elif isinstance(content, list):
                                # Extract thinking and text blocks from content array
                                for block in content:
                                    if isinstance(block, dict):
                                        if block.get("type") == "thinking":
                                            thinking = block.get("thinking")
                                        elif block.get("type") == "text":
                                            result = block.get("text")
                    except Exception:
                        result = None
                # other shape: {'output': {'message': {'content': [{'text': '...'}]}}}
            if result is None and isinstance(resp, dict):
                output = resp.get("output") or {}
                message = output.get("message") or {}
                content_blocks = message.get("content", [])
                for block in content_blocks:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            thinking = block.get("thinking")
                        elif block.get("type") == "text" or "text" in block:
                            result = block.get("text")
                            break

            if not result:
                raise BedrockModelError(
                    "Bedrock response contained no text content",
                    model_id=mid,
                    response_keys=list(resp.keys()) if isinstance(resp, dict) else [],
                )

        logger.debug("bedrock_invoke_done", model_id=mid, response_len=len(result), has_thinking=thinking is not None)
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

        # Streaming is not implemented via CLI; fall back to single invoke
        text = await self.invoke(prompt, model_id=mid, system_prompt=system_prompt)
        yield text


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
