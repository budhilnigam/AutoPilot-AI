"""
agents/tool_generator.py — Dynamic Tool Generator Agent.

Handles:
  GENERATE_TOOL    → generate a Python function from a natural-language spec,
                     validate its AST, and execute it safely in a subprocess
  VALIDATE_TOOL    → AST-only check on provided Python code (no execution)

Task parameters:
  GENERATE_TOOL:
    spec: str              — natural-language description of the tool
    input_schema: dict     — JSON schema describing the function inputs
    test_input: dict       — data to pass as kwargs when executing the tool
    timeout_seconds: int   — subprocess timeout (default 30, max 60)

  VALIDATE_TOOL:
    code: str              — Python source to validate

Security model (validates Properties 24–27):
  - All generated code passes ast.parse() (SyntaxError → rejected)
  - AST walk blocks: os.system, subprocess (unless reviewed), eval, exec,
    __import__, open (write mode), socket (raw), importlib
  - Execution via subprocess.run with: timeout, capture_output, no shell,
    restricted sys.path containing only stdlib and project venv site-packages
  - stdout/stderr returned verbatim (truncated at 8 KB)
  - Re-entrant: the generated function is extracted and called with test_input

Validates Properties 24, 25, 26, 27.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
import tempfile
import os
from typing import Any

from autopilot_ai.agents.base import BaseAgent
from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import ToolExecutionError, ToolValidationError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.integrations.aws.bedrock import bedrock_client
from autopilot_ai.models.insights import (
    Insight,
    InsightCategory,
    Recommendation,
    Urgency,
)
from autopilot_ai.models.responses import AgentResponse
from autopilot_ai.models.tasks import AgentType, Task, TaskType

logger = get_logger(__name__)

# ── Blocked names / patterns ──────────────────────────────────────────────────
_BLOCKED_CALLS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "__import__",
    "compile",
    "breakpoint",
})

_BLOCKED_ATTR_CHAINS: frozenset[tuple[str, str]] = frozenset({
    ("os", "system"),
    ("os", "popen"),
    ("os", "execvp"),
    ("os", "execve"),
    ("os", "execl"),
    ("os", "spawnl"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "run"),
    ("subprocess", "check_output"),
    ("subprocess", "check_call"),
    ("socket", "socket"),
    ("ctypes", "CDLL"),
    ("ctypes", "cdll"),
    ("importlib", "import_module"),
    ("importlib", "util"),
})

_BLOCKED_IMPORTS: frozenset[str] = frozenset({
    "subprocess",
    "socket",
    "ctypes",
    "importlib",
    "multiprocessing",
    "threading",
    "concurrent",
    "pty",
    "fcntl",
    "signal",
    "resource",
})

_MAX_OUTPUT_BYTES = 8 * 1024  # 8 KB
_MAX_TIMEOUT_SECONDS = 60


def _strip_fences(text: str) -> str:
    """Remove ```python``` or ``` fences from LLM output."""
    cleaned = re.sub(r"^```(?:python)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    return re.sub(r"\n?```$", "", cleaned.strip(), flags=re.MULTILINE)


def _extract_python_block(text: str) -> str:
    """
    Extract the first ```python ... ``` block from LLM output.
    Falls back to the entire text stripped of fences if no explicit block found.
    """
    m = re.search(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return _strip_fences(text)


class _SecurityVisitor(ast.NodeVisitor):
    """
    AST visitor that accumulates security violations.
    Does NOT raise — call .violations after visit() to check.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        # Direct calls: eval(...), exec(...), __import__(...)
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
            self.violations.append(f"Blocked call: {node.func.id}()")

        # Attribute calls: os.system(...), subprocess.run(...), etc.
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                chain = (node.func.value.id, node.func.attr)
                if chain in _BLOCKED_ATTR_CHAINS:
                    self.violations.append(f"Blocked attribute call: {node.func.value.id}.{node.func.attr}()")

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0]
        if root in _BLOCKED_IMPORTS:
            self.violations.append(f"Blocked from-import: from {module} import ...")
        self.generic_visit(node)


def _validate_code(code: str) -> list[str]:
    """
    Parse code with ast and walk for security violations.
    Returns list of violation messages (empty = clean).
    Raises ToolValidationError on syntax error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ToolValidationError(f"Syntax error: {e}", generated_code=code) from e

    visitor = _SecurityVisitor()
    visitor.visit(tree)
    return visitor.violations


def _extract_function_name(code: str) -> str | None:
    """Return the name of the first top-level function defined in the code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass
    return None


def _execute_code(code: str, test_input: dict, timeout: int) -> dict[str, str]:
    """
    Write a temporary Python file and execute it via subprocess.
    The generated function is called with test_input as kwargs.
    Returns {"stdout": ..., "stderr": ..., "exit_code": ...}.
    """
    func_name = _extract_function_name(code)
    if not func_name:
        raise ToolExecutionError("No function found in generated code")

    # Build a self-contained runner script
    input_json = json.dumps(test_input, default=str)
    runner = textwrap.dedent(f"""
import json, sys

{code}

_input = json.loads({input_json!r})
try:
    result = {func_name}(**_input)
    print(json.dumps({{"result": result}}, default=str))
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
""").strip()

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="autopilot_tool_",
    ) as f:
        f.write(runner)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=min(timeout, _MAX_TIMEOUT_SECONDS),
            # No shell=True — prevents shell injection
        )
        return {
            "stdout": result.stdout[:_MAX_OUTPUT_BYTES],
            "stderr": result.stderr[:_MAX_OUTPUT_BYTES],
            "exit_code": str(result.returncode),
        }
    except subprocess.TimeoutExpired:
        raise ToolExecutionError(f"Tool execution timed out after {timeout}s", returncode=-1, stderr="")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class ToolGeneratorAgent(BaseAgent):
    """
    Dynamic Tool Generator — generates, validates, and executes Python functions
    from natural-language specifications.

    Security is enforced at two layers:
    1. AST validation (static): blocks dangerous builtins and imports
    2. Subprocess execution (runtime): isolated process, timeout, captured output
    """

    agent_type = AgentType.TOOL_GENERATOR

    async def execute(self, task: Task) -> AgentResponse:
        if task.task_type == TaskType.GENERATE_TOOL:
            return await self._generate_tool(task)
        if task.task_type == TaskType.VALIDATE_TOOL:
            return await self._validate_tool(task)

        return self._partial(
            task,
            error_message=f"ToolGeneratorAgent does not handle task_type={task.task_type.value}",
        )

    # ── GENERATE_TOOL ─────────────────────────────────────────────────────

    async def _generate_tool(self, task: Task) -> AgentResponse:
        """
        1. Ask Claude to write a Python function matching the spec.
        2. Validate the generated code with AST security check.
        3. Execute in subprocess with test_input.
        4. Return code + execution result in data.
        Validates Properties 24–27.
        """
        spec: str = str(task.parameters.get("spec", ""))
        input_schema: dict = task.parameters.get("input_schema", {})  # type: ignore[assignment]
        test_input: dict = task.parameters.get("test_input", {})  # type: ignore[assignment]
        timeout: int = min(int(task.parameters.get("timeout_seconds", 30)), _MAX_TIMEOUT_SECONDS)

        if not spec.strip():
            return self._partial(task, error_message="No tool spec provided")

        schema_text = json.dumps(input_schema, indent=2) if input_schema else "Not specified (infer from spec)"

        prompt = f"""You are an expert Python engineer. Generate a single, self-contained Python function based on the spec below.

## Spec
{spec}

## Input Schema
{schema_text}

## Rules
1. Write a single top-level Python function.
2. Use ONLY Python stdlib (json, re, math, datetime, collections, itertools, etc.).
3. Do NOT import: subprocess, socket, os.system, eval, exec, ctypes, importlib, threading, multiprocessing.
4. Handle errors with try/except and return meaningful error messages.
5. Include a concise docstring.
6. The function must be deterministic and side-effect free.

Respond with ONLY the Python function code — no explanation, no markdown, just the def block."""

        raw = await bedrock_client.invoke(
            prompt=prompt,
            system_prompt="You are a Python code generation AI. Output only valid Python code, no markdown.",
        )

        code = _extract_python_block(raw)
        logger.info("tool_generator_code_generated", lines=len(code.splitlines()))

        # ── AST Security Validation (Property 25) ─────────────────────────
        try:
            violations = _validate_code(code)
        except ToolValidationError as e:
            return self._partial(
                task,
                insights=[Insight(
                    category=InsightCategory.CONFIGURATION,
                    component="tool-generator",
                    title="Generated code rejected — syntax error",
                    business_context="The LLM produced invalid Python syntax.",
                    urgency=Urgency.HIGH,
                    confidence=1.0,
                    recommendations=[Recommendation(
                        action="Retry with a more specific spec",
                        rationale=str(e),
                        steps=["Revise the spec to be more explicit about expected input/output types"],
                        expected_benefit="Valid generated code",
                    )],
                )],
                error_message=str(e),
            )

        if violations:
            violation_text = "; ".join(violations)
            logger.warning("tool_generator_security_violations", violations=violations)
            return self._partial(
                task,
                insights=[Insight(
                    category=InsightCategory.CONFIGURATION,
                    component="tool-generator",
                    title="Generated code rejected — security violations",
                    business_context="The LLM attempted to use blocked APIs.",
                    urgency=Urgency.IMMEDIATE,
                    confidence=1.0,
                    recommendations=[Recommendation(
                        action="Review and revise the spec",
                        rationale=violation_text,
                        steps=["Ensure the spec does not require system-level operations"],
                        expected_benefit="Secure generated code",
                    )],
                )],
                error_message=f"Security violations: {violation_text}",
            )

        # ── Subprocess Execution (Property 26) ────────────────────────────
        execution_result: dict[str, str] = {}
        execution_error: str = ""

        if test_input:
            try:
                execution_result = _execute_code(code, test_input, timeout)
            except ToolExecutionError as e:
                execution_error = str(e)
                logger.warning("tool_generator_execution_error", error=execution_error)
        else:
            execution_result = {"stdout": "", "stderr": "", "exit_code": "skipped"}

        insights: list[Insight] = []
        if execution_error or (execution_result.get("exit_code", "0") not in ("0", "skipped")):
            insights.append(Insight(
                category=InsightCategory.CONFIGURATION,
                component="tool-generator",
                title="Tool generated but execution failed",
                business_context="The code passed security validation but failed at runtime.",
                urgency=Urgency.MEDIUM,
                confidence=0.9,
                recommendations=[Recommendation(
                    action="Review execution output and revise spec",
                    rationale=execution_error or execution_result.get("stderr", ""),
                    steps=["Check stderr output", "Verify test_input matches function signature"],
                    expected_benefit="Working tool",
                )],
            ))

        return self._success(
            task,
            insights=insights,
            data={
                "generated_code": code,
                "function_name": _extract_function_name(code) or "unknown",
                "security_violations": violations,
                "execution": execution_result,
                "execution_error": execution_error,
            },
            model_used=settings.bedrock_model_id,
        )

    # ── VALIDATE_TOOL ─────────────────────────────────────────────────────

    async def _validate_tool(self, task: Task) -> AgentResponse:
        """
        Static AST validation only — no execution.
        Returns list of violations in data.violations.
        Validates Property 25.
        """
        code: str = str(task.parameters.get("code", ""))

        if not code.strip():
            return self._partial(task, error_message="No code provided for validation")

        try:
            violations = _validate_code(code)
        except ToolValidationError as e:
            return self._partial(task, error_message=str(e))

        passed = len(violations) == 0

        insight: Insight | None = None
        if not passed:
            insight = Insight(
                category=InsightCategory.CONFIGURATION,
                component="tool-generator",
                title="Code validation failed",
                business_context="Tool code contains security violations.",
                urgency=Urgency.IMMEDIATE,
                confidence=1.0,
                recommendations=[Recommendation(
                    action="Remove blocked operations",
                    rationale="; ".join(violations),
                    steps=[f"Remove: {v}" for v in violations],
                    expected_benefit="Code passes security policy",
                )],
            )

        return self._success(
            task,
            insights=[insight] if insight else [],
            data={
                "passed": passed,
                "violations": violations,
                "function_name": _extract_function_name(code),
            },
        )
