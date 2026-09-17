"""Style checker tool: given a diff and a style guide, asks an LLM to
return structured style violations.

Structured output is enforced by forcing a tool call (``tool_choice``)
against a fixed JSON schema, rather than asking the model to emit JSON in
prose and hoping it parses.
"""

from __future__ import annotations

import os

import anthropic
from pydantic import ValidationError

from src.tools.models import ErrorCode, Result, StyleViolation, ToolError

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 4096

_REPORT_TOOL_NAME = "report_style_violations"
_REPORT_TOOL = {
    "name": _REPORT_TOOL_NAME,
    "description": "Report every style-guide violation found in the diff.",
    "input_schema": {
        "type": "object",
        "properties": {
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "violation": {"type": "string"},
                        "suggested_fix": {"type": "string"},
                    },
                    "required": ["file", "line", "violation", "suggested_fix"],
                },
            }
        },
        "required": ["violations"],
    },
}


def check_style(
    diff: str,
    style_guide: str,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Result[list[StyleViolation]]:
    """Ask an LLM to find style-guide violations in ``diff``.

    ``client`` is injected so tests can substitute a fake Anthropic
    client without hitting the network or an API key.
    """
    active_client = client or _default_client(timeout)
    if isinstance(active_client, ToolError):
        return Result[list[StyleViolation]].fail(active_client)

    try:
        response = active_client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            tools=[_REPORT_TOOL],
            tool_choice={"type": "tool", "name": _REPORT_TOOL_NAME},
            messages=[{"role": "user", "content": _prompt(diff, style_guide)}],
        )
    except anthropic.APITimeoutError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.TIMEOUT, message=str(exc), retriable=True)
        )
    except anthropic.RateLimitError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.RATE_LIMITED, message=str(exc), retriable=True)
        )
    except anthropic.AuthenticationError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.AUTH_FAILED, message=str(exc), retriable=False)
        )
    except anthropic.APIConnectionError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.UPSTREAM_ERROR, message=str(exc), retriable=True)
        )
    except anthropic.APIError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.UPSTREAM_ERROR, message=str(exc), retriable=True)
        )

    return _parse_response(response)


def _default_client(timeout: float) -> anthropic.Anthropic | ToolError:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ToolError(code=ErrorCode.AUTH_FAILED, message="ANTHROPIC_API_KEY is not set", retriable=False)
    return anthropic.Anthropic(api_key=api_key, timeout=timeout)


def _prompt(diff: str, style_guide: str) -> str:
    return (
        "You are reviewing a pull request diff against a style guide. "
        f"Call {_REPORT_TOOL_NAME} with every violation you find. "
        "If there are none, call it with an empty violations list.\n\n"
        f"## Style guide\n{style_guide}\n\n## Diff\n{diff}"
    )


def _parse_response(response: anthropic.types.Message) -> Result[list[StyleViolation]]:
    for block in response.content:
        if block.type == "tool_use" and block.name == _REPORT_TOOL_NAME:
            try:
                violations = [StyleViolation(**item) for item in block.input["violations"]]
            except (KeyError, TypeError, ValidationError) as exc:
                return Result[list[StyleViolation]].fail(
                    ToolError(code=ErrorCode.PARSE_ERROR, message=f"malformed tool_use input: {exc}")
                )
            return Result[list[StyleViolation]].ok(violations)

    return Result[list[StyleViolation]].fail(
        ToolError(code=ErrorCode.PARSE_ERROR, message=f"model did not call {_REPORT_TOOL_NAME}")
    )
