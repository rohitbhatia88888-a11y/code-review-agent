"""Style checker tool: given a diff and a style guide, asks an LLM to
return structured style violations.

Calls the model through OpenRouter's OpenAI-compatible API, so any
model OpenRouter proxies (not just Claude) can be swapped in via the
``model`` parameter. Structured output is enforced by forcing a tool
call (``tool_choice``) against a fixed JSON schema, rather than asking
the model to emit JSON in prose and hoping it parses.
"""

from __future__ import annotations

import json
import os

import openai
from pydantic import ValidationError

from src.tools.models import ErrorCode, Result, StyleViolation, ToolError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Must match a model id OpenRouter actually lists (check
# https://openrouter.ai/models or GET /api/v1/models) -- catalog naming
# can lag or vary. Override via the `model` parameter regardless.
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 4096

_REPORT_TOOL_NAME = "report_style_violations"
_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": _REPORT_TOOL_NAME,
        "description": "Report every style-guide violation found in the diff.",
        "parameters": {
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
    },
}


def check_style(
    diff: str,
    style_guide: str,
    *,
    client: openai.OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Result[list[StyleViolation]]:
    """Ask an LLM to find style-guide violations in ``diff``.

    ``client`` is injected so tests can substitute a fake OpenAI-shaped
    client without hitting the network or an API key.
    """
    active_client = client or _default_client(timeout)
    if isinstance(active_client, ToolError):
        return Result[list[StyleViolation]].fail(active_client)

    try:
        response = active_client.chat.completions.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            tools=[_REPORT_TOOL],
            tool_choice={"type": "function", "function": {"name": _REPORT_TOOL_NAME}},
            messages=[{"role": "user", "content": _prompt(diff, style_guide)}],
            # OpenRouter extension: returns real per-call cost in the
            # response's usage object instead of us estimating from a
            # pricing table (see src/eval/runner.py).
            extra_body={"usage": {"include": True}},
        )
    except openai.APITimeoutError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.TIMEOUT, message=str(exc), retriable=True)
        )
    except openai.RateLimitError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.RATE_LIMITED, message=str(exc), retriable=True)
        )
    except openai.AuthenticationError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.AUTH_FAILED, message=str(exc), retriable=False)
        )
    except openai.APIConnectionError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.UPSTREAM_ERROR, message=str(exc), retriable=True)
        )
    except openai.APIStatusError as exc:
        retriable = exc.status_code >= 500 or exc.status_code == 429
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.UPSTREAM_ERROR, message=str(exc), retriable=retriable)
        )
    except openai.APIError as exc:
        return Result[list[StyleViolation]].fail(
            ToolError(code=ErrorCode.UPSTREAM_ERROR, message=str(exc), retriable=True)
        )

    return _parse_response(response)


def _default_client(timeout: float) -> openai.OpenAI | ToolError:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ToolError(code=ErrorCode.AUTH_FAILED, message="OPENROUTER_API_KEY is not set", retriable=False)
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=timeout)


def _prompt(diff: str, style_guide: str) -> str:
    return (
        "You are reviewing a pull request diff against a style guide. "
        f"Call {_REPORT_TOOL_NAME} with every violation you find. "
        "If there are none, call it with an empty violations list.\n\n"
        f"## Style guide\n{style_guide}\n\n## Diff\n{diff}"
    )


def _parse_response(response) -> Result[list[StyleViolation]]:
    message = response.choices[0].message
    for call in message.tool_calls or []:
        if call.function.name != _REPORT_TOOL_NAME:
            continue
        try:
            arguments = json.loads(call.function.arguments)
            violations = [StyleViolation(**item) for item in arguments["violations"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
            return Result[list[StyleViolation]].fail(
                ToolError(code=ErrorCode.PARSE_ERROR, message=f"malformed tool call arguments: {exc}")
            )
        return Result[list[StyleViolation]].ok(violations)

    return Result[list[StyleViolation]].fail(
        ToolError(code=ErrorCode.PARSE_ERROR, message=f"model did not call {_REPORT_TOOL_NAME}")
    )
