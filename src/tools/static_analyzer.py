"""Static analysis tool: runs ruff over a set of files and normalizes its
output into ``Finding`` objects.
"""

from __future__ import annotations

import json
import subprocess

from src.tools.models import ErrorCode, Finding, Result, Severity, ToolError

DEFAULT_TIMEOUT_SECONDS = 30.0

# Fallback for ruff versions whose JSON output has no "severity" field:
# rule-code prefixes mapped by hand. Anything unlisted defaults to INFO.
_ERROR_PREFIXES = ("E9", "F")  # syntax errors, pyflakes (undefined names, unused imports, ...)
_WARNING_PREFIXES = ("E", "W", "B", "C90")  # pycodestyle, pep8-naming complexity, bugbear


def run_ruff(
    files: list[str],
    *,
    cwd: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Result[list[Finding]]:
    """Run ``ruff check`` on ``files`` and return normalized Findings.

    An empty ``files`` list is a valid input (a PR that touches no Python
    files) and returns an empty, successful result rather than invoking
    ruff at all.
    """
    if not files:
        return Result[list[Finding]].ok([])

    command = ["ruff", "check", "--output-format=json", "--exit-zero", *files]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Result[list[Finding]].fail(
            ToolError(code=ErrorCode.TIMEOUT, message=f"ruff timed out after {timeout}s", retriable=True)
        )
    except FileNotFoundError:
        return Result[list[Finding]].fail(
            ToolError(code=ErrorCode.UPSTREAM_ERROR, message="ruff executable not found on PATH", retriable=False)
        )

    # --exit-zero means a non-zero return code here is ruff itself failing
    # (bad invocation, internal error), never "lint findings exist".
    if completed.returncode != 0:
        return Result[list[Finding]].fail(
            ToolError(
                code=ErrorCode.UPSTREAM_ERROR,
                message=f"ruff exited {completed.returncode}: {completed.stderr.strip()[:300]}",
                retriable=False,
            )
        )

    try:
        raw_findings = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        return Result[list[Finding]].fail(
            ToolError(code=ErrorCode.PARSE_ERROR, message=f"couldn't parse ruff output: {exc}")
        )

    try:
        findings = [_to_finding(entry) for entry in raw_findings]
    except (KeyError, TypeError) as exc:
        return Result[list[Finding]].fail(
            ToolError(code=ErrorCode.PARSE_ERROR, message=f"malformed ruff finding: {exc}")
        )

    return Result[list[Finding]].ok(findings)


def _to_finding(entry: dict) -> Finding:
    rule_id = entry.get("code") or "unknown"
    return Finding(
        file=entry["filename"],
        line=entry["location"]["row"],
        rule_id=rule_id,
        severity=_severity(entry, rule_id),
        message=entry["message"],
    )


def _severity(entry: dict, rule_id: str) -> Severity:
    raw = entry.get("severity")
    if raw is not None:
        try:
            return Severity(raw)
        except ValueError:
            pass  # unrecognized value from a newer ruff release; fall through
    if rule_id.startswith(_ERROR_PREFIXES):
        return Severity.ERROR
    if rule_id.startswith(_WARNING_PREFIXES):
        return Severity.WARNING
    return Severity.INFO
