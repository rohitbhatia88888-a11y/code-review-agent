"""Shared data contracts for every tool in ``src/tools``.

Every tool function returns a ``Result[T]``: success plus a typed payload,
or failure plus a typed ``ToolError``. Tools never raise for *expected*
failure modes (rate limits, timeouts, malformed upstream responses) --
only for programmer errors (bad arguments). Callers branch on
``result.success`` and never need a try/except around a tool call.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

T = TypeVar("T")


class ErrorCode(str, Enum):
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    AUTH_FAILED = "auth_failed"
    INVALID_INPUT = "invalid_input"
    UPSTREAM_ERROR = "upstream_error"
    PARSE_ERROR = "parse_error"


class ToolError(BaseModel):
    code: ErrorCode
    message: str
    retriable: bool = False
    retry_after_seconds: float | None = None


class Result(BaseModel, Generic[T]):
    """The single return shape for every tool function.

    Exactly one of ``payload`` / ``error`` is set; construct via
    ``Result.ok(...)`` or ``Result.fail(...)`` rather than the
    constructor directly.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    payload: T | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _check_payload_error_exclusive(self) -> Result[T]:
        if self.success and self.error is not None:
            raise ValueError("a successful Result cannot carry an error")
        if not self.success and self.payload is not None:
            raise ValueError("a failed Result cannot carry a payload")
        if not self.success and self.error is None:
            raise ValueError("a failed Result must carry an error")
        return self

    @classmethod
    def ok(cls, payload: T) -> Result[T]:
        return cls(success=True, payload=payload, error=None)

    @classmethod
    def fail(cls, error: ToolError) -> Result[T]:
        return cls(success=False, payload=None, error=error)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Finding(BaseModel):
    """One static-analysis result, normalized across analyzers."""

    file: str
    line: int
    rule_id: str
    severity: Severity
    message: str


class StyleViolation(BaseModel):
    """One style-guide violation, as judged by the style-checker tool."""

    file: str
    line: int
    violation: str
    suggested_fix: str


class PullRequestDiff(BaseModel):
    owner: str
    repo: str
    pr_number: int
    diff: str


class ChangedFile(BaseModel):
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None = None


class PostedComment(BaseModel):
    comment_id: int
    url: str
