"""Data contracts for the agent's triage and decision loop."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from src.tools.models import Severity


class TriageAction(str, Enum):
    KEPT = "kept"
    DROPPED_DUPLICATE = "dropped_duplicate"
    DROPPED_SUPPRESSED = "dropped_suppressed"


class FindingSource(str, Enum):
    STATIC_ANALYSIS = "static_analysis"
    STYLE_GUIDE = "style_guide"


class TriagedFinding(BaseModel):
    """One finding after triage: a static-analysis Finding or an LLM
    StyleViolation, normalized to a common shape and carrying the
    triage decision (kept / dropped) and why.
    """

    file: str
    line: int
    severity: Severity
    source: FindingSource
    rule_id: str | None = None
    message: str
    suggested_fix: str | None = None
    action: TriageAction
    reason: str


class FileReviewStatus(str, Enum):
    FULL = "full"
    PARTIAL_REVIEW = "partial_review"  # static analysis failed; style-check-only
    STYLE_CHECK_FAILED = "style_check_failed"  # static analysis ok, style check didn't
    TRUNCATED = "truncated"  # skipped entirely: diff-size budget exhausted


class FileReviewSummary(BaseModel):
    filename: str
    status: FileReviewStatus
    findings: list[TriagedFinding] = Field(default_factory=list)


class CommentPostOutcome(BaseModel):
    posted_inline: int = 0
    failed_inline: list[TriagedFinding] = Field(default_factory=list)
    summary_comment_posted: bool = False
    summary_comment_error: str | None = None


class PRReviewOutcome(BaseModel):
    owner: str
    repo: str
    pr_number: int
    files: list[FileReviewSummary]
    truncated_files: list[str] = Field(default_factory=list)
    comment_outcome: CommentPostOutcome
    trace_path: str


class TraceEntry(BaseModel):
    timestamp: datetime
    stage: str
    action: str
    reason: str
    file: str | None = None
    line: int | None = None
