"""Golden-set schema.

A golden set is a frozen collection of synthetic PRs with human-reviewed
ground truth. It exists to measure the agent, not to demonstrate it --
every case says exactly what the agent is and isn't required to catch,
and why, so a human can audit disagreements line by line.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class CaseCategory(str, Enum):
    REAL_BUG = "real_bug"  # a genuine, injected defect the agent should catch
    CLEAN = "clean"  # ordinary correct code; nothing should be flagged
    STYLE_ONLY = "style_only"  # no bug, but a real style-guide violation
    FALSE_POSITIVE_TRAP = "false_positive_trap"  # looks wrong, isn't; nothing should be flagged


class BugType(str, Enum):
    OFF_BY_ONE = "off_by_one"
    NULL_CHECK_REMOVED = "null_check_removed"
    SQL_INJECTION = "sql_injection"
    RESOURCE_LEAK = "resource_leak"
    INCORRECT_COMPARISON = "incorrect_comparison"
    MUTABLE_DEFAULT_ARG = "mutable_default_arg"
    BROAD_EXCEPT_SWALLOWED = "broad_except_swallowed"
    HARDCODED_SECRET = "hardcoded_secret"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    RACE_CONDITION = "race_condition"


class ExpectedFinding(BaseModel):
    """One thing a case's ground truth says about a location in the diff.

    ``required=True`` means recall is measured against it: the agent
    must flag this location (any tool, any message) or it counts as a
    false negative. ``required=False`` documents an acceptable-but-
    optional finding (e.g. an incidental style nit next to the real
    bug) -- flagging it is fine, but missing it isn't penalized, and
    it still absorbs an agent finding at that location so that finding
    isn't miscounted as a false positive.
    """

    file: str
    line: int
    line_end: int | None = None  # inclusive; defaults to `line` if unset
    category: str  # "bug" | "security" | "style" -- free text, not load-bearing
    severity: str  # "error" | "warning" | "info"
    required: bool
    description: str  # why this is (or isn't) an issue -- for the human reviewer

    @model_validator(mode="after")
    def _line_end_defaults_to_line(self) -> ExpectedFinding:
        if self.line_end is None:
            self.line_end = self.line
        if self.line_end < self.line:
            raise ValueError("line_end must be >= line")
        return self


class PRFile(BaseModel):
    filename: str
    content_before: str | None = None  # None for a newly added file
    content_after: str


class GoldenCase(BaseModel):
    case_id: str
    category: CaseCategory
    bug_type: BugType | None = None
    description: str  # scenario summary for humans
    files: list[PRFile]
    expected_findings: list[ExpectedFinding] = Field(default_factory=list)
    reviewed: bool = False  # human sign-off gate; every case must be true to freeze
    reviewer_notes: str = ""

    @model_validator(mode="after")
    def _bug_type_matches_category(self) -> GoldenCase:
        if self.category == CaseCategory.REAL_BUG and self.bug_type is None:
            raise ValueError("real_bug cases must set bug_type")
        if self.category != CaseCategory.REAL_BUG and self.bug_type is not None:
            raise ValueError("bug_type is only meaningful on real_bug cases")
        return self

    @model_validator(mode="after")
    def _no_should_be_silent_requires(self) -> GoldenCase:
        if self.category in (CaseCategory.CLEAN, CaseCategory.FALSE_POSITIVE_TRAP) and self.expected_findings:
            raise ValueError(f"{self.category.value} cases must have no expected findings (nothing should be flagged)")
        return self


class GoldenSet(BaseModel):
    version: str = "1.0"
    style_guide: str  # the style-guide text every case is checked against
    cases: list[GoldenCase]

    @model_validator(mode="after")
    def _case_ids_unique(self) -> GoldenSet:
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate case_id(s): {duplicates}")
        return self
