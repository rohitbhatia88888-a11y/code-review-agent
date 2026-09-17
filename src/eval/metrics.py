"""Scoring: match the agent's kept findings against a golden case's
ground truth, and roll per-case matches up into precision / recall /
false-positive-rate.

Matching is deliberately simple and auditable: an agent finding matches
an expected finding iff they're on the same file and the agent's line
falls within ``[expected.line, expected.line_end]``. No message-text
similarity is involved -- location is the one thing that's cheap to
verify by eye when a human is deciding whether a match is fair.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.models import TriagedFinding
from src.eval.models import CaseCategory, ExpectedFinding, GoldenCase


class CaseMatch(BaseModel):
    case_id: str
    category: CaseCategory
    matched_required: list[TriagedFinding] = Field(default_factory=list)
    matched_optional: list[TriagedFinding] = Field(default_factory=list)
    false_positives: list[TriagedFinding] = Field(default_factory=list)
    missed_required: list[ExpectedFinding] = Field(default_factory=list)
    missed_optional: list[ExpectedFinding] = Field(default_factory=list)

    @property
    def matched(self) -> list[TriagedFinding]:
        return self.matched_required + self.matched_optional

    @property
    def has_false_positive(self) -> bool:
        return len(self.false_positives) > 0

    @property
    def should_be_silent(self) -> bool:
        return self.category in (CaseCategory.CLEAN, CaseCategory.FALSE_POSITIVE_TRAP)


def score_case(case: GoldenCase, kept_findings: list[TriagedFinding]) -> CaseMatch:
    remaining_expected = list(case.expected_findings)
    matched_required: list[TriagedFinding] = []
    matched_optional: list[TriagedFinding] = []
    false_positives: list[TriagedFinding] = []

    for finding in kept_findings:
        match = next(
            (e for e in remaining_expected if e.file == finding.file and e.line <= finding.line <= e.line_end),
            None,
        )
        if match is None:
            false_positives.append(finding)
            continue
        remaining_expected.remove(match)
        (matched_required if match.required else matched_optional).append(finding)

    return CaseMatch(
        case_id=case.case_id,
        category=case.category,
        matched_required=matched_required,
        matched_optional=matched_optional,
        false_positives=false_positives,
        missed_required=[e for e in remaining_expected if e.required],
        missed_optional=[e for e in remaining_expected if not e.required],
    )


class CategoryMetrics(BaseModel):
    case_count: int
    recall: float | None = None  # None when the category has no required findings to measure
    precision: float | None = None  # None when the agent raised nothing in this category
    pr_false_positive_rate: float | None = None  # None when the category isn't a should-be-silent one


class AggregateMetrics(BaseModel):
    case_count: int

    total_required: int
    matched_required: int
    recall: float | None

    total_agent_findings: int
    true_positive_findings: int
    precision: float | None

    should_be_silent_cases: int
    should_be_silent_cases_with_fp: int
    pr_false_positive_rate: float | None

    missed_case_ids: list[str] = Field(default_factory=list)
    false_positive_case_ids: list[str] = Field(default_factory=list)

    by_category: dict[str, CategoryMetrics] = Field(default_factory=dict)


def aggregate(matches: list[CaseMatch]) -> AggregateMetrics:
    by_category: dict[str, CategoryMetrics] = {}
    for category in {m.category for m in matches}:
        by_category[category.value] = _category_metrics([m for m in matches if m.category == category])

    total_required = sum(len(m.matched_required) + len(m.missed_required) for m in matches)
    matched_required = sum(len(m.matched_required) for m in matches)
    total_agent_findings = sum(len(m.matched) + len(m.false_positives) for m in matches)
    true_positive_findings = sum(len(m.matched) for m in matches)
    should_be_silent = [m for m in matches if m.should_be_silent]
    should_be_silent_with_fp = [m for m in should_be_silent if m.has_false_positive]

    return AggregateMetrics(
        case_count=len(matches),
        total_required=total_required,
        matched_required=matched_required,
        recall=(matched_required / total_required) if total_required else None,
        total_agent_findings=total_agent_findings,
        true_positive_findings=true_positive_findings,
        precision=(true_positive_findings / total_agent_findings) if total_agent_findings else None,
        should_be_silent_cases=len(should_be_silent),
        should_be_silent_cases_with_fp=len(should_be_silent_with_fp),
        pr_false_positive_rate=(len(should_be_silent_with_fp) / len(should_be_silent)) if should_be_silent else None,
        missed_case_ids=sorted(m.case_id for m in matches if m.missed_required),
        false_positive_case_ids=sorted(m.case_id for m in matches if m.has_false_positive),
        by_category=by_category,
    )


def _category_metrics(matches: list[CaseMatch]) -> CategoryMetrics:
    total_required = sum(len(m.matched_required) + len(m.missed_required) for m in matches)
    matched_required = sum(len(m.matched_required) for m in matches)
    total_agent_findings = sum(len(m.matched) + len(m.false_positives) for m in matches)
    true_positive_findings = sum(len(m.matched) for m in matches)
    should_be_silent = [m for m in matches if m.should_be_silent]
    should_be_silent_with_fp = [m for m in should_be_silent if m.has_false_positive]

    return CategoryMetrics(
        case_count=len(matches),
        recall=(matched_required / total_required) if total_required else None,
        precision=(true_positive_findings / total_agent_findings) if total_agent_findings else None,
        pr_false_positive_rate=(len(should_be_silent_with_fp) / len(should_be_silent)) if should_be_silent else None,
    )
