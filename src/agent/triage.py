"""Triage: merge one file's static-analysis Findings and LLM
StyleViolations into a single ranked, explainable list.

Two rules, both keyed on (file, line) collisions:

- Dedup: a static Finding and a StyleViolation at the same location are
  the same underlying issue seen by two tools. Keep the static Finding
  (deterministic, higher confidence); drop the style violation.
- Suppression: if the colliding static Finding is an ERROR, the style
  violation isn't just a duplicate, it's noise next to a real bug --
  dropped for that reason instead, so the trace says why.
"""

from __future__ import annotations

from src.agent.models import FindingSource, TriageAction, TriagedFinding
from src.agent.tracer import DecisionTracer
from src.tools.models import Finding, Severity, StyleViolation

_SEVERITY_RANK = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


def triage_findings(
    static_findings: list[Finding],
    style_violations: list[StyleViolation],
    *,
    tracer: DecisionTracer,
) -> list[TriagedFinding]:
    """Combine one file's findings into a single triaged list: kept
    items first (most severe first), then dropped items with the reason
    they were dropped. Every decision is also written to ``tracer``.
    """
    by_location = _sharpest_finding_per_location(static_findings)

    triaged: list[TriagedFinding] = []

    for finding in by_location.values():
        record = _keep_static(finding)
        triaged.append(record)
        tracer.record(stage="triage", file=record.file, line=record.line, action=record.action.value, reason=record.reason)

    for violation in style_violations:
        colliding = by_location.get((violation.file, violation.line))
        record = _triage_style_violation(violation, colliding)
        triaged.append(record)
        tracer.record(stage="triage", file=record.file, line=record.line, action=record.action.value, reason=record.reason)

    triaged.sort(key=lambda t: (t.action != TriageAction.KEPT, _SEVERITY_RANK[t.severity], t.file, t.line))
    return triaged


def _sharpest_finding_per_location(findings: list[Finding]) -> dict[tuple[str, int], Finding]:
    """If static analysis itself produced two findings on the same
    line, keep the more severe one so dedup has one canonical finding
    per location to compare style violations against.
    """
    by_location: dict[tuple[str, int], Finding] = {}
    for finding in findings:
        key = (finding.file, finding.line)
        existing = by_location.get(key)
        if existing is None or _SEVERITY_RANK[finding.severity] < _SEVERITY_RANK[existing.severity]:
            by_location[key] = finding
    return by_location


def _keep_static(finding: Finding) -> TriagedFinding:
    return TriagedFinding(
        file=finding.file,
        line=finding.line,
        severity=finding.severity,
        source=FindingSource.STATIC_ANALYSIS,
        rule_id=finding.rule_id,
        message=finding.message,
        action=TriageAction.KEPT,
        reason="static analysis finding",
    )


def _triage_style_violation(violation: StyleViolation, colliding: Finding | None) -> TriagedFinding:
    if colliding is None:
        return TriagedFinding(
            file=violation.file,
            line=violation.line,
            severity=Severity.INFO,
            source=FindingSource.STYLE_GUIDE,
            message=violation.violation,
            suggested_fix=violation.suggested_fix,
            action=TriageAction.KEPT,
            reason="style guide violation, no static finding at this location",
        )

    if colliding.severity == Severity.ERROR:
        action = TriageAction.DROPPED_SUPPRESSED
        reason = f"suppressed: real bug already flagged on this line ({colliding.rule_id})"
    else:
        action = TriageAction.DROPPED_DUPLICATE
        reason = f"duplicate: static analysis already flags this location ({colliding.rule_id})"

    return TriagedFinding(
        file=violation.file,
        line=violation.line,
        severity=Severity.INFO,
        source=FindingSource.STYLE_GUIDE,
        message=violation.violation,
        suggested_fix=violation.suggested_fix,
        action=action,
        reason=reason,
    )
