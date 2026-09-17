"""The agent's decision loop for one PR.

Fetch changed files, run static analysis on every Python file (cheap,
local, so it runs before any budgeting decision), rank files by risk and
cap how much diff actually gets sent to the LLM style checker, triage the
combined findings per file, then post the results back to GitHub.

Every external call (GitHub, static analyzer, style checker) can fail.
Failures are handled explicitly and locally -- this function itself
never raises for a tool failure, it always returns a ``Result``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.agent.models import (
    CommentPostOutcome,
    FileReviewStatus,
    FileReviewSummary,
    PRReviewOutcome,
    TriagedFinding,
)
from src.agent.retry import (
    DEFAULT_BASE_DELAY_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    call_with_retry,
)
from src.agent.tracer import DecisionTracer
from src.agent.triage import triage_findings
from src.tools.github_client import GitHubClient
from src.tools.models import ChangedFile, Finding, Result, StyleViolation
from src.tools.static_analyzer import run_ruff
from src.tools.style_checker import check_style

DEFAULT_MAX_DIFF_LINES = 1000

StaticAnalyzer = Callable[..., Result[list[Finding]]]
StyleCheck = Callable[[str, str], Result[list[StyleViolation]]]


def review_pull_request(
    owner: str,
    repo: str,
    pr_number: int,
    style_guide: str,
    *,
    github: GitHubClient,
    repo_root: str,
    commit_sha: str,
    tracer: DecisionTracer,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    run_static_analysis: StaticAnalyzer = run_ruff,
    run_style_check: StyleCheck = check_style,
    sleep: Callable[[float], None] = time.sleep,
) -> Result[PRReviewOutcome]:
    files_result = github.fetch_changed_files(owner, repo, pr_number)
    if not files_result.success:
        tracer.record(
            stage="intake",
            action="aborted",
            reason=f"couldn't fetch changed files: {files_result.error.message}",
        )
        return Result[PRReviewOutcome].fail(files_result.error)

    reviewable, skipped = _partition_reviewable(files_result.payload)
    for f in skipped:
        tracer.record(
            stage="intake",
            file=f.filename,
            action="skipped",
            reason="GitHub returned no patch for this file (binary, removed, or too large to diff)",
        )

    static_findings, file_status = _run_static_analysis_pass(reviewable, repo_root, run_static_analysis, tracer)

    selected, truncated = _budget_by_diff_size(reviewable, static_findings, max_diff_lines)
    for f in truncated:
        file_status[f.filename] = FileReviewStatus.TRUNCATED
        tracer.record(
            stage="diff_chunking",
            file=f.filename,
            action="truncated",
            reason=(
                f"diff-size budget ({max_diff_lines} lines) exhausted before this file; "
                f"{f.additions + f.deletions} line(s) not style-checked, "
                "static-analysis findings (if any) still included"
            ),
        )

    style_violations = _run_style_check_pass(selected, style_guide, run_style_check, sleep, file_status, tracer)

    file_summaries: list[FileReviewSummary] = []
    all_kept: list[TriagedFinding] = []
    for f in reviewable:
        triaged = triage_findings(
            static_findings.get(f.filename, []),
            style_violations.get(f.filename, []),
            tracer=tracer,
        )
        file_summaries.append(FileReviewSummary(filename=f.filename, status=file_status[f.filename], findings=triaged))
        all_kept.extend(t for t in triaged if t.action.value == "kept")

    comment_outcome = _post_comments(github, owner, repo, pr_number, commit_sha, all_kept, truncated, sleep, tracer)

    outcome = PRReviewOutcome(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        files=file_summaries,
        truncated_files=[f.filename for f in truncated],
        comment_outcome=comment_outcome,
        trace_path=str(tracer.path),
    )
    return Result[PRReviewOutcome].ok(outcome)


def _partition_reviewable(files: list[ChangedFile]) -> tuple[list[ChangedFile], list[ChangedFile]]:
    reviewable = [f for f in files if f.patch]
    skipped = [f for f in files if not f.patch]
    return reviewable, skipped


def _run_static_analysis_pass(
    files: list[ChangedFile],
    repo_root: str,
    run_static_analysis: StaticAnalyzer,
    tracer: DecisionTracer,
) -> tuple[dict[str, list[Finding]], dict[str, FileReviewStatus]]:
    findings: dict[str, list[Finding]] = {}
    status: dict[str, FileReviewStatus] = {}

    for f in files:
        status[f.filename] = FileReviewStatus.FULL
        if not f.filename.endswith(".py"):
            findings[f.filename] = []
            continue

        result = run_static_analysis([f.filename], cwd=repo_root)
        if result.success:
            # The analyzer may report its own (e.g. absolute, resolved)
            # path for the file it just scanned -- we scoped this call to
            # exactly one file, so re-key every finding to the identity
            # GitHub gave us for it. Anything else would post an inline
            # comment against a local filesystem path GitHub doesn't
            # recognize as part of the PR's diff.
            findings[f.filename] = [finding.model_copy(update={"file": f.filename}) for finding in result.payload]
            tracer.record(
                stage="static_analysis", file=f.filename, action="completed", reason=f"{len(result.payload)} finding(s)"
            )
        else:
            # No retry: a crashing/timing-out analyzer means "unavailable
            # for this file right now", not a transient blip worth
            # spinning on. Degrade immediately and keep going.
            findings[f.filename] = []
            status[f.filename] = FileReviewStatus.PARTIAL_REVIEW
            tracer.record(
                stage="static_analysis",
                file=f.filename,
                action="failed",
                reason=(
                    f"static analyzer failed ({result.error.code.value}): {result.error.message}; "
                    "continuing with style-check-only for this file"
                ),
            )

    return findings, status


def _budget_by_diff_size(
    files: list[ChangedFile],
    static_findings: dict[str, list[Finding]],
    max_diff_lines: int,
) -> tuple[list[ChangedFile], list[ChangedFile]]:
    """Rank files by risk (most static-analysis findings, then largest
    diff) and keep a contiguous prefix of that ranking within
    ``max_diff_lines``. Always keeps at least the single highest-risk
    file, even if it alone exceeds the budget. Everything after the
    budget is exhausted is truncated, in ranked order.
    """

    def risk_key(f: ChangedFile) -> tuple[int, int]:
        return (len(static_findings.get(f.filename, [])), f.additions + f.deletions)

    ranked = sorted(files, key=risk_key, reverse=True)

    selected: list[ChangedFile] = []
    truncated: list[ChangedFile] = []
    total_lines = 0
    over_budget = False
    for f in ranked:
        size = f.additions + f.deletions
        if over_budget or (selected and total_lines + size > max_diff_lines):
            truncated.append(f)
            over_budget = True
            continue
        selected.append(f)
        total_lines += size

    selected_names = {f.filename for f in selected}
    selected_in_pr_order = [f for f in files if f.filename in selected_names]
    return selected_in_pr_order, truncated


def _run_style_check_pass(
    files: list[ChangedFile],
    style_guide: str,
    run_style_check: StyleCheck,
    sleep: Callable[[float], None],
    file_status: dict[str, FileReviewStatus],
    tracer: DecisionTracer,
) -> dict[str, list[StyleViolation]]:
    violations: dict[str, list[StyleViolation]] = {}

    for f in files:
        result = call_with_retry(
            lambda f=f: run_style_check(f.patch, style_guide),
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            base_delay=DEFAULT_BASE_DELAY_SECONDS,
            sleep=sleep,
        )
        if result.success:
            violations[f.filename] = result.payload
            tracer.record(
                stage="style_check", file=f.filename, action="completed", reason=f"{len(result.payload)} violation(s)"
            )
        else:
            violations[f.filename] = []
            if file_status.get(f.filename) == FileReviewStatus.FULL:
                file_status[f.filename] = FileReviewStatus.STYLE_CHECK_FAILED
            tracer.record(
                stage="style_check",
                file=f.filename,
                action="failed",
                reason=f"style checker failed after retries ({result.error.code.value}): {result.error.message}",
            )

    return violations


def _post_comments(
    github: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    commit_sha: str,
    findings: list[TriagedFinding],
    truncated: list[ChangedFile],
    sleep: Callable[[float], None],
    tracer: DecisionTracer,
) -> CommentPostOutcome:
    posted = 0
    failed: list[TriagedFinding] = []

    for finding in findings:
        result = call_with_retry(
            lambda finding=finding: github.post_inline_comment(
                owner,
                repo,
                pr_number,
                commit_sha=commit_sha,
                file=finding.file,
                line=finding.line,
                body=_format_inline_comment(finding),
            ),
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            base_delay=DEFAULT_BASE_DELAY_SECONDS,
            sleep=sleep,
        )
        if result.success:
            posted += 1
            tracer.record(stage="comment_posting", file=finding.file, line=finding.line, action="posted_inline", reason="posted")
        else:
            failed.append(finding)
            tracer.record(
                stage="comment_posting",
                file=finding.file,
                line=finding.line,
                action="failed_inline",
                reason=f"gave up after retries ({result.error.code.value}): {result.error.message}",
            )

    summary_body = _format_summary_comment(posted, failed, truncated)
    summary_result = call_with_retry(
        lambda: github.post_summary_comment(owner, repo, pr_number, body=summary_body),
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        base_delay=DEFAULT_BASE_DELAY_SECONDS,
        sleep=sleep,
    )
    if summary_result.success:
        tracer.record(
            stage="comment_posting",
            action="posted_summary",
            reason=f"{len(failed)} failed-inline fallback item(s), {len(truncated)} truncated file(s) noted",
        )
    else:
        tracer.record(
            stage="comment_posting",
            action="summary_failed",
            reason=f"summary comment also failed after retries: {summary_result.error.message}",
        )

    return CommentPostOutcome(
        posted_inline=posted,
        failed_inline=failed,
        summary_comment_posted=summary_result.success,
        summary_comment_error=None if summary_result.success else summary_result.error.message,
    )


def _format_inline_comment(finding: TriagedFinding) -> str:
    header = f"**{finding.severity.value.upper()}**"
    if finding.rule_id:
        header += f" `{finding.rule_id}`"
    body = f"{header}: {finding.message}"
    if finding.suggested_fix:
        body += f"\n\nSuggested fix: {finding.suggested_fix}"
    return body


def _format_summary_comment(posted: int, failed: list[TriagedFinding], truncated: list[ChangedFile]) -> str:
    lines = [f"## Code Review Agent\n{posted} finding(s) posted inline."]

    if failed:
        lines.append(f"\n{len(failed)} finding(s) could not be posted inline after retries:")
        lines.extend(f"- `{f.file}:{f.line}` - {f.message}" for f in failed)

    if truncated:
        lines.append(f"\n{len(truncated)} file(s) were not fully reviewed (diff-size budget exceeded):")
        lines.extend(f"- `{f.filename}` ({f.additions + f.deletions} line(s) changed)" for f in truncated)

    return "\n".join(lines)
