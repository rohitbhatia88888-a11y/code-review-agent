"""Failure-mode simulations for the agent's decision loop.

Every test here injects a fake GitHub client plus fake static-analysis /
style-check callables that misbehave in a specific way, then asserts the
loop degrades gracefully (returns a successful Result describing the
degradation) rather than raising or hanging.
"""

from src.agent.decision_loop import review_pull_request
from src.agent.models import FileReviewStatus
from src.agent.tracer import DecisionTracer
from src.tools.models import (
    ChangedFile,
    ErrorCode,
    Finding,
    PostedComment,
    Result,
    Severity,
    StyleViolation,
    ToolError,
)


class _FakeGitHubClient:
    def __init__(self, changed_files_result, *, post_inline=None, post_summary=None):
        self._changed_files_result = changed_files_result
        self._post_inline = post_inline or (
            lambda **kw: Result[PostedComment].ok(PostedComment(comment_id=1, url="https://x"))
        )
        self._post_summary = post_summary or (
            lambda **kw: Result[PostedComment].ok(PostedComment(comment_id=2, url="https://x"))
        )
        self.inline_calls: list[tuple[str, int]] = []
        self.summary_calls: list[str] = []

    def fetch_changed_files(self, owner, repo, pr_number):
        return self._changed_files_result

    def post_inline_comment(self, owner, repo, pr_number, *, commit_sha, file, line, body):
        self.inline_calls.append((file, line))
        return self._post_inline(file=file, line=line, body=body)

    def post_summary_comment(self, owner, repo, pr_number, *, body):
        self.summary_calls.append(body)
        return self._post_summary(body=body)


def _tracer(tmp_path) -> DecisionTracer:
    return DecisionTracer(tmp_path / "trace.jsonl")


def test_fetch_changed_files_failure_returns_failed_result_not_exception(tmp_path):
    github = _FakeGitHubClient(Result[list[ChangedFile]].fail(ToolError(code=ErrorCode.NOT_FOUND, message="no such PR")))

    result = review_pull_request(
        "acme",
        "widgets",
        999,
        "style guide",
        github=github,
        repo_root=str(tmp_path),
        commit_sha="sha",
        tracer=_tracer(tmp_path),
        run_static_analysis=lambda *a, **k: Result[list[Finding]].ok([]),
        run_style_check=lambda *a, **k: Result[list[StyleViolation]].ok([]),
        sleep=lambda s: None,
    )

    assert not result.success
    assert result.error.code == ErrorCode.NOT_FOUND


def test_static_analyzer_crash_degrades_to_partial_review(tmp_path):
    """Failure mode 1: static analyzer times out/crashes -> log it,
    continue with style-check-only for that file, mark it 'partial
    review' in the output.
    """
    changed_files = [ChangedFile(filename="broken.py", status="modified", additions=5, deletions=1, patch="@@ -1 +1 @@\n+bad")]
    github = _FakeGitHubClient(Result[list[ChangedFile]].ok(changed_files))
    tracer = _tracer(tmp_path)

    def crashing_static_analyzer(files, **kwargs):
        return Result[list[Finding]].fail(ToolError(code=ErrorCode.TIMEOUT, message="ruff timed out", retriable=True))

    def style_check(diff, style_guide):
        return Result[list[StyleViolation]].ok(
            [StyleViolation(file="broken.py", line=1, violation="bad style", suggested_fix="fix it")]
        )

    result = review_pull_request(
        "acme",
        "widgets",
        1,
        "style guide",
        github=github,
        repo_root=str(tmp_path),
        commit_sha="sha",
        tracer=tracer,
        run_static_analysis=crashing_static_analyzer,
        run_style_check=style_check,
        sleep=lambda s: None,
    )

    assert result.success  # the loop itself never raises
    file_summary = result.payload.files[0]
    assert file_summary.status == FileReviewStatus.PARTIAL_REVIEW
    assert any(f.source.value == "style_guide" for f in file_summary.findings)
    assert any(e.stage == "static_analysis" and e.action == "failed" for e in tracer.entries)


def test_style_check_file_field_is_overridden_with_the_real_filename(tmp_path):
    """GitHub's PR 'patch' field never includes `--- a/...`/`+++ b/...`
    headers, so a real style-check response can't reliably self-report
    which file it's about (observed live: OpenRouter returned "unknown"
    for every violation's `file`, which then 422'd when posted inline
    against a nonexistent path). The caller already knows the answer
    with certainty, so it must win over whatever the model says.
    """
    changed_files = [ChangedFile(filename="app.py", status="modified", additions=1, deletions=0, patch="@@ -0,0 +1 @@\n+def calculateTotal(): pass\n")]
    github = _FakeGitHubClient(Result[list[ChangedFile]].ok(changed_files))
    tracer = _tracer(tmp_path)

    def static_ok(files, **kwargs):
        return Result[list[Finding]].ok([])

    def style_check_reporting_wrong_file(diff, style_guide):
        return Result[list[StyleViolation]].ok(
            [StyleViolation(file="unknown", line=1, violation="camelCase function name", suggested_fix="rename")]
        )

    result = review_pull_request(
        "acme",
        "widgets",
        1,
        "style guide",
        github=github,
        repo_root=str(tmp_path),
        commit_sha="sha",
        tracer=tracer,
        run_static_analysis=static_ok,
        run_style_check=style_check_reporting_wrong_file,
        sleep=lambda s: None,
    )

    assert result.success
    findings = result.payload.files[0].findings
    assert len(findings) == 1
    assert findings[0].file == "app.py"  # overridden, not the model's "unknown"


def test_comment_post_failure_retries_then_falls_back_to_summary(tmp_path):
    """Failure mode 2: inline comment posting fails -> retry with
    exponential backoff (max 3), then fall back to one summary comment
    listing what couldn't be posted inline.
    """
    changed_files = [ChangedFile(filename="app.py", status="modified", additions=2, deletions=0, patch="@@ -1 +1 @@\n+x=1")]
    inline_attempts = {"n": 0}

    def flaky_post_inline(**kwargs):
        inline_attempts["n"] += 1
        return Result[PostedComment].fail(ToolError(code=ErrorCode.UPSTREAM_ERROR, message="502", retriable=True))

    github = _FakeGitHubClient(Result[list[ChangedFile]].ok(changed_files), post_inline=flaky_post_inline)
    tracer = _tracer(tmp_path)

    def static_ok(files, **kwargs):
        return Result[list[Finding]].ok(
            [Finding(file="app.py", line=1, rule_id="F401", severity=Severity.ERROR, message="unused import")]
        )

    def style_ok(diff, style_guide):
        return Result[list[StyleViolation]].ok([])

    result = review_pull_request(
        "acme",
        "widgets",
        1,
        "style guide",
        github=github,
        repo_root=str(tmp_path),
        commit_sha="sha",
        tracer=tracer,
        run_static_analysis=static_ok,
        run_style_check=style_ok,
        sleep=lambda s: None,
    )

    assert result.success
    assert inline_attempts["n"] == 3  # hard-capped retries: 1 initial + 2 retries, never unbounded
    outcome = result.payload
    assert outcome.comment_outcome.posted_inline == 0
    assert len(outcome.comment_outcome.failed_inline) == 1
    assert outcome.comment_outcome.summary_comment_posted
    assert len(github.summary_calls) == 1
    assert "could not be posted inline" in github.summary_calls[0]


def test_large_diff_is_chunked_by_risk_with_truncation_noted(tmp_path):
    """Failure mode 3: diff over N changed lines -> chunk by file,
    review highest-risk files first (most static findings, then
    largest diff), note truncation explicitly in the summary comment.
    """
    risky = ChangedFile(filename="risky.py", status="modified", additions=50, deletions=0, patch="@@ risky @@")
    big = ChangedFile(filename="big.py", status="modified", additions=800, deletions=0, patch="@@ big @@")
    safe = ChangedFile(filename="safe.py", status="modified", additions=30, deletions=0, patch="@@ safe @@")
    changed_files = [risky, big, safe]

    github = _FakeGitHubClient(Result[list[ChangedFile]].ok(changed_files))
    tracer = _tracer(tmp_path)

    def static_analyzer(files, **kwargs):
        filename = files[0]
        if filename == "risky.py":
            return Result[list[Finding]].ok(
                [Finding(file="risky.py", line=1, rule_id="F821", severity=Severity.ERROR, message="undefined name")]
            )
        return Result[list[Finding]].ok([])

    style_checked_diffs = []

    def style_check(diff, style_guide):
        style_checked_diffs.append(diff)
        return Result[list[StyleViolation]].ok([])

    result = review_pull_request(
        "acme",
        "widgets",
        1,
        "style guide",
        github=github,
        repo_root=str(tmp_path),
        commit_sha="sha",
        tracer=tracer,
        run_static_analysis=static_analyzer,
        run_style_check=style_check,
        max_diff_lines=100,
        sleep=lambda s: None,
    )

    assert result.success
    outcome = result.payload
    # risky.py ranks highest (has a finding) and fits the 100-line budget alone;
    # big.py and safe.py rank lower and get truncated once the budget is spent.
    assert set(outcome.truncated_files) == {"big.py", "safe.py"}
    assert style_checked_diffs == ["@@ risky @@"]  # only the selected file was sent to the LLM

    statuses = {f.filename: f.status for f in outcome.files}
    assert statuses["risky.py"] == FileReviewStatus.FULL
    assert statuses["big.py"] == FileReviewStatus.TRUNCATED
    assert statuses["safe.py"] == FileReviewStatus.TRUNCATED

    assert len(github.summary_calls) == 1
    assert "not fully reviewed" in github.summary_calls[0]
    assert "big.py" in github.summary_calls[0]


def test_style_check_repeated_failure_is_capped_and_does_not_hang(tmp_path):
    """Failure mode 4: hard cap on retries for any single tool call --
    never loop indefinitely, even when a tool fails every time.
    """
    changed_files = [ChangedFile(filename="app.py", status="modified", additions=5, deletions=0, patch="@@ -1 +1 @@\n+x=1")]
    github = _FakeGitHubClient(Result[list[ChangedFile]].ok(changed_files))
    tracer = _tracer(tmp_path)

    style_attempts = {"n": 0}

    def always_failing_style_check(diff, style_guide):
        style_attempts["n"] += 1
        return Result[list[StyleViolation]].fail(ToolError(code=ErrorCode.RATE_LIMITED, message="rate limited", retriable=True))

    def static_ok(files, **kwargs):
        return Result[list[Finding]].ok([])

    result = review_pull_request(
        "acme",
        "widgets",
        1,
        "style guide",
        github=github,
        repo_root=str(tmp_path),
        commit_sha="sha",
        tracer=tracer,
        run_static_analysis=static_ok,
        run_style_check=always_failing_style_check,
        sleep=lambda s: None,
    )

    assert result.success  # degrades gracefully rather than hanging or raising
    assert style_attempts["n"] == 3  # hard-capped, never unbounded
    file_summary = result.payload.files[0]
    assert file_summary.status == FileReviewStatus.STYLE_CHECK_FAILED
    assert file_summary.findings == []


def test_everything_fails_at_once_still_returns_gracefully(tmp_path):
    """Compound failure: static analysis, style check, inline posting,
    and the summary-comment fallback all fail. The loop must still
    return a Result, never raise.
    """
    changed_files = [ChangedFile(filename="app.py", status="modified", additions=5, deletions=0, patch="@@ -1 +1 @@\n+x=1")]

    def failing_post_inline(**kwargs):
        return Result[PostedComment].fail(ToolError(code=ErrorCode.UPSTREAM_ERROR, message="down", retriable=True))

    def failing_post_summary(**kwargs):
        return Result[PostedComment].fail(ToolError(code=ErrorCode.UPSTREAM_ERROR, message="down", retriable=True))

    github = _FakeGitHubClient(
        Result[list[ChangedFile]].ok(changed_files),
        post_inline=failing_post_inline,
        post_summary=failing_post_summary,
    )
    tracer = _tracer(tmp_path)

    def failing_static(files, **kwargs):
        return Result[list[Finding]].fail(ToolError(code=ErrorCode.TIMEOUT, message="timed out", retriable=True))

    def failing_style(diff, style_guide):
        return Result[list[StyleViolation]].fail(ToolError(code=ErrorCode.RATE_LIMITED, message="rate limited", retriable=True))

    result = review_pull_request(
        "acme",
        "widgets",
        1,
        "style guide",
        github=github,
        repo_root=str(tmp_path),
        commit_sha="sha",
        tracer=tracer,
        run_static_analysis=failing_static,
        run_style_check=failing_style,
        sleep=lambda s: None,
    )

    assert result.success
    outcome = result.payload
    assert outcome.files[0].status == FileReviewStatus.PARTIAL_REVIEW
    assert outcome.comment_outcome.posted_inline == 0
    assert not outcome.comment_outcome.summary_comment_posted
    assert outcome.comment_outcome.summary_comment_error is not None
