import io

import pytest

from src.agent.review import build_arg_parser, parse_pr_url, run
from src.tools.models import ChangedFile, ErrorCode, PostedComment, Result, ToolError


def test_parse_pr_url_valid():
    assert parse_pr_url("https://github.com/acme/widgets/pull/42") == ("acme", "widgets", 42)


def test_parse_pr_url_valid_with_trailing_slash():
    assert parse_pr_url("https://github.com/acme/widgets/pull/42/") == ("acme", "widgets", 42)


@pytest.mark.parametrize(
    "url",
    [
        "not a url",
        "https://github.com/acme/widgets/issues/42",
        "https://gitlab.com/acme/widgets/pull/42",
        "https://github.com/acme/widgets/pull/not-a-number",
    ],
)
def test_parse_pr_url_rejects_non_pr_urls(url):
    with pytest.raises(ValueError, match="not a GitHub PR URL"):
        parse_pr_url(url)


class _FakeClient:
    def __init__(self, *, changed_files=None, sha_result=None, post_inline=None, post_summary=None):
        self._changed_files = changed_files if changed_files is not None else []
        self._sha_result = sha_result or Result[str].ok("deadbeef")
        self._post_inline = post_inline
        self._post_summary = post_summary
        self.inline_calls = []
        self.summary_calls = []

    def fetch_pr_head_sha(self, owner, repo, pr_number):
        return self._sha_result

    def fetch_changed_files(self, owner, repo, pr_number):
        return Result[list[ChangedFile]].ok(self._changed_files)

    def post_inline_comment(self, owner, repo, pr_number, *, commit_sha, file, line, body):
        self.inline_calls.append((file, line))
        if self._post_inline:
            return self._post_inline(file=file, line=line, body=body)
        return Result[PostedComment].ok(PostedComment(comment_id=1, url="https://x"))

    def post_summary_comment(self, owner, repo, pr_number, *, body):
        self.summary_calls.append(body)
        if self._post_summary:
            return self._post_summary(body=body)
        return Result[PostedComment].ok(PostedComment(comment_id=2, url="https://x"))


def _parse_args(extra, tmp_path, style_guide_path):
    return build_arg_parser().parse_args(
        [
            "--pr-url",
            "https://github.com/acme/widgets/pull/7",
            "--style-guide",
            str(style_guide_path),
            "--trace-dir",
            str(tmp_path / "traces"),
            *extra,
        ]
    )


@pytest.fixture
def style_guide_path(tmp_path):
    path = tmp_path / "STYLE_GUIDE.md"
    path.write_text("Use snake_case.\n")
    return path


def test_dry_run_never_calls_post(tmp_path, style_guide_path):
    client = _FakeClient(changed_files=[ChangedFile(filename="a.py", status="modified", additions=1, deletions=0, patch="@@ -0,0 +1 @@\n+x=1")])
    args = _parse_args([], tmp_path, style_guide_path)
    out = io.StringIO()

    code = run(args, client, out=out, err=out)

    assert code == 0
    assert client.inline_calls == []  # dry run never reaches the real client's post methods
    assert client.summary_calls == []
    assert "dry run" in out.getvalue()


def test_post_flag_calls_through_to_real_client(tmp_path, style_guide_path):
    client = _FakeClient(changed_files=[ChangedFile(filename="a.py", status="modified", additions=1, deletions=0, patch="@@ -0,0 +1 @@\n+x=1")])
    args = _parse_args(["--post"], tmp_path, style_guide_path)
    out = io.StringIO()

    code = run(args, client, out=out, err=out)

    assert code == 0
    assert len(client.summary_calls) == 1  # summary comment always posts; no findings in this fixture
    assert "dry run" not in out.getvalue()


def test_missing_style_guide_is_a_clean_error(tmp_path):
    client = _FakeClient()
    args = build_arg_parser().parse_args(
        [
            "--pr-url",
            "https://github.com/acme/widgets/pull/7",
            "--style-guide",
            str(tmp_path / "does-not-exist.md"),
            "--trace-dir",
            str(tmp_path / "traces"),
        ]
    )
    out = io.StringIO()

    code = run(args, client, out=out, err=out)

    assert code == 2
    assert "style guide not found" in out.getvalue()


def test_invalid_pr_url_is_a_clean_error(tmp_path, style_guide_path):
    client = _FakeClient()
    args = build_arg_parser().parse_args(
        [
            "--pr-url",
            "https://not-github.com/acme/widgets/pull/7",
            "--style-guide",
            str(style_guide_path),
            "--trace-dir",
            str(tmp_path / "traces"),
        ]
    )
    out = io.StringIO()

    code = run(args, client, out=out, err=out)

    assert code == 2
    assert "not a GitHub PR URL" in out.getvalue()


def test_head_sha_failure_is_reported_not_raised(tmp_path, style_guide_path):
    client = _FakeClient(sha_result=Result[str].fail(ToolError(code=ErrorCode.NOT_FOUND, message="no such PR")))
    args = _parse_args([], tmp_path, style_guide_path)
    out = io.StringIO()

    code = run(args, client, out=out, err=out)

    assert code == 1
    assert "couldn't fetch PR head commit" in out.getvalue()


def test_writes_decision_trace(tmp_path, style_guide_path):
    client = _FakeClient(changed_files=[])
    args = _parse_args([], tmp_path, style_guide_path)
    out = io.StringIO()

    run(args, client, out=out, err=out)

    trace_path = tmp_path / "traces" / "acme-widgets-pr7.jsonl"
    assert trace_path.exists()
