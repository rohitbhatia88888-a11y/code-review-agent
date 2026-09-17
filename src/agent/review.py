"""CLI entry point: ``python -m src.agent.review --pr-url <url>``.

Reviews one GitHub PR through the same decision loop the eval harness
runs. Defaults to a dry run -- findings are computed and printed, but
nothing is posted to GitHub -- unless ``--post`` is passed.

Assumes it's run from (or given, via --repo-root) a local checkout of
the PR's head commit -- static analysis runs against files on disk, and
this CLI doesn't clone or check anything out itself.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from src.agent.decision_loop import DEFAULT_MAX_DIFF_LINES, review_pull_request
from src.agent.models import FileReviewStatus, PRReviewOutcome, TriageAction
from src.agent.tracer import DecisionTracer
from src.tools.github_client import GitHubClient
from src.tools.models import PostedComment, Result

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STYLE_GUIDE_PATH = REPO_ROOT / "STYLE_GUIDE.md"
DEFAULT_TRACE_DIR = Path("results/traces")

_PR_URL_RE = re.compile(r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)/?$")


def parse_pr_url(url: str) -> tuple[str, str, int]:
    match = _PR_URL_RE.match(url.strip())
    if not match:
        raise ValueError(f"not a GitHub PR URL: {url!r} (expected https://github.com/<owner>/<repo>/pull/<number>)")
    return match["owner"], match["repo"], int(match["number"])


class _DryRunGitHubClient:
    """Wraps a real GitHubClient: reads pass through unchanged, writes
    are recorded but never sent. Backs the CLI's default dry-run mode.
    """

    def __init__(self, real_client: GitHubClient) -> None:
        self._real = real_client
        self.would_post_inline: list[dict] = []
        self.would_post_summary: list[str] = []

    def fetch_changed_files(self, owner, repo, pr_number):
        return self._real.fetch_changed_files(owner, repo, pr_number)

    def post_inline_comment(self, owner, repo, pr_number, *, commit_sha, file, line, body):
        self.would_post_inline.append({"file": file, "line": line, "body": body})
        return Result[PostedComment].ok(PostedComment(comment_id=-1, url="dry-run://inline"))

    def post_summary_comment(self, owner, repo, pr_number, *, body):
        self.would_post_summary.append(body)
        return Result[PostedComment].ok(PostedComment(comment_id=-1, url="dry-run://summary"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.agent.review", description="Review a single GitHub PR.")
    parser.add_argument("--pr-url", required=True, help="e.g. https://github.com/owner/repo/pull/123")
    parser.add_argument("--post", action="store_true", help="Actually post comments to GitHub (default: dry run, prints only)")
    parser.add_argument("--style-guide", type=Path, default=DEFAULT_STYLE_GUIDE_PATH, help="Path to the style guide markdown file")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Local checkout to run static analysis against")
    parser.add_argument("--max-diff-lines", type=int, default=DEFAULT_MAX_DIFF_LINES)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR, help="Directory to write the decision trace to")
    return parser


def run(args: argparse.Namespace, real_client: GitHubClient, *, out=sys.stdout, err=sys.stderr) -> int:
    """The testable core: everything after argument parsing and
    GitHubClient construction. Takes an already-built ``real_client`` so
    tests can inject a fake one without a token or network access.
    """
    try:
        owner, repo, pr_number = parse_pr_url(args.pr_url)
    except ValueError as exc:
        print(f"error: {exc}", file=err)
        return 2

    if not args.style_guide.exists():
        print(f"error: style guide not found at {args.style_guide}", file=err)
        return 2
    style_guide = args.style_guide.read_text()

    sha_result = real_client.fetch_pr_head_sha(owner, repo, pr_number)
    if not sha_result.success:
        print(f"error: couldn't fetch PR head commit: {sha_result.error.message}", file=err)
        return 1
    commit_sha = sha_result.payload

    github = real_client if args.post else _DryRunGitHubClient(real_client)

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.trace_dir / f"{owner}-{repo}-pr{pr_number}.jsonl"
    tracer = DecisionTracer(trace_path)

    result = review_pull_request(
        owner,
        repo,
        pr_number,
        style_guide,
        github=github,
        repo_root=str(args.repo_root),
        commit_sha=commit_sha,
        tracer=tracer,
        max_diff_lines=args.max_diff_lines,
    )

    if not result.success:
        print(f"error: review failed: {result.error.message}", file=err)
        return 1

    _print_summary(result.payload, dry_run=not args.post, out=out)
    print(f"\nDecision trace: {trace_path}", file=out)
    return 0


def _print_summary(outcome: PRReviewOutcome, *, dry_run: bool, out) -> None:
    print(f"Reviewed {outcome.owner}/{outcome.repo}#{outcome.pr_number}", file=out)
    if dry_run:
        print("(dry run -- no comments posted; pass --post to post them)", file=out)
    print(file=out)

    for file_summary in outcome.files:
        kept = [f for f in file_summary.findings if f.action == TriageAction.KEPT]
        status_note = "" if file_summary.status == FileReviewStatus.FULL else f"  [{file_summary.status.value}]"
        print(f"{file_summary.filename}{status_note}: {len(kept)} finding(s)", file=out)
        for finding in kept:
            print(f"  {finding.file}:{finding.line} [{finding.severity.value}] {finding.message}", file=out)

    print(file=out)
    print(f"Posted inline: {outcome.comment_outcome.posted_inline}", file=out)
    if outcome.comment_outcome.failed_inline:
        print(f"Failed to post inline (see summary comment fallback): {len(outcome.comment_outcome.failed_inline)}", file=out)
    print(f"Summary comment posted: {outcome.comment_outcome.summary_comment_posted}", file=out)
    if outcome.truncated_files:
        print(f"Truncated (diff-size budget exceeded): {', '.join(outcome.truncated_files)}", file=out)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: GITHUB_TOKEN is not set", file=sys.stderr)
        return 2

    return run(args, GitHubClient(token))


if __name__ == "__main__":
    sys.exit(main())
