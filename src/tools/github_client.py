"""GitHub REST client tool.

Wraps the subset of the GitHub REST API the review agent needs: reading a
PR's diff and changed files, and posting comments back onto it. Every
public method returns a ``Result`` -- expected failures (auth, rate
limits, not found, timeouts, malformed responses) are values, not
exceptions.
"""

from __future__ import annotations

import time

import requests

from src.tools.models import (
    ChangedFile,
    ErrorCode,
    PostedComment,
    PullRequestDiff,
    Result,
    ToolError,
)

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT_SECONDS = 10.0
PAGE_SIZE = 100


class GitHubClient:
    """Thin, injectable wrapper around GitHub's REST API.

    ``session`` is injected so tests can substitute a fake/mocked
    ``requests.Session`` without patching global state.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = GITHUB_API_BASE,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout

    def fetch_pr_head_sha(self, owner: str, repo: str, pr_number: int) -> Result[str]:
        outcome = self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
        if isinstance(outcome, ToolError):
            return Result[str].fail(outcome)
        try:
            return Result[str].ok(outcome.json()["head"]["sha"])
        except (ValueError, KeyError) as exc:
            return Result[str].fail(
                ToolError(code=ErrorCode.PARSE_ERROR, message=f"couldn't parse PR head commit: {exc}")
            )

    def fetch_pr_diff(self, owner: str, repo: str, pr_number: int) -> Result[PullRequestDiff]:
        outcome = self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            accept="application/vnd.github.v3.diff",
        )
        if isinstance(outcome, ToolError):
            return Result[PullRequestDiff].fail(outcome)
        return Result[PullRequestDiff].ok(
            PullRequestDiff(owner=owner, repo=repo, pr_number=pr_number, diff=outcome.text)
        )

    def fetch_changed_files(self, owner: str, repo: str, pr_number: int) -> Result[list[ChangedFile]]:
        files: list[ChangedFile] = []
        page = 1
        while True:
            outcome = self._request(
                "GET",
                f"/repos/{owner}/{repo}/pulls/{pr_number}/files",
                params={"per_page": PAGE_SIZE, "page": page},
            )
            if isinstance(outcome, ToolError):
                return Result[list[ChangedFile]].fail(outcome)

            try:
                batch = outcome.json()
            except ValueError:
                return Result[list[ChangedFile]].fail(
                    ToolError(
                        code=ErrorCode.PARSE_ERROR,
                        message="GitHub returned non-JSON for changed files",
                    )
                )

            try:
                files.extend(
                    ChangedFile(
                        filename=entry["filename"],
                        status=entry["status"],
                        additions=entry["additions"],
                        deletions=entry["deletions"],
                        patch=entry.get("patch"),
                    )
                    for entry in batch
                )
            except KeyError as exc:
                return Result[list[ChangedFile]].fail(
                    ToolError(code=ErrorCode.PARSE_ERROR, message=f"malformed file entry: missing {exc}")
                )

            if len(batch) < PAGE_SIZE:
                break
            page += 1

        return Result[list[ChangedFile]].ok(files)

    def post_inline_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        *,
        commit_sha: str,
        file: str,
        line: int,
        body: str,
    ) -> Result[PostedComment]:
        outcome = self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            json={"body": body, "commit_id": commit_sha, "path": file, "line": line},
        )
        if isinstance(outcome, ToolError):
            return Result[PostedComment].fail(outcome)
        return _posted_comment_from_response(outcome)

    def post_summary_comment(self, owner: str, repo: str, pr_number: int, *, body: str) -> Result[PostedComment]:
        outcome = self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        if isinstance(outcome, ToolError):
            return Result[PostedComment].fail(outcome)
        return _posted_comment_from_response(outcome)

    def _headers(self, accept: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        accept: str = "application/vnd.github+json",
        **kwargs,
    ) -> requests.Response | ToolError:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.request(
                method, url, headers=self._headers(accept), timeout=self._timeout, **kwargs
            )
        except requests.Timeout:
            return ToolError(code=ErrorCode.TIMEOUT, message=f"{method} {url} timed out", retriable=True)
        except requests.RequestException as exc:
            return ToolError(code=ErrorCode.UPSTREAM_ERROR, message=str(exc), retriable=True)

        return _to_tool_error(response) or response


def _to_tool_error(response: requests.Response) -> ToolError | None:
    if response.status_code == 404:
        return ToolError(code=ErrorCode.NOT_FOUND, message=f"{response.url} -> 404", retriable=False)
    if response.status_code == 401:
        return ToolError(code=ErrorCode.AUTH_FAILED, message="GitHub rejected the credentials", retriable=False)
    if response.status_code == 429 or (
        response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0"
    ):
        return ToolError(
            code=ErrorCode.RATE_LIMITED,
            message="GitHub API rate limit exceeded",
            retriable=True,
            retry_after_seconds=_parse_retry_after(response),
        )
    if response.status_code >= 400:
        return ToolError(
            code=ErrorCode.UPSTREAM_ERROR,
            message=f"{response.url} -> {response.status_code}: {_safe_body(response)}",
            retriable=response.status_code >= 500,
        )
    return None


def _posted_comment_from_response(response: requests.Response) -> Result[PostedComment]:
    try:
        body = response.json()
        return Result[PostedComment].ok(PostedComment(comment_id=body["id"], url=body["html_url"]))
    except (ValueError, KeyError) as exc:
        return Result[PostedComment].fail(
            ToolError(code=ErrorCode.PARSE_ERROR, message=f"couldn't parse GitHub comment response: {exc}")
        )


def _parse_retry_after(response: requests.Response) -> float | None:
    if "Retry-After" in response.headers:
        try:
            return float(response.headers["Retry-After"])
        except ValueError:
            pass
    reset = response.headers.get("X-RateLimit-Reset")
    if reset is not None:
        try:
            return max(0.0, float(reset) - time.time())
        except ValueError:
            pass
    return None


def _safe_body(response: requests.Response, limit: int = 300) -> str:
    try:
        message = response.json().get("message", response.text)
    except ValueError:
        message = response.text
    return message[:limit]
