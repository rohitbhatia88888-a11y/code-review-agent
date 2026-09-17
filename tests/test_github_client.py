import requests

from src.tools.github_client import GitHubClient
from src.tools.models import ErrorCode


class _FakeResponse:
    def __init__(self, status_code, *, json_data=None, text="", headers=None, url="https://api.github.com/x"):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}
        self.url = url

    def json(self):
        if self._json_data is None:
            raise ValueError("response has no JSON body")
        return self._json_data


class _FakeSession:
    """Returns canned responses in order; raises if it runs out."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)


def _client(responses) -> tuple[GitHubClient, _FakeSession]:
    session = _FakeSession(responses)
    client = GitHubClient("fake-token", session=session)
    return client, session


def test_fetch_pr_diff_success(sample_diff):
    client, session = _client([_FakeResponse(200, text=sample_diff)])

    result = client.fetch_pr_diff("acme", "widgets", 42)

    assert result.success
    assert result.payload.diff == sample_diff
    assert result.payload.pr_number == 42
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert url == "https://api.github.com/repos/acme/widgets/pulls/42"
    assert kwargs["headers"]["Accept"] == "application/vnd.github.v3.diff"


def test_fetch_pr_head_sha_success():
    client, session = _client([_FakeResponse(200, json_data={"head": {"sha": "abc123"}})])

    result = client.fetch_pr_head_sha("acme", "widgets", 42)

    assert result.success
    assert result.payload == "abc123"
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert url == "https://api.github.com/repos/acme/widgets/pulls/42"
    assert kwargs["headers"]["Accept"] == "application/vnd.github+json"


def test_fetch_pr_head_sha_malformed_response():
    client, _ = _client([_FakeResponse(200, json_data={"head": {}})])

    result = client.fetch_pr_head_sha("acme", "widgets", 42)

    assert not result.success
    assert result.error.code == ErrorCode.PARSE_ERROR


def test_fetch_pr_diff_not_found():
    client, _ = _client([_FakeResponse(404, text="Not Found")])

    result = client.fetch_pr_diff("acme", "widgets", 999)

    assert not result.success
    assert result.error.code == ErrorCode.NOT_FOUND
    assert not result.error.retriable


def test_fetch_changed_files_success():
    client, _ = _client(
        [
            _FakeResponse(
                200,
                json_data=[
                    {
                        "filename": "app.py",
                        "status": "modified",
                        "additions": 3,
                        "deletions": 1,
                        "patch": "@@ -1,3 +1,6 @@ ...",
                    }
                ],
            )
        ]
    )

    result = client.fetch_changed_files("acme", "widgets", 42)

    assert result.success
    assert len(result.payload) == 1
    assert result.payload[0].filename == "app.py"
    assert result.payload[0].status == "modified"


def test_fetch_changed_files_paginates():
    page_1 = [{"filename": f"f{i}.py", "status": "modified", "additions": 1, "deletions": 0} for i in range(100)]
    page_2 = [{"filename": "last.py", "status": "added", "additions": 5, "deletions": 0}]
    client, session = _client([_FakeResponse(200, json_data=page_1), _FakeResponse(200, json_data=page_2)])

    result = client.fetch_changed_files("acme", "widgets", 42)

    assert result.success
    assert len(result.payload) == 101
    assert len(session.calls) == 2
    assert session.calls[0][2]["params"] == {"per_page": 100, "page": 1}
    assert session.calls[1][2]["params"] == {"per_page": 100, "page": 2}


def test_fetch_changed_files_rate_limited():
    client, _ = _client(
        [_FakeResponse(403, headers={"X-RateLimit-Remaining": "0", "Retry-After": "30"})]
    )

    result = client.fetch_changed_files("acme", "widgets", 42)

    assert not result.success
    assert result.error.code == ErrorCode.RATE_LIMITED
    assert result.error.retriable
    assert result.error.retry_after_seconds == 30.0


def test_post_inline_comment_success():
    client, session = _client(
        [_FakeResponse(201, json_data={"id": 123, "html_url": "https://github.com/acme/widgets/pull/42#comment-123"})]
    )

    result = client.post_inline_comment(
        "acme", "widgets", 42, commit_sha="abc123", file="app.py", line=5, body="fix this"
    )

    assert result.success
    assert result.payload.comment_id == 123
    method, _, kwargs = session.calls[0]
    assert method == "POST"
    assert kwargs["json"] == {"body": "fix this", "commit_id": "abc123", "path": "app.py", "line": 5}


def test_post_summary_comment_success():
    client, _ = _client(
        [_FakeResponse(201, json_data={"id": 7, "html_url": "https://github.com/acme/widgets/pull/42#issuecomment-7"})]
    )

    result = client.post_summary_comment("acme", "widgets", 42, body="LGTM overall")

    assert result.success
    assert result.payload.comment_id == 7


def test_post_summary_comment_auth_failed():
    client, _ = _client([_FakeResponse(401, json_data={"message": "Bad credentials"})])

    result = client.post_summary_comment("acme", "widgets", 42, body="hi")

    assert not result.success
    assert result.error.code == ErrorCode.AUTH_FAILED


def test_request_timeout_never_raises():
    class _TimeoutSession:
        def request(self, *args, **kwargs):
            raise requests.Timeout("connect timed out")

    client = GitHubClient("fake-token", session=_TimeoutSession())

    result = client.fetch_pr_diff("acme", "widgets", 42)

    assert not result.success
    assert result.error.code == ErrorCode.TIMEOUT
    assert result.error.retriable
