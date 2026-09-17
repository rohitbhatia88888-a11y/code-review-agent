from types import SimpleNamespace

from src.tools.models import ErrorCode
from src.tools.style_checker import check_style


class _FakeMessages:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.messages = _FakeMessages(response=response, exc=exc)


def _tool_use_response(violations: list[dict]):
    block = SimpleNamespace(type="tool_use", name="report_style_violations", input={"violations": violations})
    return SimpleNamespace(content=[block])


def test_check_style_returns_violations(sample_diff):
    response = _tool_use_response(
        [
            {
                "file": "app.py",
                "line": 4,
                "violation": "function name should be snake_case",
                "suggested_fix": "rename addNumbers to add_numbers",
            }
        ]
    )
    client = _FakeClient(response=response)

    result = check_style(sample_diff, "Use snake_case for function names.", client=client)

    assert result.success
    assert len(result.payload) == 1
    assert result.payload[0].file == "app.py"
    assert result.payload[0].line == 4
    assert result.payload[0].suggested_fix == "rename addNumbers to add_numbers"


def test_check_style_empty_violations_is_success(sample_diff):
    client = _FakeClient(response=_tool_use_response([]))

    result = check_style(sample_diff, "Use snake_case.", client=client)

    assert result.success
    assert result.payload == []


def test_check_style_model_skips_tool_call(sample_diff):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="looks fine, no tool call")])
    client = _FakeClient(response=response)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.PARSE_ERROR


def test_check_style_malformed_tool_input(sample_diff):
    response = _tool_use_response([{"file": "app.py", "line": "not-a-number", "violation": "x"}])
    client = _FakeClient(response=response)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.PARSE_ERROR


def test_check_style_missing_api_key_without_injected_client(monkeypatch, sample_diff):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = check_style(sample_diff, "guide")

    assert not result.success
    assert result.error.code == ErrorCode.AUTH_FAILED


def test_check_style_rate_limited(sample_diff):
    import anthropic

    fake_request = SimpleNamespace(method="POST", url="https://api.anthropic.com/v1/messages")
    exc = anthropic.RateLimitError(
        message="rate limited",
        response=SimpleNamespace(status_code=429, headers={}, request=fake_request),
        body=None,
    )
    client = _FakeClient(exc=exc)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.RATE_LIMITED
    assert result.error.retriable
