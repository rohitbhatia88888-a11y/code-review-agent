import json
from types import SimpleNamespace

from src.tools.models import ErrorCode
from src.tools.style_checker import check_style


class _FakeCompletions:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response=response, exc=exc))


def _tool_call_response(violations: list[dict]):
    call = SimpleNamespace(
        function=SimpleNamespace(name="report_style_violations", arguments=json.dumps({"violations": violations}))
    )
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_check_style_returns_violations(sample_diff):
    response = _tool_call_response(
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
    client = _FakeClient(response=_tool_call_response([]))

    result = check_style(sample_diff, "Use snake_case.", client=client)

    assert result.success
    assert result.payload == []


def test_check_style_model_skips_tool_call(sample_diff):
    message = SimpleNamespace(tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = _FakeClient(response=response)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.PARSE_ERROR


def test_check_style_malformed_tool_arguments(sample_diff):
    call = SimpleNamespace(function=SimpleNamespace(name="report_style_violations", arguments="not json"))
    message = SimpleNamespace(tool_calls=[call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = _FakeClient(response=response)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.PARSE_ERROR


def test_check_style_malformed_violation_fields(sample_diff):
    response = _tool_call_response([{"file": "app.py", "line": "not-a-number", "violation": "x"}])
    client = _FakeClient(response=response)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.PARSE_ERROR


def test_check_style_missing_api_key_without_injected_client(monkeypatch, sample_diff):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = check_style(sample_diff, "guide")

    assert not result.success
    assert result.error.code == ErrorCode.AUTH_FAILED


def test_check_style_rate_limited(sample_diff):
    import openai

    fake_request = SimpleNamespace(method="POST", url="https://openrouter.ai/api/v1/chat/completions")
    exc = openai.RateLimitError(
        message="rate limited",
        response=SimpleNamespace(status_code=429, headers={}, request=fake_request),
        body=None,
    )
    client = _FakeClient(exc=exc)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.RATE_LIMITED
    assert result.error.retriable


def test_check_style_bad_request_is_not_retriable(sample_diff):
    import openai

    fake_request = SimpleNamespace(method="POST", url="https://openrouter.ai/api/v1/chat/completions")
    exc = openai.BadRequestError(
        message="Your credit balance is too low",
        response=SimpleNamespace(status_code=400, headers={}, request=fake_request),
        body=None,
    )
    client = _FakeClient(exc=exc)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.UPSTREAM_ERROR
    assert not result.error.retriable


def test_check_style_server_error_is_retriable(sample_diff):
    import openai

    fake_request = SimpleNamespace(method="POST", url="https://openrouter.ai/api/v1/chat/completions")
    exc = openai.InternalServerError(
        message="internal error",
        response=SimpleNamespace(status_code=500, headers={}, request=fake_request),
        body=None,
    )
    client = _FakeClient(exc=exc)

    result = check_style(sample_diff, "guide", client=client)

    assert not result.success
    assert result.error.code == ErrorCode.UPSTREAM_ERROR
    assert result.error.retriable
