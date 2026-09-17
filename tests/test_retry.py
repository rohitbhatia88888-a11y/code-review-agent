from src.agent.retry import call_with_retry
from src.tools.models import ErrorCode, Result, ToolError


def _fail(code=ErrorCode.RATE_LIMITED, retriable=True):
    return Result[str].fail(ToolError(code=code, message="boom", retriable=retriable))


def test_call_with_retry_succeeds_first_try():
    calls = []

    def call():
        calls.append(1)
        return Result[str].ok("ok")

    result = call_with_retry(call, sleep=lambda s: None)

    assert result.success
    assert len(calls) == 1


def test_call_with_retry_succeeds_after_transient_failures():
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _fail()
        return Result[str].ok("ok")

    slept = []
    result = call_with_retry(call, sleep=slept.append)

    assert result.success
    assert attempts["n"] == 3
    assert slept == [1.0, 2.0]  # exponential backoff: base * 2**0, base * 2**1


def test_call_with_retry_hard_caps_at_max_attempts():
    calls = []

    def call():
        calls.append(1)
        return _fail()

    result = call_with_retry(call, max_attempts=3, sleep=lambda s: None)

    assert not result.success
    assert len(calls) == 3  # never exceeds max_attempts, however long failures continue


def test_call_with_retry_does_not_retry_non_retriable_errors():
    calls = []

    def call():
        calls.append(1)
        return _fail(code=ErrorCode.NOT_FOUND, retriable=False)

    slept = []
    result = call_with_retry(call, sleep=slept.append)

    assert not result.success
    assert len(calls) == 1
    assert slept == []
