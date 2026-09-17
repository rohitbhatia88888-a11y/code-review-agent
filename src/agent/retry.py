"""Bounded retry with exponential backoff for tool calls that return a
Result. Only retries errors the tool itself marked retriable, and never
exceeds ``max_attempts`` regardless of how many times a call keeps
failing -- this is the one place retry policy lives, so no tool call in
the agent can loop indefinitely.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from src.tools.models import Result

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0


def call_with_retry(
    call: Callable[[], Result[T]],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Result[T]:
    """Call ``call`` up to ``max_attempts`` times total.

    Stops immediately (no further attempts) on success or on a
    non-retriable failure. Between retriable failures, sleeps
    ``base_delay * 2**(attempt - 1)`` -- 1x, 2x, 4x, ... -- via the
    injected ``sleep`` so tests never actually wait.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    result = call()
    attempt = 1
    while not result.success and result.error.retriable and attempt < max_attempts:
        sleep(base_delay * (2 ** (attempt - 1)))
        attempt += 1
        result = call()
    return result
