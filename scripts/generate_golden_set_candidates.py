#!/usr/bin/env python3
"""Generate the golden-set candidates for human review.

Writes ``eval/golden_set.candidate.json`` -- 25 synthetic one-file PRs
spanning real_bug / clean / style_only / false_positive_trap. Every
case is written with ``reviewed=False``; nothing here is ground truth
until a human reviews it and the set is frozen with
``src.eval.golden_set.freeze``.

Line numbers for expected findings are never hand-counted: each target
line is annotated inline with ``# GOLDEN:MARK:<name>`` in the *_after
source below, and ``_marked_file`` strips the marker and records its
line number, so the file a reader sees here is exactly (module the
marker comment) what ends up in the golden set, with no separate
source of truth to drift out of sync.

Re-run this script any time to regenerate the candidate file from
scratch; it never touches the frozen ``eval/golden_set.json`` if one
exists.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.golden_set import save_golden_set
from src.eval.models import (
    BugType,
    CaseCategory,
    ExpectedFinding,
    GoldenCase,
    GoldenSet,
    PRFile,
)

CANDIDATE_PATH = REPO_ROOT / "eval" / "golden_set.candidate.json"

_MARK_RE = re.compile(r"[ \t]*# GOLDEN:MARK:(\S+)[ \t]*$", re.MULTILINE)


def _marked_file(filename: str, content_before: str | None, content_after_marked: str) -> tuple[PRFile, dict[str, int]]:
    """Strip ``# GOLDEN:MARK:<name>`` markers from ``content_after_marked``,
    returning the PRFile with clean content plus a {name: line_number}
    map for building ExpectedFindings against it.
    """
    marks: dict[str, int] = {}
    lines = content_after_marked.splitlines()
    cleaned_lines = []
    for i, line in enumerate(lines, start=1):
        m = _MARK_RE.search(line)
        if m:
            marks[m.group(1)] = i
            line = _MARK_RE.sub("", line)
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    if content_after_marked.endswith("\n"):
        cleaned += "\n"
    return PRFile(filename=filename, content_before=content_before, content_after=cleaned), marks


STYLE_GUIDE_PATH = REPO_ROOT / "STYLE_GUIDE.md"


def _real_bug(case_id, bug_type, description, filename, before, after_marked, *, optional_marks=()):
    pr_file, marks = _marked_file(filename, before, after_marked)
    expected = [
        ExpectedFinding(
            file=filename,
            line=marks["bug"],
            category="security" if bug_type in (BugType.SQL_INJECTION, BugType.HARDCODED_SECRET, BugType.INSECURE_DESERIALIZATION) else "bug",
            severity="error",
            required=True,
            description=description,
        )
    ]
    for name in optional_marks:
        expected.append(
            ExpectedFinding(
                file=filename,
                line=marks[name],
                category="style",
                severity="info",
                required=False,
                description="incidental, acceptable if flagged but not required",
            )
        )
    return GoldenCase(
        case_id=case_id,
        category=CaseCategory.REAL_BUG,
        bug_type=bug_type,
        description=description,
        files=[pr_file],
        expected_findings=expected,
    )


def _clean(case_id, description, filename, before, after):
    pr_file = PRFile(filename=filename, content_before=before, content_after=after)
    return GoldenCase(case_id=case_id, category=CaseCategory.CLEAN, description=description, files=[pr_file])


def _style_only(case_id, description, filename, before, after_marked, *, optional_marks=()):
    pr_file, marks = _marked_file(filename, before, after_marked)
    expected = [
        ExpectedFinding(
            file=filename, line=marks["style"], category="style", severity="info", required=True, description=description
        )
    ]
    for name in optional_marks:
        expected.append(
            ExpectedFinding(
                file=filename,
                line=marks[name],
                category="style",
                severity="info",
                required=False,
                description="a downstream consequence of the same root violation, acceptable if also flagged",
            )
        )
    return GoldenCase(case_id=case_id, category=CaseCategory.STYLE_ONLY, description=description, files=[pr_file], expected_findings=expected)


def _trap(case_id, description, filename, before, after):
    pr_file = PRFile(filename=filename, content_before=before, content_after=after)
    return GoldenCase(case_id=case_id, category=CaseCategory.FALSE_POSITIVE_TRAP, description=description, files=[pr_file])


def build_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    # ---- real_bug ------------------------------------------------------
    cases.append(
        _real_bug(
            "real_bug-001-off-by-one",
            BugType.OFF_BY_ONE,
            "Pagination slice now reads one element past the intended page (end + 1), leaking the next page's first item.",
            "pagination.py",
            before='def get_page(items, page_number, page_size=10):\n    """Return the 1-indexed page of items."""\n    start = (page_number - 1) * page_size\n    end = start + page_size\n    return items[start:end]\n',
            after_marked='def get_page(items, page_number, page_size=10):\n    """Return the 1-indexed page of items."""\n    start = (page_number - 1) * page_size\n    end = start + page_size\n    return items[start:end + 1]  # GOLDEN:MARK:bug\n',
        )
    )
    cases.append(
        _real_bug(
            "real_bug-002-null-check-removed",
            BugType.NULL_CHECK_REMOVED,
            "The None-guard for an optional `user` argument was dropped; crashes with AttributeError when user is anonymous/missing.",
            "user_service.py",
            before='def get_display_name(user):\n    """Return a printable name, or \"Guest\" if user is None."""\n    if user is None:\n        return "Guest"\n    return user.name.upper()\n',
            after_marked='def get_display_name(user):\n    """Return a printable name, or \"Guest\" if user is None."""\n    return user.name.upper()  # GOLDEN:MARK:bug\n',
        )
    )
    cases.append(
        _real_bug(
            "real_bug-003-sql-injection",
            BugType.SQL_INJECTION,
            "Email is interpolated directly into the SQL string instead of using a parameterized query -- classic SQL injection.",
            "reports.py",
            before='def find_user_by_email(cursor, email):\n    """Look up a user row by email address."""\n    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))\n    return cursor.fetchone()\n',
            after_marked='def find_user_by_email(cursor, email):\n    """Look up a user row by email address."""\n    query = f"SELECT * FROM users WHERE email = \'{email}\'"  # GOLDEN:MARK:bug\n    cursor.execute(query)\n    return cursor.fetchone()\n',
        )
    )
    cases.append(
        _real_bug(
            "real_bug-004-resource-leak",
            BugType.RESOURCE_LEAK,
            "File handle is opened without `with` and never closed -- leaks a file descriptor on every call.",
            "file_utils.py",
            before='def read_config(path):\n    """Read and return the full contents of a config file."""\n    with open(path) as f:\n        return f.read()\n',
            after_marked='def read_config(path):\n    """Read and return the full contents of a config file."""\n    f = open(path)  # GOLDEN:MARK:bug\n    data = f.read()\n    return data\n',
        )
    )
    cases.append(
        _real_bug(
            "real_bug-005-incorrect-comparison",
            BugType.INCORRECT_COMPARISON,
            "Uses `is` to compare a string literal instead of `==` -- relies on CPython string interning, unreliable across builds/inputs.",
            "auth.py",
            before='def is_admin(role):\n    """Return True if role is the admin role."""\n    if role == "admin":\n        return True\n    return False\n',
            after_marked='def is_admin(role):\n    """Return True if role is the admin role."""\n    if role is "admin":  # GOLDEN:MARK:bug\n        return True\n    return False\n',
        )
    )
    cases.append(
        _real_bug(
            "real_bug-006-mutable-default-arg",
            BugType.MUTABLE_DEFAULT_ARG,
            "Mutable list default argument is shared and accumulates across calls instead of starting fresh each time.",
            "cart.py",
            before='def add_item(item, cart=None):\n    """Add item to cart, returning the updated cart."""\n    cart = cart if cart is not None else []\n    cart.append(item)\n    return cart\n',
            after_marked='def add_item(item, cart=[]):  # GOLDEN:MARK:bug\n    """Add item to cart, returning the updated cart."""\n    cart.append(item)\n    return cart\n',
        )
    )
    cases.append(
        _real_bug(
            "real_bug-007-broad-except-swallowed",
            BugType.BROAD_EXCEPT_SWALLOWED,
            "Catches Exception and silently discards it with `pass` -- a failed charge looks like success to the caller.",
            "payment.py",
            before='def charge_card(gateway, amount):\n    """Charge amount via gateway; propagates gateway errors."""\n    try:\n        return gateway.charge(amount)\n    except GatewayError:\n        logger.error("charge failed for amount=%s", amount)\n        raise\n',
            after_marked='def charge_card(gateway, amount):\n    """Charge amount via gateway; propagates gateway errors."""\n    try:\n        return gateway.charge(amount)\n    except Exception:  # GOLDEN:MARK:bug\n        pass\n',
        )
    )
    cases.append(
        _real_bug(
            "real_bug-008-hardcoded-secret",
            BugType.HARDCODED_SECRET,
            "Live-looking auth token is hardcoded as a module constant instead of read from configuration/environment.",
            "client.py",
            before=(
                "import os\n\n\n"
                "class ApiClient:\n"
                '    """Minimal client stub."""\n\n'
                "    def __init__(self, auth_token):\n"
                "        self.auth_token = auth_token\n\n\n"
                'AUTH_TOKEN = os.environ["PAYMENTS_AUTH_TOKEN"]\n\n\n'
                "def create_client():\n"
                '    """Build an ApiClient using the configured auth token."""\n'
                "    return ApiClient(auth_token=AUTH_TOKEN)\n"
            ),
            after_marked=(
                "class ApiClient:\n"
                '    """Minimal client stub."""\n\n'
                "    def __init__(self, auth_token):\n"
                "        self.auth_token = auth_token\n\n\n"
                'AUTH_TOKEN = "sk_live_51Hc8X9JzT3fake000000000000"  # GOLDEN:MARK:bug\n\n\n'
                "def create_client():\n"
                '    """Build an ApiClient using the configured auth token."""\n'
                "    return ApiClient(auth_token=AUTH_TOKEN)\n"
            ),
        )
    )
    cases.append(
        _real_bug(
            "real_bug-009-insecure-deserialization",
            BugType.INSECURE_DESERIALIZATION,
            "Deserializes cached bytes with pickle instead of json -- arbitrary code execution if the cache is ever attacker-controlled.",
            "cache.py",
            before='import json\n\n\ndef load_cached_value(raw_bytes):\n    """Deserialize a cached value previously stored as JSON."""\n    return json.loads(raw_bytes)\n',
            after_marked='import pickle\n\n\ndef load_cached_value(raw_bytes):\n    """Deserialize a cached value previously stored as JSON."""\n    return pickle.loads(raw_bytes)  # GOLDEN:MARK:bug\n',
        )
    )

    # ---- clean -----------------------------------------------------------
    cases.append(
        _clean(
            "clean-001-clamp-helper",
            "New, correctly-implemented bounds-clamping helper.",
            "mathutils.py",
            before=None,
            after='def clamp(value, low, high):\n    """Clamp value into the inclusive range [low, high]."""\n    if value < low:\n        return low\n    if value > high:\n        return high\n    return value\n',
        )
    )
    cases.append(
        _clean(
            "clean-002-retry-with-named-constant",
            "Retry loop using a properly named ALL_CAPS constant and correct raise-on-last-attempt logic.",
            "http_client.py",
            before=None,
            after=(
                "MAX_RETRIES = 3\n\n\n"
                "def fetch_with_retry(client, url):\n"
                '    """GET url, retrying up to MAX_RETRIES times on connection errors."""\n'
                "    for attempt in range(MAX_RETRIES):\n"
                "        try:\n"
                "            return client.get(url)\n"
                "        except ConnectionError:\n"
                "            if attempt == MAX_RETRIES - 1:\n"
                "                raise\n"
                "    return None\n"
            ),
        )
    )
    cases.append(
        _clean(
            "clean-003-point-dataclass",
            "Small, well-formed dataclass with a documented method.",
            "geometry.py",
            before=None,
            after=(
                "from dataclasses import dataclass\n\n\n"
                "@dataclass\n"
                "class Point:\n"
                '    """A 2D point."""\n\n'
                "    x: float\n"
                "    y: float\n\n"
                '    def distance_to(self, other: "Point") -> float:\n'
                '        """Euclidean distance to another point."""\n'
                "        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5\n"
            ),
        )
    )
    cases.append(
        _clean(
            "clean-004-null-safe-accessor",
            "Correctly null-checked accessor (the fixed counterpart of real_bug-002).",
            "profile_service.py",
            before=None,
            after=(
                "def get_display_name(user):\n"
                '    """Return a printable name, or "Guest" if user is None."""\n'
                "    if user is None:\n"
                '        return "Guest"\n'
                "    return user.name.upper()\n"
            ),
        )
    )
    cases.append(
        _clean(
            "clean-005-parameterized-query",
            "Correctly parameterized query (the fixed counterpart of real_bug-003).",
            "user_lookup.py",
            before=None,
            after=(
                "def find_user_by_email(cursor, email):\n"
                '    """Look up a user row by email address."""\n'
                '    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))\n'
                "    return cursor.fetchone()\n"
            ),
        )
    )

    # ---- style_only --------------------------------------------------
    cases.append(
        _style_only(
            "style-001-camelcase-function-name",
            "Function name `calculateTotal` uses camelCase; style guide requires snake_case.",
            "billing.py",
            before='def calculate_total(items):\n    """Sum the price of every item."""\n    total = 0\n    for item in items:\n        total += item.price\n    return total\n',
            after_marked='def calculateTotal(items):  # GOLDEN:MARK:style\n    """Sum the price of every item."""\n    total = 0\n    for item in items:\n        total += item.price\n    return total\n',
        )
    )
    cases.append(
        _style_only(
            "style-002-missing-docstring-and-format",
            "Public function has no docstring and uses `.format()` instead of an f-string.",
            "greetings.py",
            before='def greet(name):\n    """Return a friendly greeting for name."""\n    return f"Hello, {name}"\n',
            after_marked='def greet(name):\n    return "Hello, {}".format(name)  # GOLDEN:MARK:style\n',
        )
    )
    cases.append(
        _style_only(
            "style-003-single-letter-variables",
            "Non-index variables `t` and `x` violate the no-single-letter-names rule (only loop indices i/j/k are exempt).",
            "billing_totals.py",
            before='def total_price(items):\n    """Sum the price of every item."""\n    total = 0\n    for item in items:\n        total += item.price\n    return total\n',
            after_marked='def total_price(items):\n    """Sum the price of every item."""\n    t = 0\n    for x in items:  # GOLDEN:MARK:style\n        t += x.price\n    return t\n',
        )
    )
    cases.append(
        _style_only(
            "style-004-constant-not-all-caps",
            "Module-level constant `default_timeout` is not ALL_CAPS.",
            "settings.py",
            before='DEFAULT_TIMEOUT = 30\n\n\ndef get_timeout():\n    """Return the default request timeout in seconds."""\n    return DEFAULT_TIMEOUT\n',
            after_marked='default_timeout = 30  # GOLDEN:MARK:style\n\n\ndef get_timeout():\n    """Return the default request timeout in seconds."""\n    return default_timeout\n',
        )
    )
    cases.append(
        _style_only(
            "style-005-wildcard-import",
            "Wildcard import (`from utils import *`) is banned by the style guide.",
            "text_processing.py",
            before='from utils import strip_whitespace\n\n\ndef normalize(text):\n    """Strip whitespace and lowercase text."""\n    return strip_whitespace(text).lower()\n',
            after_marked=(
                "from utils import *  # GOLDEN:MARK:style\n\n\n"
                "def normalize(text):\n"
                '    """Strip whitespace and lowercase text."""\n'
                "    return strip_whitespace(text).lower()  # GOLDEN:MARK:usage\n"
            ),
            optional_marks=("usage",),
        )
    )

    # ---- false_positive_trap ------------------------------------------
    cases.append(
        _trap(
            "trap-001-intentional-inclusive-range",
            "The `+ 1` looks like the off-by-one bug pattern, but the function is documented as inclusive-of-both-endpoints and the math is correct.",
            "date_range.py",
            before=None,
            after=(
                "def days_inclusive(start_day, end_day):\n"
                '    """Count days from start_day to end_day, inclusive of both endpoints."""\n'
                "    return end_day - start_day + 1\n"
            ),
        )
    )
    cases.append(
        _trap(
            "trap-002-broad-except-but-relogged-and-reraised",
            "Looks like the swallowed-exception bug pattern, but it logs with full context and re-raises -- nothing is discarded. "
            "(Note: static analysis's blind-except rule cannot always see the re-raise and may still flag this -- a known, honest false-positive risk this case is designed to surface.)",
            "batch_processor.py",
            before=None,
            after=(
                "import logging\n\n"
                "logger = logging.getLogger(__name__)\n\n\n"
                "def _process(batch):\n"
                '    """Process a single batch; raises on failure."""\n'
                "    raise NotImplementedError\n\n\n"
                "def process_batch(batch):\n"
                '    """Process batch, logging and re-raising on failure."""\n'
                "    try:\n"
                "        return _process(batch)\n"
                "    except Exception:\n"
                '        logger.exception("batch processing failed for batch_id=%s", batch.id)\n'
                "        raise\n"
            ),
        )
    )
    cases.append(
        _trap(
            "trap-003-interpolated-string-not-sql",
            "An f-string with interpolation looks superficially like the SQL-injection pattern, but it builds a cache key, not a query -- no SQL, no injection risk.",
            "cache_keys.py",
            before=None,
            after=(
                "def cache_key_for_user(user_id):\n"
                '    """Build the cache key for a user\'s profile."""\n'
                '    return f"user:{user_id}:profile"\n'
            ),
        )
    )
    cases.append(
        _trap(
            "trap-004-subprocess-but-fixed-args",
            "A subprocess call looks risky on sight, but the argument list is fully hardcoded (no untrusted input, no shell=True) -- safe. "
            "(Note: bandit-style rules often flag any subprocess call regardless of input safety -- a known, honest false-positive risk this case is designed to surface.)",
            "vcs_info.py",
            before=None,
            after=(
                "import subprocess\n\n\n"
                "def get_git_commit_hash():\n"
                '    """Return the current commit hash of HEAD."""\n'
                '    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)\n'
                "    return result.stdout.strip()\n"
            ),
        )
    )
    cases.append(
        _trap(
            "trap-005-immutable-default-looks-mutable",
            "Default argument looks like the mutable-default-arg bug pattern, but a tuple is immutable -- there's nothing to accumulate across calls.",
            "summary.py",
            before=None,
            after=(
                'def summarize(items, labels=("total", "count")):\n'
                '    """Return a dict of the given labels mapped to len(items)."""\n'
                "    return {label: len(items) for label in labels}\n"
            ),
        )
    )
    cases.append(
        _trap(
            "trap-006-idiomatic-none-comparison",
            "Uses `is` to compare against None -- the idiomatic, correct pattern (contrast with real_bug-005, which misuses `is` on a string literal).",
            "defaults.py",
            before=None,
            after=(
                "def get_or_default(value, default):\n"
                '    """Return value, or default if value is None."""\n'
                "    if value is None:\n"
                "        return default\n"
                "    return value\n"
            ),
        )
    )

    return cases


def main() -> None:
    golden_set = GoldenSet(style_guide=STYLE_GUIDE_PATH.read_text(), cases=build_cases())
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_golden_set(golden_set, CANDIDATE_PATH)
    by_category: dict[str, int] = {}
    for case in golden_set.cases:
        by_category[case.category.value] = by_category.get(case.category.value, 0) + 1
    print(f"Wrote {len(golden_set.cases)} candidate cases to {CANDIDATE_PATH}")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
