from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_diff() -> str:
    return (FIXTURES_DIR / "sample_pr.diff").read_text()


@pytest.fixture
def lint_target_path() -> str:
    return str(FIXTURES_DIR / "lint_target.py")
