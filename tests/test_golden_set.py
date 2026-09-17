import pytest
from pydantic import ValidationError

from src.eval.golden_set import (
    compute_hash,
    freeze,
    is_frozen,
    load_golden_set,
    save_golden_set,
    verify_integrity,
)
from src.eval.models import CaseCategory, ExpectedFinding, GoldenCase, GoldenSet, PRFile


def _case(**overrides) -> GoldenCase:
    defaults = {
        "case_id": "c1",
        "category": CaseCategory.CLEAN,
        "description": "a clean case",
        "files": [PRFile(filename="a.py", content_before=None, content_after="x = 1\n")],
    }
    defaults.update(overrides)
    return GoldenCase(**defaults)


def test_clean_case_with_expected_findings_is_rejected():
    with pytest.raises(ValidationError, match="nothing should be flagged"):
        _case(
            category=CaseCategory.CLEAN,
            expected_findings=[
                ExpectedFinding(file="a.py", line=1, category="bug", severity="error", required=True, description="x")
            ],
        )


def test_real_bug_case_without_bug_type_is_rejected():
    with pytest.raises(ValidationError, match="bug_type"):
        _case(category=CaseCategory.REAL_BUG, bug_type=None)


def test_duplicate_case_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        GoldenSet(style_guide="guide", cases=[_case(case_id="dup"), _case(case_id="dup")])


def test_line_end_defaults_to_line():
    ef = ExpectedFinding(file="a.py", line=5, category="bug", severity="error", required=True, description="x")
    assert ef.line_end == 5


def test_save_and_load_round_trip(tmp_path):
    golden_set = GoldenSet(style_guide="guide", cases=[_case()])
    path = tmp_path / "candidate.json"

    save_golden_set(golden_set, path)
    loaded = load_golden_set(path)

    assert loaded == golden_set


def test_freeze_refuses_unreviewed_cases(tmp_path):
    candidate = tmp_path / "candidate.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_case(reviewed=False)]), candidate)

    with pytest.raises(ValueError, match="not yet reviewed"):
        freeze(candidate, tmp_path / "golden_set.json")


def test_freeze_succeeds_once_every_case_is_reviewed(tmp_path):
    candidate = tmp_path / "candidate.json"
    frozen = tmp_path / "golden_set.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_case(reviewed=True)]), candidate)

    digest = freeze(candidate, frozen)

    assert frozen.exists()
    assert is_frozen(frozen)
    assert compute_hash(load_golden_set(frozen)) == digest
    verify_integrity(frozen)  # does not raise


def test_verify_integrity_raises_if_never_frozen(tmp_path):
    path = tmp_path / "candidate.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_case()]), path)

    with pytest.raises(FileNotFoundError):
        verify_integrity(path)


def test_verify_integrity_detects_post_freeze_edit(tmp_path):
    candidate = tmp_path / "candidate.json"
    frozen = tmp_path / "golden_set.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_case(reviewed=True)]), candidate)
    freeze(candidate, frozen)

    # Someone edits the frozen file by hand after the fact.
    tampered = load_golden_set(frozen)
    tampered.cases[0].description = "quietly changed to improve the score"
    save_golden_set(tampered, frozen)

    with pytest.raises(ValueError, match="does not match its recorded hash"):
        verify_integrity(frozen)


def test_the_generated_candidate_file_loads_and_validates():
    golden_set = load_golden_set("eval/golden_set.candidate.json")

    assert 20 <= len(golden_set.cases) <= 30
    categories = {c.category for c in golden_set.cases}
    assert categories == {
        CaseCategory.REAL_BUG,
        CaseCategory.CLEAN,
        CaseCategory.STYLE_ONLY,
        CaseCategory.FALSE_POSITIVE_TRAP,
    }
    assert all(not c.reviewed for c in golden_set.cases)  # nothing is ground truth yet
