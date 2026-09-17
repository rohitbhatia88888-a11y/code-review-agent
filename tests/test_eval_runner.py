import json

import pytest

from src.eval.golden_set import freeze, save_golden_set
from src.eval.models import CaseCategory, ExpectedFinding, GoldenCase, GoldenSet, PRFile
from src.eval.runner import _LLMCallRecord, run_eval
from src.tools.models import (
    ErrorCode,
    Finding,
    Result,
    Severity,
    StyleViolation,
    ToolError,
)


def _bug_case() -> GoldenCase:
    return GoldenCase(
        case_id="real_bug-fake",
        category=CaseCategory.REAL_BUG,
        bug_type="off_by_one",
        description="fake injected bug",
        files=[PRFile(filename="a.py", content_before="x = 1\n", content_after="x = 2\n")],
        expected_findings=[
            ExpectedFinding(file="a.py", line=1, category="bug", severity="error", required=True, description="d")
        ],
    )


def _clean_case() -> GoldenCase:
    return GoldenCase(
        case_id="clean-fake",
        category=CaseCategory.CLEAN,
        description="fake clean PR",
        files=[PRFile(filename="b.py", content_before=None, content_after="y = 1\n")],
    )


def _fake_static_analysis_hitting(target_filename: str, line: int):
    def run(files, **kwargs):
        if files == [target_filename]:
            return Result[list[Finding]].ok(
                [Finding(file=files[0], line=line, rule_id="FAKE1", severity=Severity.ERROR, message="fake finding")]
            )
        return Result[list[Finding]].ok([])

    return run


def _fake_style_check_factory(violations_by_file: dict[str, list[StyleViolation]], input_tokens=100, output_tokens=50):
    def factory():
        records: list[_LLMCallRecord] = []

        def run_style_check(diff, style_guide):
            records.append(_LLMCallRecord(input_tokens=input_tokens, output_tokens=output_tokens, latency_seconds=0.01))
            return Result[list[StyleViolation]].ok([])

        return run_style_check, records

    return factory


def test_run_eval_writes_summary_and_per_case_files(tmp_path):
    golden_set = GoldenSet(style_guide="guide", cases=[_bug_case(), _clean_case()])
    golden_set_path = tmp_path / "candidate.json"
    save_golden_set(golden_set, golden_set_path)

    metrics = run_eval(
        golden_set_path,
        run_id="test-run",
        results_dir=tmp_path / "results" / "runs",
        run_static_analysis=_fake_static_analysis_hitting("a.py", 1),
        style_check_factory=_fake_style_check_factory({}),
        sleep=lambda s: None,
    )

    assert metrics.recall == 1.0
    assert metrics.pr_false_positive_rate == 0.0

    run_dir = tmp_path / "results" / "runs" / "test-run"
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "cases" / "real_bug-fake.json").exists()
    assert (run_dir / "cases" / "clean-fake.json").exists()

    case_payload = json.loads((run_dir / "cases" / "real_bug-fake.json").read_text())
    assert case_payload["case_id"] == "real_bug-fake"
    assert len(case_payload["matched_required"]) == 1
    assert case_payload["llm_calls"] == 1
    assert case_payload["cost_dollars"] > 0

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["case_count"] == 2
    assert summary["metrics"]["recall"] == 1.0


def test_run_eval_appends_csv_row_per_run(tmp_path):
    golden_set_path = tmp_path / "candidate.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_clean_case()]), golden_set_path)

    run_eval(
        golden_set_path,
        run_id="run-1",
        results_dir=tmp_path / "results" / "runs",
        run_static_analysis=lambda files, **kw: Result[list[Finding]].ok([]),
        style_check_factory=_fake_style_check_factory({}),
        sleep=lambda s: None,
    )
    run_eval(
        golden_set_path,
        run_id="run-2",
        results_dir=tmp_path / "results" / "runs",
        run_static_analysis=lambda files, **kw: Result[list[Finding]].ok([]),
        style_check_factory=_fake_style_check_factory({}),
        sleep=lambda s: None,
    )

    csv_path = tmp_path / "results" / "eval_runs.csv"
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 runs
    assert "run-1" in lines[1]
    assert "run-2" in lines[2]


def test_run_eval_computes_cost_from_token_usage(tmp_path):
    golden_set_path = tmp_path / "candidate.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_clean_case()]), golden_set_path)

    metrics = run_eval(
        golden_set_path,
        run_id="cost-test",
        results_dir=tmp_path / "results" / "runs",
        run_static_analysis=lambda files, **kw: Result[list[Finding]].ok([]),
        style_check_factory=_fake_style_check_factory({}, input_tokens=1_000_000, output_tokens=1_000_000),
        model="claude-sonnet-5",
        sleep=lambda s: None,
    )

    run_dir = tmp_path / "results" / "runs" / "cost-test"
    case_payload = json.loads((run_dir / "cases" / "clean-fake.json").read_text())
    # 1M input tokens @ $2/M + 1M output tokens @ $10/M = $12 for this one PR's single LLM call.
    assert case_payload["cost_dollars"] == pytest.approx(12.0)
    del metrics  # unused here; the assertion is on the per-case file


def test_run_eval_unknown_model_pricing_raises(tmp_path):
    golden_set_path = tmp_path / "candidate.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_clean_case()]), golden_set_path)

    with pytest.raises(ValueError, match="no pricing configured"):
        run_eval(
            golden_set_path,
            run_id="bad-model",
            results_dir=tmp_path / "results" / "runs",
            run_static_analysis=lambda files, **kw: Result[list[Finding]].ok([]),
            style_check_factory=_fake_style_check_factory({}),
            model="some-unpriced-model",
            sleep=lambda s: None,
        )


def test_run_eval_require_frozen_refuses_unfrozen_candidate(tmp_path):
    golden_set_path = tmp_path / "candidate.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_clean_case()]), golden_set_path)

    with pytest.raises(FileNotFoundError):
        run_eval(golden_set_path, results_dir=tmp_path / "results" / "runs", require_frozen=True)


def test_run_eval_require_frozen_succeeds_after_freeze(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    frozen_path = tmp_path / "golden_set.json"
    case = _clean_case()
    case.reviewed = True
    save_golden_set(GoldenSet(style_guide="guide", cases=[case]), candidate_path)
    freeze(candidate_path, frozen_path)

    metrics = run_eval(
        frozen_path,
        run_id="frozen-run",
        results_dir=tmp_path / "results" / "runs",
        run_static_analysis=lambda files, **kw: Result[list[Finding]].ok([]),
        style_check_factory=_fake_style_check_factory({}),
        require_frozen=True,
        sleep=lambda s: None,
    )

    assert metrics.case_count == 1


def test_run_case_survives_static_analyzer_failure_without_crashing(tmp_path):
    golden_set_path = tmp_path / "candidate.json"
    save_golden_set(GoldenSet(style_guide="guide", cases=[_bug_case()]), golden_set_path)

    def crashing_static_analysis(files, **kwargs):
        return Result[list[Finding]].fail(ToolError(code=ErrorCode.TIMEOUT, message="ruff timed out", retriable=True))

    metrics = run_eval(
        golden_set_path,
        run_id="degraded-run",
        results_dir=tmp_path / "results" / "runs",
        run_static_analysis=crashing_static_analysis,
        style_check_factory=_fake_style_check_factory({}),
        sleep=lambda s: None,
    )

    # The bug is real and only static analysis could catch it in this
    # fixture, so a crashing analyzer means a guaranteed miss -- but the
    # run itself must complete and report it, not raise.
    assert metrics.case_count == 1
    assert metrics.recall == 0.0
