from src.agent.models import FindingSource, TriageAction, TriagedFinding
from src.eval.metrics import aggregate, score_case
from src.eval.models import CaseCategory, ExpectedFinding, GoldenCase, PRFile


def _finding(file="a.py", line=1, severity="error") -> TriagedFinding:
    return TriagedFinding(
        file=file,
        line=line,
        severity=severity,
        source=FindingSource.STATIC_ANALYSIS,
        rule_id="F401",
        message="unused import",
        action=TriageAction.KEPT,
        reason="static analysis finding",
    )


def _case(expected_findings, category=CaseCategory.REAL_BUG) -> GoldenCase:
    return GoldenCase(
        case_id="c1",
        category=category,
        bug_type="off_by_one" if category.value == "real_bug" else None,
        description="d",
        files=[PRFile(filename="a.py", content_before=None, content_after="x = 1\n")],
        expected_findings=expected_findings,
    )


def test_exact_match_counts_as_matched_required():
    expected = [ExpectedFinding(file="a.py", line=5, category="bug", severity="error", required=True, description="d")]
    case = _case(expected)

    match = score_case(case, [_finding(line=5)])

    assert len(match.matched_required) == 1
    assert not match.false_positives
    assert not match.missed_required


def test_finding_within_line_range_matches():
    expected = [
        ExpectedFinding(file="a.py", line=5, line_end=8, category="bug", severity="error", required=True, description="d")
    ]
    case = _case(expected)

    match = score_case(case, [_finding(line=7)])

    assert len(match.matched_required) == 1


def test_finding_outside_line_range_is_false_positive():
    expected = [
        ExpectedFinding(file="a.py", line=5, line_end=8, category="bug", severity="error", required=True, description="d")
    ]
    case = _case(expected)

    match = score_case(case, [_finding(line=9)])

    assert not match.matched_required
    assert len(match.false_positives) == 1
    assert len(match.missed_required) == 1


def test_finding_in_different_file_is_false_positive():
    expected = [ExpectedFinding(file="a.py", line=5, category="bug", severity="error", required=True, description="d")]
    case = _case(expected)

    match = score_case(case, [_finding(file="b.py", line=5)])

    assert len(match.false_positives) == 1
    assert len(match.missed_required) == 1


def test_missed_required_finding_when_nothing_reported():
    expected = [ExpectedFinding(file="a.py", line=5, category="bug", severity="error", required=True, description="d")]
    case = _case(expected)

    match = score_case(case, [])

    assert len(match.missed_required) == 1
    assert not match.matched_required


def test_optional_expected_finding_not_flagged_is_not_a_miss_for_recall():
    expected = [ExpectedFinding(file="a.py", line=5, category="style", severity="info", required=False, description="d")]
    case = _case(expected)

    match = score_case(case, [])

    assert not match.missed_required
    assert len(match.missed_optional) == 1


def test_optional_expected_finding_when_flagged_is_matched_not_false_positive():
    expected = [ExpectedFinding(file="a.py", line=5, category="style", severity="info", required=False, description="d")]
    case = _case(expected)

    match = score_case(case, [_finding(line=5)])

    assert len(match.matched_optional) == 1
    assert not match.false_positives


def test_clean_case_any_finding_is_false_positive():
    case = _case([], category=CaseCategory.CLEAN)

    match = score_case(case, [_finding(line=1)])

    assert match.should_be_silent
    assert match.has_false_positive


def test_two_findings_do_not_double_match_one_expected():
    expected = [ExpectedFinding(file="a.py", line=5, category="bug", severity="error", required=True, description="d")]
    case = _case(expected)

    match = score_case(case, [_finding(line=5), _finding(line=5)])

    assert len(match.matched_required) == 1
    assert len(match.false_positives) == 1  # the second finding at the same spot has nothing left to match


def test_aggregate_computes_recall_precision_and_fp_rate():
    real_bug_expected = [ExpectedFinding(file="a.py", line=5, category="bug", severity="error", required=True, description="d")]
    real_bug_case = _case(real_bug_expected, category=CaseCategory.REAL_BUG)
    clean_case = _case([], category=CaseCategory.CLEAN)

    matches = [
        score_case(real_bug_case, [_finding(line=5)]),  # true positive
        score_case(clean_case, [_finding(line=1)]),  # false positive
    ]

    metrics = aggregate(matches)

    assert metrics.recall == 1.0  # 1/1 required matched
    assert metrics.precision == 0.5  # 1 true positive / 2 total findings
    assert metrics.should_be_silent_cases == 1
    assert metrics.pr_false_positive_rate == 1.0  # the one should-be-silent case had a FP
    assert metrics.false_positive_case_ids == ["c1"]


def test_aggregate_handles_empty_denominators_as_none():
    clean_case = _case([], category=CaseCategory.CLEAN)

    metrics = aggregate([score_case(clean_case, [])])

    assert metrics.recall is None  # no required findings anywhere
    assert metrics.precision is None  # no findings raised at all
    assert metrics.pr_false_positive_rate == 0.0
