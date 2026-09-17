from src.agent.models import TriageAction
from src.agent.tracer import DecisionTracer
from src.agent.triage import triage_findings
from src.tools.models import Finding, Severity, StyleViolation


def _tracer(tmp_path) -> DecisionTracer:
    return DecisionTracer(tmp_path / "trace.jsonl")


def test_dedup_keeps_static_over_style_at_same_location(tmp_path):
    static = [Finding(file="app.py", line=10, rule_id="E501", severity=Severity.WARNING, message="line too long")]
    style = [StyleViolation(file="app.py", line=10, violation="line too long", suggested_fix="wrap it")]

    triaged = triage_findings(static, style, tracer=_tracer(tmp_path))

    kept = [t for t in triaged if t.action == TriageAction.KEPT]
    dropped = [t for t in triaged if t.action != TriageAction.KEPT]
    assert len(kept) == 1
    assert kept[0].source.value == "static_analysis"
    assert len(dropped) == 1
    assert dropped[0].action == TriageAction.DROPPED_DUPLICATE


def test_suppresses_style_nit_when_real_bug_on_same_line(tmp_path):
    static = [Finding(file="app.py", line=5, rule_id="F821", severity=Severity.ERROR, message="undefined name")]
    style = [StyleViolation(file="app.py", line=5, violation="prefer f-strings", suggested_fix="use f'...'")]

    triaged = triage_findings(static, style, tracer=_tracer(tmp_path))

    style_result = next(t for t in triaged if t.source.value == "style_guide")
    assert style_result.action == TriageAction.DROPPED_SUPPRESSED
    assert "real bug" in style_result.reason


def test_kept_findings_ranked_by_severity(tmp_path):
    static = [
        Finding(file="app.py", line=1, rule_id="W1", severity=Severity.WARNING, message="minor"),
        Finding(file="app.py", line=2, rule_id="F1", severity=Severity.ERROR, message="serious"),
    ]

    triaged = triage_findings(static, [], tracer=_tracer(tmp_path))

    assert [t.severity for t in triaged] == [Severity.ERROR, Severity.WARNING]


def test_independent_style_violation_is_kept(tmp_path):
    style = [StyleViolation(file="app.py", line=20, violation="bad name", suggested_fix="rename")]

    triaged = triage_findings([], style, tracer=_tracer(tmp_path))

    assert len(triaged) == 1
    assert triaged[0].action == TriageAction.KEPT


def test_two_static_findings_same_line_keeps_more_severe(tmp_path):
    static = [
        Finding(file="app.py", line=1, rule_id="E501", severity=Severity.WARNING, message="long line"),
        Finding(file="app.py", line=1, rule_id="F821", severity=Severity.ERROR, message="undefined name"),
    ]

    triaged = triage_findings(static, [], tracer=_tracer(tmp_path))

    assert len(triaged) == 1
    assert triaged[0].rule_id == "F821"


def test_triage_records_trace_entries(tmp_path):
    tracer = _tracer(tmp_path)
    static = [Finding(file="app.py", line=1, rule_id="F401", severity=Severity.ERROR, message="unused import")]

    triage_findings(static, [], tracer=tracer)

    assert len(tracer.entries) == 1
    assert tracer.entries[0].stage == "triage"
    assert tracer.path.read_text().strip() != ""
