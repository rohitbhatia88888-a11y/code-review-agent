import subprocess

from src.tools.models import ErrorCode, Severity
from src.tools.static_analyzer import run_ruff


def test_run_ruff_finds_unused_import(lint_target_path):
    result = run_ruff([lint_target_path])

    assert result.success
    assert len(result.payload) == 1

    finding = result.payload[0]
    assert finding.file == lint_target_path
    assert finding.line == 1
    assert finding.rule_id == "F401"
    assert finding.severity == Severity.ERROR
    assert "os" in finding.message


def test_run_ruff_empty_file_list_is_success_without_invoking_ruff(monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("ruff should not run when there are no files")

    monkeypatch.setattr(subprocess, "run", _fail_if_called)

    result = run_ruff([])

    assert result.success
    assert result.payload == []


def test_run_ruff_missing_executable(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("ruff not found")

    monkeypatch.setattr(subprocess, "run", _raise)

    result = run_ruff(["anything.py"])

    assert not result.success
    assert result.error.code == ErrorCode.UPSTREAM_ERROR
    assert not result.error.retriable


def test_run_ruff_timeout(monkeypatch):
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ruff", timeout=30.0)

    monkeypatch.setattr(subprocess, "run", _raise)

    result = run_ruff(["anything.py"], timeout=30.0)

    assert not result.success
    assert result.error.code == ErrorCode.TIMEOUT
    assert result.error.retriable


def test_run_ruff_malformed_json_output(monkeypatch):
    fake_completed = subprocess.CompletedProcess(args=["ruff"], returncode=0, stdout="not json", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake_completed)

    result = run_ruff(["anything.py"])

    assert not result.success
    assert result.error.code == ErrorCode.PARSE_ERROR
