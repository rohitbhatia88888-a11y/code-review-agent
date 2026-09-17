"""Eval runner: replay every golden-set case through the real decision
loop, score the result against ground truth, and write per-case plus
aggregate results to ``results/runs/<run_id>/``.

Static analysis always runs for real (local, deterministic, free). The
style checker defaults to a real Anthropic-backed call so cost and
latency numbers mean something; tests inject a fake instead so the
suite never spends money or touches the network. GitHub is always
faked -- eval never posts anywhere.
"""

from __future__ import annotations

import csv
import difflib
import json
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.agent.decision_loop import DEFAULT_MAX_DIFF_LINES, review_pull_request
from src.agent.models import TriagedFinding
from src.agent.tracer import DecisionTracer
from src.eval.golden_set import is_frozen, load_golden_set, verify_integrity
from src.eval.metrics import AggregateMetrics, CaseMatch, aggregate, score_case
from src.eval.models import GoldenCase
from src.tools.models import ChangedFile, Finding, PostedComment, Result, StyleViolation
from src.tools.static_analyzer import run_ruff
from src.tools.style_checker import DEFAULT_MODEL, check_style

EVAL_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "fixtures"
RUFF_CONFIG = EVAL_FIXTURES_DIR / "ruff.toml"

# $/1M tokens. Source: current Anthropic API pricing at the time this
# runner was written. Add a model here before using it, rather than
# silently reporting $0 or guessing.
PRICING_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

StaticAnalyzer = Callable[..., Result[list[Finding]]]
StyleCheck = Callable[[str, str], Result[list[StyleViolation]]]
StyleCheckFactory = Callable[[], tuple[StyleCheck, list["_LLMCallRecord"]]]


@dataclass
class _LLMCallRecord:
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class _NullGitHubClient:
    """Serves one case's changed files and captures posted comments
    without ever touching the network -- eval never posts to real GitHub.
    """

    def __init__(self, changed_files: list[ChangedFile]) -> None:
        self._changed_files = changed_files
        self.inline_comments: list[dict] = []
        self.summary_comments: list[str] = []

    def fetch_changed_files(self, owner, repo, pr_number):
        return Result[list[ChangedFile]].ok(self._changed_files)

    def post_inline_comment(self, owner, repo, pr_number, *, commit_sha, file, line, body):
        self.inline_comments.append({"file": file, "line": line, "body": body})
        return Result[PostedComment].ok(PostedComment(comment_id=len(self.inline_comments), url="eval://inline"))

    def post_summary_comment(self, owner, repo, pr_number, *, body):
        self.summary_comments.append(body)
        return Result[PostedComment].ok(PostedComment(comment_id=len(self.summary_comments), url="eval://summary"))


def default_style_check_factory(model: str = DEFAULT_MODEL) -> StyleCheckFactory:
    """Build a factory of real, Anthropic-backed style-check callables,
    one usage log per case, so cost/latency are attributable per PR.
    Constructing the real client is deferred into the factory so tests
    never import/construct ``anthropic.Anthropic`` unless they choose to.
    """
    import anthropic

    def factory() -> tuple[StyleCheck, list[_LLMCallRecord]]:
        records: list[_LLMCallRecord] = []
        real_client = anthropic.Anthropic()
        wrapped = _UsageTrackingClient(real_client, records)

        def run_style_check(diff: str, style_guide: str) -> Result[list[StyleViolation]]:
            return check_style(diff, style_guide, client=wrapped, model=model)

        return run_style_check, records

    return factory


class _UsageTrackingClient:
    """Wraps an Anthropic client's ``.messages`` so every call's token
    usage and wall-clock latency lands in ``records``, without check_style
    itself needing to know eval is measuring it.
    """

    def __init__(self, real_client, records: list[_LLMCallRecord]) -> None:
        self.messages = _UsageTrackingMessages(real_client.messages, records)


class _UsageTrackingMessages:
    def __init__(self, real_messages, records: list[_LLMCallRecord]) -> None:
        self._real_messages = real_messages
        self._records = records

    def create(self, **kwargs):
        start = time.monotonic()
        response = self._real_messages.create(**kwargs)
        elapsed = time.monotonic() - start
        usage = getattr(response, "usage", None)
        self._records.append(
            _LLMCallRecord(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                latency_seconds=elapsed,
            )
        )
        return response


def _cost_dollars(records: list[_LLMCallRecord], model: str) -> float:
    if not records:
        return 0.0
    pricing = PRICING_PER_MILLION_TOKENS.get(model)
    if pricing is None:
        raise ValueError(f"no pricing configured for model {model!r}; add it to PRICING_PER_MILLION_TOKENS")
    input_cost = sum(r.input_tokens for r in records) / 1_000_000 * pricing["input"]
    output_cost = sum(r.output_tokens for r in records) / 1_000_000 * pricing["output"]
    return input_cost + output_cost


def _materialize_case(case: GoldenCase, tmp_dir: Path) -> list[ChangedFile]:
    if RUFF_CONFIG.exists():
        shutil.copy(RUFF_CONFIG, tmp_dir / "ruff.toml")

    changed_files: list[ChangedFile] = []
    for pr_file in case.files:
        path = tmp_dir / pr_file.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pr_file.content_after)

        before_lines = (pr_file.content_before or "").splitlines(keepends=True)
        after_lines = pr_file.content_after.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(before_lines, after_lines, fromfile=f"a/{pr_file.filename}", tofile=f"b/{pr_file.filename}")
        )
        patch = "".join(diff_lines) or None
        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

        changed_files.append(
            ChangedFile(
                filename=pr_file.filename,
                status="added" if pr_file.content_before is None else "modified",
                additions=additions,
                deletions=deletions,
                patch=patch,
            )
        )
    return changed_files


@dataclass
class CaseRunResult:
    case_id: str
    category: str
    file_statuses: dict[str, str]
    kept_findings: list[TriagedFinding]
    match: CaseMatch
    latency_seconds: float
    cost_dollars: float
    llm_calls: int
    trace_path: str
    error: str | None = field(default=None)


def run_case(
    case: GoldenCase,
    style_guide: str,
    *,
    run_static_analysis: StaticAnalyzer,
    style_check_factory: StyleCheckFactory,
    model: str,
    trace_dir: Path,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    sleep: Callable[[float], None] = lambda s: None,
) -> CaseRunResult:
    run_style_check, records = style_check_factory()

    with tempfile.TemporaryDirectory(prefix=f"eval-{case.case_id}-") as tmp:
        tmp_path = Path(tmp)
        changed_files = _materialize_case(case, tmp_path)
        github = _NullGitHubClient(changed_files)
        tracer = DecisionTracer(trace_dir / f"{case.case_id}.jsonl")

        start = time.monotonic()
        result = review_pull_request(
            "eval",
            "eval-repo",
            1,
            style_guide,
            github=github,
            repo_root=str(tmp_path),
            commit_sha="eval",
            tracer=tracer,
            max_diff_lines=max_diff_lines,
            run_static_analysis=run_static_analysis,
            run_style_check=run_style_check,
            sleep=sleep,
        )
        latency = time.monotonic() - start

    if not result.success:
        return CaseRunResult(
            case_id=case.case_id,
            category=case.category.value,
            file_statuses={},
            kept_findings=[],
            match=score_case(case, []),
            latency_seconds=latency,
            cost_dollars=_cost_dollars(records, model),
            llm_calls=len(records),
            trace_path=str(tracer.path),
            error=f"decision loop couldn't even fetch changed files: {result.error.message}",
        )

    outcome = result.payload
    kept = [f for file_summary in outcome.files for f in file_summary.findings if f.action.value == "kept"]

    return CaseRunResult(
        case_id=case.case_id,
        category=case.category.value,
        file_statuses={f.filename: f.status.value for f in outcome.files},
        kept_findings=kept,
        match=score_case(case, kept),
        latency_seconds=latency,
        cost_dollars=_cost_dollars(records, model),
        llm_calls=len(records),
        trace_path=str(tracer.path),
    )


def run_eval(
    golden_set_path: str | Path,
    *,
    run_id: str | None = None,
    results_dir: str | Path = "results/runs",
    run_static_analysis: StaticAnalyzer = run_ruff,
    style_check_factory: StyleCheckFactory | None = None,
    model: str = DEFAULT_MODEL,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    sleep: Callable[[float], None] = lambda s: None,
    require_frozen: bool = False,
) -> AggregateMetrics:
    """Run every case in the golden set through the agent and write
    results to ``results_dir/<run_id>/``: one JSON file per case plus
    ``summary.json`` for the aggregate. Also appends one row to
    ``results/eval_runs.csv`` for cross-run comparison.

    ``require_frozen=True`` refuses to run unless ``golden_set_path`` has
    a recorded hash that still matches its content -- use this for any
    run whose numbers you intend to report, so a candidate set that
    hasn't been through human review can't be mistaken for the real
    thing.
    """
    golden_set_path = Path(golden_set_path)
    if require_frozen:
        verify_integrity(golden_set_path)
    golden_set = load_golden_set(golden_set_path)

    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(results_dir) / run_id
    cases_dir = run_dir / "cases"
    trace_dir = run_dir / "traces"
    cases_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    factory = style_check_factory or default_style_check_factory(model)

    case_results: list[CaseRunResult] = []
    for case in golden_set.cases:
        case_result = run_case(
            case,
            golden_set.style_guide,
            run_static_analysis=run_static_analysis,
            style_check_factory=factory,
            model=model,
            trace_dir=trace_dir,
            max_diff_lines=max_diff_lines,
            sleep=sleep,
        )
        case_results.append(case_result)
        _write_case_result(cases_dir / f"{case.case_id}.json", case_result)

    metrics = aggregate([r.match for r in case_results])
    total_cost = sum(r.cost_dollars for r in case_results)
    total_latency = sum(r.latency_seconds for r in case_results)
    n = len(case_results) or 1

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "golden_set_path": str(golden_set_path),
        "golden_set_frozen": is_frozen(golden_set_path),
        "model": model,
        "case_count": len(case_results),
        "total_cost_dollars": total_cost,
        "avg_cost_dollars_per_pr": total_cost / n,
        "total_latency_seconds": total_latency,
        "avg_latency_seconds_per_pr": total_latency / n,
        "metrics": metrics.model_dump(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _append_csv_row(Path(results_dir).parent / "eval_runs.csv", summary)
    return metrics


def _write_case_result(path: Path, result: CaseRunResult) -> None:
    payload = {
        "case_id": result.case_id,
        "category": result.category,
        "file_statuses": result.file_statuses,
        "kept_findings": [f.model_dump() for f in result.kept_findings],
        "matched_required": [f.model_dump() for f in result.match.matched_required],
        "matched_optional": [f.model_dump() for f in result.match.matched_optional],
        "false_positives": [f.model_dump() for f in result.match.false_positives],
        "missed_required": [e.model_dump() for e in result.match.missed_required],
        "missed_optional": [e.model_dump() for e in result.match.missed_optional],
        "latency_seconds": result.latency_seconds,
        "cost_dollars": result.cost_dollars,
        "llm_calls": result.llm_calls,
        "trace_path": result.trace_path,
        "error": result.error,
    }
    path.write_text(json.dumps(payload, indent=2))


def _append_csv_row(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary["metrics"]
    row = {
        "run_id": summary["run_id"],
        "generated_at": summary["generated_at"],
        "golden_set_frozen": summary["golden_set_frozen"],
        "model": summary["model"],
        "case_count": summary["case_count"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "pr_false_positive_rate": metrics["pr_false_positive_rate"],
        "avg_cost_dollars_per_pr": summary["avg_cost_dollars_per_pr"],
        "avg_latency_seconds_per_pr": summary["avg_latency_seconds_per_pr"],
    }
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
