"""Append-only decision trace: one JSON line per triage/agent decision, so
a review run's reasoning -- what was considered, what was kept or
dropped, and why -- is fully auditable after the fact, not just visible
in the final output.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.agent.models import TraceEntry


class DecisionTracer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[TraceEntry] = []

    def record(
        self,
        *,
        stage: str,
        action: str,
        reason: str,
        file: str | None = None,
        line: int | None = None,
    ) -> None:
        entry = TraceEntry(
            timestamp=datetime.now(UTC),
            stage=stage,
            action=action,
            reason=reason,
            file=file,
            line=line,
        )
        self.entries.append(entry)
        with self.path.open("a") as trace_file:
            trace_file.write(entry.model_dump_json() + "\n")
