"""Loading, hashing, and freezing the golden set.

"Frozen" means the file's content matches a recorded SHA-256 hash.
Freezing itself refuses to run unless every case has been marked
``reviewed`` -- an unreviewed set can be loaded and even eval'd against
(useful for sanity-checking a draft), but it can't be promoted to the
frozen golden set, and the eval runner refuses to score against a
frozen file whose content no longer matches its recorded hash. Nothing
here can edit a case to improve a score; this module only reads,
hashes, and locks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.models import GoldenSet

HASH_SUFFIX = ".sha256"


def load_golden_set(path: str | Path) -> GoldenSet:
    data = json.loads(Path(path).read_text())
    return GoldenSet.model_validate(data)


def save_golden_set(golden_set: GoldenSet, path: str | Path) -> None:
    Path(path).write_text(_canonical_json(golden_set) + "\n")


def compute_hash(golden_set: GoldenSet) -> str:
    return hashlib.sha256(_canonical_json(golden_set).encode("utf-8")).hexdigest()


def _canonical_json(golden_set: GoldenSet) -> str:
    return json.dumps(golden_set.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def freeze(candidate_path: str | Path, frozen_path: str | Path) -> str:
    """Promote a reviewed candidate set to the frozen golden set.

    Refuses if any case hasn't been marked ``reviewed`` -- freezing an
    unreviewed set would defeat the point of "human-reviewed ground
    truth". Writes the frozen JSON plus a sidecar ``.sha256`` hash
    file; ``verify_integrity`` checks the frozen file against that hash
    on every eval run so a later silent edit can't sneak through.
    Returns the recorded hash.
    """
    golden_set = load_golden_set(candidate_path)
    unreviewed = [c.case_id for c in golden_set.cases if not c.reviewed]
    if unreviewed:
        raise ValueError(f"cannot freeze: {len(unreviewed)} case(s) not yet reviewed: {unreviewed}")

    frozen_path = Path(frozen_path)
    save_golden_set(golden_set, frozen_path)
    digest = compute_hash(golden_set)
    _hash_path(frozen_path).write_text(digest + "\n")
    return digest


def verify_integrity(path: str | Path) -> None:
    """Raise if a frozen golden set's content no longer matches its
    recorded hash. Raises ``FileNotFoundError`` if the set was never
    frozen (no ``.sha256`` sidecar) -- that's a distinct, expected state
    for a candidate set, not corruption.
    """
    path = Path(path)
    hash_path = _hash_path(path)
    if not hash_path.exists():
        raise FileNotFoundError(f"no recorded hash at {hash_path}; this golden set has not been frozen")

    recorded = hash_path.read_text().strip()
    actual = compute_hash(load_golden_set(path))
    if actual != recorded:
        raise ValueError(
            f"golden set at {path} does not match its recorded hash "
            f"(recorded {recorded}, actual {actual}) -- it was edited after freezing"
        )


def is_frozen(path: str | Path) -> bool:
    return _hash_path(path).exists()


def _hash_path(path: str | Path) -> Path:
    return Path(str(path) + HASH_SUFFIX)
