"""Gap history: one JSON line per gap run, for regression trendlines.

A single scalar RMSE hides progress; a file of them across model iterations
shows whether the sim-to-real gap is actually closing. Ten lines of infra,
the strongest habit in the tool.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from scopik.metrics import PairResult
from scopik.profile import Profile


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def default_history_path(profile: Profile) -> Path:
    return profile.path.parent / f"{profile.name}_gap_history.jsonl"


def build_record(
    profile: Profile, run_dir: Path, model_path: Path, results: list[PairResult]
) -> dict[str, object]:
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "profile": profile.name,
        "run_dir": str(run_dir),
        "model": str(model_path),
        "model_sha256_12": file_hash(model_path),
        "metrics": {
            result.name: {
                "rmse": result.summary["rmse"],
                "mae": result.summary["mae"],
                "sample_count": result.summary["sample_count"],
            }
            for result in results
        },
    }


def append_history(history_path: Path, record: dict[str, object]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def read_history(history_path: Path) -> list[dict[str, object]]:
    if not history_path.exists():
        return []
    records = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
