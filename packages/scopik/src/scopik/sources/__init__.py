"""Telemetry sources: turn a run directory into a RunData.

A Source parses one file format into raw columns. Column -> signal mapping and
unit transforms are applied uniformly here, driven by the profile, so
individual sources only have to produce named columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from scopik.datamodel import RunData, ScopikError, Signal
from scopik.profile import Profile
from scopik.transforms import apply_transform


@runtime_checkable
class Source(Protocol):
    """Parses one log format into raw named columns."""

    def read_table(
        self, path: Path, profile: Profile
    ) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
        """Return (numeric columns, text columns).

        Numeric columns cover every column (unparseable cells are NaN) so
        callers never miss one; text columns additionally keep the raw strings
        for columns that are mostly non-numeric (states, phases, fault names).
        """
        ...


def split_text_columns(
    names: list[str], rows: list[list[str]]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Shared column-typing rule for CSV-shaped sources."""

    matrix = np.full((len(rows), len(names)), np.nan)
    text: dict[str, list[str]] = {}
    for j, _name in enumerate(names):
        for i, row in enumerate(rows):
            try:
                matrix[i, j] = float(row[j])
            except (TypeError, ValueError):
                continue
    numeric = {name: matrix[:, j] for j, name in enumerate(names)}
    for j, name in enumerate(names):
        column = matrix[:, j]
        if len(column) and np.isnan(column).mean() > 0.5:
            text[name] = [row[j] for row in rows]
    return numeric, text


def build_run_data(
    label: str,
    columns: dict[str, np.ndarray],
    profile: Profile,
    text_columns: dict[str, list[str]] | None = None,
) -> RunData:
    """Apply the profile's time and signal mappings to raw columns."""

    time_column = profile.time.column
    if time_column not in columns:
        available = ", ".join(sorted(columns)) or "<none>"
        raise ScopikError(
            f"time column {time_column!r} not found in {label} data (has: {available})"
        )
    times = apply_transform(columns[time_column], profile.time.transform)

    run = RunData(label=label, columns=columns, text_columns=text_columns or {}, times=times)
    for spec in profile.signals:
        if spec.column not in columns:
            continue  # profiles may describe a superset of columns (schema versions differ)
        run.signals[spec.name] = Signal(
            name=spec.name,
            unit=spec.unit,
            times=times,
            values=apply_transform(columns[spec.column], spec.transform),
            group=spec.group,
        )
    return run


def load_run(run_dir: Path, profile: Profile, label: str = "real") -> RunData:
    """Load a run directory using the profile's declared source."""

    from scopik.plugins import resolve_source

    source = resolve_source(profile.source.type)
    data_path = run_dir / profile.source.file
    if run_dir.is_file():
        data_path = run_dir  # allow pointing straight at the file
    if not data_path.exists():
        raise ScopikError(f"missing telemetry file: {data_path}")
    columns, text_columns = source.read_table(data_path, profile)
    run = build_run_data(label, columns, profile, text_columns)
    run.meta["run_dir"] = str(run_dir)
    run.meta["source_file"] = str(data_path)
    return run


def load_events(run_dir: Path, profile: Profile) -> list[tuple[float, str]]:
    """Parse the run's event log into (time_s, message) pairs, if configured."""

    events_profile = profile.events
    if events_profile is None or run_dir.is_file():
        return []
    path = run_dir / events_profile.file
    if not path.exists():
        return []

    events: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(events_profile.prefix):
            continue
        cells = line.split(",")
        if len(cells) <= events_profile.time_index:
            continue
        try:
            time_raw = float(cells[events_profile.time_index])
        except ValueError:
            continue
        time_s = float(apply_transform(np.array([time_raw]), events_profile.time_transform)[0])
        message = ",".join(cells[events_profile.time_index + 1 :]).strip() or line
        events.append((time_s, message))
    return events


def phase_changes(run: RunData, column: str) -> list[tuple[float, str]]:
    """Return (time_s, label) at every value change of a text column."""

    values = run.text_columns.get(column)
    if values is None:
        return []
    times = run.require_times()
    changes: list[tuple[float, str]] = []
    previous: str | None = None
    for i, value in enumerate(values):
        if value != previous:
            changes.append((float(times[i]), f"{column}: {value}"))
            previous = value
    return changes
