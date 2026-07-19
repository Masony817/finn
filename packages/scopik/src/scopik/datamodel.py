"""Core data types shared by sources, replay, metrics, and logging.

Everything is plain numpy: a Signal is one named time series and RunData is one
recorded or simulated run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class ScopikError(Exception):
    """Expected failure with a concise user-facing message."""


@dataclass
class Signal:
    """One named time series with times in seconds."""

    name: str
    unit: str
    times: np.ndarray
    values: np.ndarray
    group: str = "signals"

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.times.shape != self.values.shape:
            raise ScopikError(
                f"signal {self.name!r}: times shape {self.times.shape} != "
                f"values shape {self.values.shape}"
            )


@dataclass
class RunData:
    """One run: a label ("real"/"sim"), signals by name, and metadata."""

    label: str
    signals: dict[str, Signal] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)
    # Raw columns survive so replay can use command fields that were not
    # promoted to named display signals.
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    # Mostly-non-numeric columns (state machine names, phase labels, fault
    # strings) keep their text form so they can become timeline annotations.
    text_columns: dict[str, list[str]] = field(default_factory=dict)
    times: np.ndarray | None = None

    def require_column(self, name: str) -> np.ndarray:
        if name not in self.columns:
            available = ", ".join(sorted(self.columns)) or "<none>"
            raise ScopikError(f"run {self.label!r} has no column {name!r} (has: {available})")
        return self.columns[name]

    def require_times(self) -> np.ndarray:
        if self.times is None or len(self.times) == 0:
            raise ScopikError(f"run {self.label!r} has no time base")
        return self.times


@dataclass(frozen=True)
class TimeBase:
    """A shared uniform grid two runs can be resampled onto."""

    times: np.ndarray

    @staticmethod
    def overlap(a_times: np.ndarray, b_times: np.ndarray, dt_s: float | None = None) -> TimeBase:
        start = max(float(a_times[0]), float(b_times[0]))
        end = min(float(a_times[-1]), float(b_times[-1]))
        if end <= start:
            raise ScopikError(
                f"runs do not overlap in time: [{a_times[0]:.3f}, {a_times[-1]:.3f}] vs "
                f"[{b_times[0]:.3f}, {b_times[-1]:.3f}]"
            )
        if dt_s is None:
            diffs = np.diff(a_times)
            positive = diffs[diffs > 0]
            if len(positive) == 0:
                raise ScopikError("cannot infer timebase dt: no increasing timestamps")
            dt_s = float(np.median(positive))
        # Epsilon guards against float truncation (e.g. 1.18/0.01 -> 117.9999...).
        count = int((end - start) / dt_s + 1e-9) + 1
        return TimeBase(times=start + dt_s * np.arange(count))
