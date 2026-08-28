"""Stream named scalar signals to a Rerun viewer while a robot is still running.

Scopik's main job is recorded, open-loop sim-to-real comparison, and live
streaming was deliberately excluded until a real workflow needed it. Driving a
balancing robot by hand is that workflow: a gap report written after the run
cannot tell you what the robot is doing while you steer it.

This stays deliberately smaller than the gap pipeline. There is no profile, no
model, and no comparison -- just named scalars on a shared timeline, so any robot
can point it at whatever it already computes each control tick.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from scopik import rrlog

DEFAULT_APP_ID = "scopik-live"

# Signals sharing a group land under one entity path, so the viewer stacks them
# on the same plot instead of one panel per signal.
DEFAULT_COLORS = (
    (230, 110, 60),
    (90, 140, 255),
    (140, 200, 120),
    (200, 60, 90),
    (170, 140, 220),
    (140, 140, 150),
)


@dataclass(frozen=True)
class LiveSignal:
    """One scalar to stream: where it comes from, and where it should be drawn."""

    column: str
    group: str
    color: tuple[int, int, int] | None = None

    def entity_path(self, root: str) -> str:
        return f"{root}/{self.group}/{self.column}"


@dataclass
class LiveSession:
    """Open a Rerun sink and push one row of named scalars per control tick."""

    signals: tuple[LiveSignal, ...]
    root: str = "/live"
    app_id: str = DEFAULT_APP_ID
    _declared: bool = field(default=False, init=False)

    @classmethod
    def from_columns(cls, grouped: Mapping[str, Iterable[str]], **kwargs) -> LiveSession:
        signals: list[LiveSignal] = []
        for group, columns in grouped.items():
            for index, column in enumerate(columns):
                signals.append(
                    LiveSignal(column, group, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
                )
        return cls(tuple(signals), **kwargs)

    def spawn(self) -> None:
        """Open a viewer window and stream into it."""

        rrlog.init(self.app_id)
        rrlog.sink_spawn()
        self._declare()

    def save(self, path: Path) -> None:
        """Stream into a recording instead of a window, for tests and CI."""

        rrlog.init(self.app_id)
        rrlog.sink_save(path)
        self._declare()

    def _declare(self) -> None:
        for signal in self.signals:
            rrlog.declare_series(
                signal.entity_path(self.root),
                signal.column,
                signal.color or DEFAULT_COLORS[0],
            )
        self._declared = True

    def log_row(self, time_s: float, row: Mapping[str, float]) -> None:
        """Push whichever declared signals this row carries.

        Missing columns are skipped rather than raising: a live sink must never be
        able to interrupt the control loop feeding it.
        """

        if not self._declared:
            raise RuntimeError("call LiveSession.spawn() or .save() before log_row()")
        rrlog.set_time(time_s)
        for signal in self.signals:
            value = row.get(signal.column)
            if value is None:
                continue
            rrlog.log_scalar(signal.entity_path(self.root), float(value))

    def close(self) -> None:
        rrlog.disconnect()
