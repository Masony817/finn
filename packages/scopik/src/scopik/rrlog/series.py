"""Scalar time-series logging: batched for replay, per-frame for live."""

from __future__ import annotations

import numpy as np
import rerun as rr

from scopik.datamodel import RunData, Signal
from scopik.metrics import PairResult
from scopik.rrlog import (
    DEFAULT_REAL_COLOR,
    DEFAULT_SIM_COLOR,
    RESIDUAL_COLOR,
    ROLLING_COLOR,
    TIMELINE,
)


def log_series(
    entity_path: str,
    times: np.ndarray,
    values: np.ndarray,
    display_name: str,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Batch-log one scalar series (one send_columns call, not one log per row)."""

    finite = np.isfinite(times) & np.isfinite(values)
    if not finite.any():
        return
    rr.send_columns(
        entity_path,
        indexes=[rr.TimeColumn(TIMELINE, duration=times[finite])],
        columns=rr.Scalars.columns(scalars=values[finite]),
    )
    rr.log(
        entity_path,
        rr.SeriesLines(names=display_name, colors=color),
        static=True,
    )


def log_signal_point(entity_path: str, time_s: float, value: float) -> None:
    """Per-frame path used by live streaming; same entities as log_series."""

    rr.set_time(TIMELINE, duration=time_s)
    rr.log(entity_path, rr.Scalars(value))


def signal_entity(prefix: str, signal: Signal) -> str:
    return f"{prefix}/signals/{signal.group}/{signal.name}"


def log_run_signals(run: RunData, prefix: str, color: tuple[int, int, int] | None = None) -> None:
    for signal in run.signals.values():
        display = f"{signal.name} [{signal.unit}]" if signal.unit else signal.name
        log_series(signal_entity(prefix, signal), signal.times, signal.values, display, color)


def log_pair(
    pair: PairResult,
    base: str = "/compare",
    real_color: tuple[int, int, int] = DEFAULT_REAL_COLOR,
    sim_color: tuple[int, int, int] = DEFAULT_SIM_COLOR,
) -> None:
    """Log one compare pair: real+sim overlay series, residual, rolling RMSE."""

    root = f"{base}/{pair.name}"
    unit = f" [{pair.unit}]" if pair.unit else ""
    log_series(f"{root}/real", pair.times, pair.real, f"real{unit}", real_color)
    log_series(f"{root}/sim", pair.times, pair.sim, f"sim{unit}", sim_color)
    log_series(f"{root}/residual", pair.times, pair.residual, f"sim - real{unit}", RESIDUAL_COLOR)
    log_series(
        f"{root}/rolling_rmse", pair.times, pair.rolling_rmse, f"rolling RMSE{unit}", ROLLING_COLOR
    )


def log_summary_document(markdown: str, entity_path: str = "/summary") -> None:
    rr.log(
        entity_path,
        rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )
