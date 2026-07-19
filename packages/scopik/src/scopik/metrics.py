"""Sim-vs-real comparison: resampling, residuals, rolling RMSE, summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scopik.datamodel import RunData, ScopikError, Signal, TimeBase
from scopik.profile import ComparePair


@dataclass(frozen=True)
class PairResult:
    name: str
    unit: str
    times: np.ndarray
    real: np.ndarray
    sim: np.ndarray
    residual: np.ndarray
    rolling_rmse: np.ndarray
    summary: dict[str, float]


def resample_signal(signal: Signal, timebase: TimeBase) -> np.ndarray:
    valid = np.isfinite(signal.values) & np.isfinite(signal.times)
    if valid.sum() < 2:
        raise ScopikError(f"signal {signal.name!r} has fewer than 2 finite samples")
    return np.interp(timebase.times, signal.times[valid], signal.values[valid])


def rolling_rmse(residual: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray:
    """RMSE over a trailing time window, O(n) via cumulative sums."""

    if len(residual) == 0:
        return residual
    squared = residual * residual
    cumsum = np.concatenate(([0.0], np.cumsum(squared)))
    # For each i, find the first index j with times[j] > times[i] - window_s.
    starts = np.searchsorted(times, times - window_s, side="right")
    starts = np.minimum(starts, np.arange(len(times)))  # window always includes self
    counts = np.arange(1, len(times) + 1) - starts
    sums = cumsum[1:] - cumsum[starts]
    return np.sqrt(sums / np.maximum(counts, 1))


def summarize(real: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    err = sim - real
    finite = np.isfinite(err)
    err = err[finite]
    if len(err) == 0:
        raise ScopikError("no finite overlapping samples to compare")
    return {
        "sample_count": len(err),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "real_mean": float(np.mean(real[finite])),
        "sim_mean": float(np.mean(sim[finite])),
    }


def compare_runs(
    real_run: RunData,
    sim_run: RunData,
    pairs: tuple[ComparePair, ...],
    window_s: float = 1.0,
) -> list[PairResult]:
    """Compute per-pair residuals and summaries on a shared timebase.

    When both signals already share identical timestamps (command replay), the
    shared timebase is exactly those timestamps and interpolation is a no-op.
    """

    results: list[PairResult] = []
    for pair in pairs:
        real_signal = real_run.signals.get(pair.real)
        sim_signal = sim_run.signals.get(pair.sim)
        if real_signal is None or sim_signal is None:
            continue  # pair not present in this run's schema; skip, don't fail
        if np.array_equal(real_signal.times, sim_signal.times):
            timebase = TimeBase(times=real_signal.times)
            real_values = real_signal.values
            sim_values = sim_signal.values
        else:
            timebase = TimeBase.overlap(real_signal.times, sim_signal.times)
            real_values = resample_signal(real_signal, timebase)
            sim_values = resample_signal(sim_signal, timebase)
        residual = sim_values - real_values
        results.append(
            PairResult(
                name=pair.name,
                unit=pair.unit or real_signal.unit,
                times=timebase.times,
                real=real_values,
                sim=sim_values,
                residual=residual,
                rolling_rmse=rolling_rmse(
                    np.nan_to_num(residual, nan=0.0), timebase.times, window_s
                ),
                summary=summarize(real_values, sim_values),
            )
        )
    return results


def summary_markdown(results: list[PairResult], scope_note: str | None = None) -> str:
    lines = [
        "# Sim vs real",
        "",
        "| signal | unit | RMSE | MAE | max abs err | real mean | sim mean | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        s = result.summary
        lines.append(
            f"| {result.name} | {result.unit} | {s['rmse']:.5g} | {s['mae']:.5g} "
            f"| {s['max_abs']:.5g} | {s['real_mean']:.5g} | {s['sim_mean']:.5g} "
            f"| {s['sample_count']} |"
        )
    if scope_note:
        lines += ["", f"*{scope_note}*"]
    return "\n".join(lines)


def summary_json(results: list[PairResult]) -> dict[str, dict[str, float]]:
    return {result.name: result.summary for result in results}
