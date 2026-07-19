"""Per-phase diagnostic analysis: turn residuals into verdicts.

Sysid and test runs are scripted sequences of phases, and each phase isolates
different physics. Slicing the residuals by phase and extracting a few
deterministic features (bias, RMS, gain, lag) converts "the RMSE is 0.44"
into "the error is concentrated in loaded phases and is bias-dominated" -
which is the shape of answer a human (or an LLM reviewing the run) can act on.

Everything here is deterministic numpy; no thresholds are learned. The
mapping from patterns to *parameter* suspects is deliberately not in scopik:
that knowledge is robot-specific and lives with the robot (e.g. a review
checklist or agent skill in the robot's repo).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from scopik.datamodel import RunData
from scopik.metrics import PairResult

MIN_PHASE_SAMPLES = 20
LAG_WINDOW_S = 0.25
LAG_SIGNIFICANT_S = 0.015
GAIN_BAND = (0.85, 1.2)
CORRELATION_FLOOR = 0.6
CLEAN_RELATIVE_RMSE = 0.1
QUIET_RELATIVE_LEVEL = 0.1
MIN_PHASE_ACTIVITY_FRACTION = 0.02


@dataclass(frozen=True)
class PhaseSegment:
    label: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class PhaseStats:
    """Residual statistics for one compare pair within one phase."""

    pair: str
    phase: str
    unit: str
    start_s: float
    end_s: float
    sample_count: int
    bias: float  # mean(sim - real): systematic offset
    rms: float
    real_mean: float
    real_std: float  # how active the real signal was in this phase
    real_rms: float  # distinguishes stationary zero from steady nonzero motion
    gain: float | None
    correlation: float | None
    lag_s: float | None


@dataclass(frozen=True)
class Finding:
    severity: str  # "issue" | "note" | "ok"
    kind: str  # concentration | static_offset | lag | gain | bias | clean | unclassified
    pair: str
    phase: str | None
    value: float
    message: str


def phase_segments(run: RunData, column: str) -> list[PhaseSegment]:
    """Merge consecutive identical phase labels into time segments."""

    values = run.text_columns.get(column)
    if not values:
        return []
    times = run.require_times()
    segments: list[PhaseSegment] = []
    start_index = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start_index]:
            segments.append(
                PhaseSegment(
                    label=values[start_index],
                    start_s=float(times[start_index]),
                    # end_s is inclusive. Using times[i] here would duplicate
                    # the next phase's first sample in both segments.
                    end_s=float(times[i - 1]),
                )
            )
            start_index = i
    return segments


def phase_stats(
    pair: PairResult,
    segments: list[PhaseSegment],
    *,
    min_real_std: float = 0.0,
) -> list[PhaseStats]:
    stats: list[PhaseStats] = []
    for segment in segments:
        lo = int(np.searchsorted(pair.times, segment.start_s, side="left"))
        hi = int(np.searchsorted(pair.times, segment.end_s, side="right"))
        times = pair.times[lo:hi]
        real = pair.real[lo:hi]
        sim = pair.sim[lo:hi]
        residual = pair.residual[lo:hi]
        finite = np.isfinite(times) & np.isfinite(real) & np.isfinite(sim) & np.isfinite(residual)
        if finite.sum() < MIN_PHASE_SAMPLES:
            continue
        times = times[finite]
        real = real[finite]
        sim = sim[finite]
        residual = residual[finite]
        real_mean = float(np.mean(real))
        real_std = float(np.std(real))
        real_rms = float(np.sqrt(np.mean(real * real)))
        gain: float | None = None
        correlation: float | None = None
        lag_s = None
        if real_std >= max(min_real_std, 1e-9):
            gain, correlation = _estimate_gain(real, sim)
        if correlation is not None and abs(correlation) >= CORRELATION_FLOOR:
            lag_s = _estimate_lag_s(times, real, sim)
        stats.append(
            PhaseStats(
                pair=pair.name,
                phase=segment.label,
                unit=pair.unit,
                start_s=segment.start_s,
                end_s=segment.end_s,
                sample_count=len(residual),
                bias=float(np.mean(residual)),
                rms=float(np.sqrt(np.mean(residual * residual))),
                real_mean=real_mean,
                real_std=real_std,
                real_rms=real_rms,
                gain=gain,
                correlation=correlation,
                lag_s=lag_s,
            )
        )
    return stats


def _estimate_gain(
    real_values: np.ndarray, sim_values: np.ndarray
) -> tuple[float | None, float | None]:
    real = real_values - np.mean(real_values)
    sim = sim_values - np.mean(sim_values)
    denom = float(np.dot(real, real))
    sim_denom = float(np.dot(sim, sim))
    if denom < 1e-12 or sim_denom < 1e-12:
        return None, None
    gain = float(np.dot(real, sim) / denom)
    correlation = float(np.dot(real, sim) / np.sqrt(denom * sim_denom))
    return gain, correlation


def estimate_gain(pair: PairResult) -> tuple[float | None, float | None]:
    """Least-squares gain sim ~= g * real, with correlation coefficient."""

    finite = np.isfinite(pair.real) & np.isfinite(pair.sim)
    if finite.sum() < 3:
        return None, None
    return _estimate_gain(pair.real[finite], pair.sim[finite])


def _estimate_lag_s(
    times: np.ndarray,
    real_values: np.ndarray,
    sim_values: np.ndarray,
    window_s: float = LAG_WINDOW_S,
) -> float | None:
    """Estimate lag using normalized correlation on a uniform time grid."""

    if len(times) < 3:
        return None
    positive = np.diff(times)
    positive = positive[positive > 0]
    if len(positive) == 0:
        return None
    dt_s = float(np.median(positive))
    grid = np.arange(float(times[0]), float(times[-1]) + 0.5 * dt_s, dt_s)
    if len(grid) < 3:
        return None
    real = np.interp(grid, times, real_values)
    sim = np.interp(grid, times, sim_values)
    max_shift = min(max(1, int(window_s / dt_s)), len(grid) - 3)
    best_shift, best_score = 0, -np.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            real_overlap = real[: len(real) - shift or None]
            sim_overlap = sim[shift:]
        else:
            real_overlap = real[-shift:]
            sim_overlap = sim[: len(sim) + shift]
        real_overlap = real_overlap - np.mean(real_overlap)
        sim_overlap = sim_overlap - np.mean(sim_overlap)
        denom = float(np.linalg.norm(real_overlap) * np.linalg.norm(sim_overlap))
        if denom < 1e-12:
            continue
        score = float(np.dot(real_overlap, sim_overlap) / denom)
        if score > best_score:
            best_score, best_shift = score, shift
    # A peak at the search boundary means the data does not identify a lag
    # inside the requested window. Reporting the boundary as a measurement is
    # misleading, especially for monotonic or nearly constant phase signals.
    if abs(best_shift) == max_shift:
        return None
    return best_shift * dt_s


def estimate_lag_s(pair: PairResult, window_s: float = LAG_WINDOW_S) -> float:
    """Positive lag: sim responds LATER than real. Cross-correlation argmax."""

    finite = np.isfinite(pair.times) & np.isfinite(pair.real) & np.isfinite(pair.sim)
    times = pair.times[finite]
    real = pair.real[finite]
    sim = pair.sim[finite]
    if len(real) < 3 or np.std(real) < 1e-9 or np.std(sim) < 1e-9:
        return 0.0
    return _estimate_lag_s(times, real, sim, window_s) or 0.0


def analyze(
    results: list[PairResult],
    segments: list[PhaseSegment],
) -> tuple[list[PhaseStats], list[Finding]]:
    """Compute per-phase stats and pattern findings for every compare pair."""

    all_stats: list[PhaseStats] = []
    findings: list[Finding] = []

    for pair in results:
        pair_findings: list[Finding] = []
        finite_real = pair.real[np.isfinite(pair.real)]
        scale = float(np.std(finite_real)) if len(finite_real) else 0.0
        signal_rms = float(np.sqrt(np.mean(finite_real * finite_real))) if len(finite_real) else 0.0
        overall_rms = pair.summary["rmse"]
        overall_bias = pair.summary["sim_mean"] - pair.summary["real_mean"]

        stats = phase_stats(
            pair,
            segments,
            min_real_std=MIN_PHASE_ACTIVITY_FRACTION * scale,
        )
        all_stats.extend(stats)

        # Error concentrated in specific phases? Aggregate into one finding
        # per pair (long runs have ~100 segments; per-phase spam helps no one).
        if len(stats) >= 3 and overall_rms > 1e-12:
            rms_values = np.array([s.rms for s in stats])
            median_rms = float(np.median(rms_values))
            threshold = max(2.5 * median_rms, 0.25 * overall_rms)
            hot = sorted((s for s in stats if s.rms > threshold), key=lambda s: -s.rms)
            if hot:
                worst = ", ".join(f"{s.phase} ({s.rms:.3g})" for s in hot[:3])
                pair_findings.append(
                    Finding(
                        severity="issue",
                        kind="concentration",
                        pair=pair.name,
                        phase=hot[0].phase,
                        value=hot[0].rms,
                        message=(
                            f"{pair.name}: error concentrated in {len(hot)} of "
                            f"{len(stats)} phases (median phase rms {median_rms:.3g} "
                            f"{pair.unit}); worst: {worst}"
                        ),
                    )
                )

        # Static offset: phases where the real signal is quiet but sim drifts.
        quiet_drift = [
            s
            for s in stats
            if signal_rms > 1e-12
            and s.real_rms < QUIET_RELATIVE_LEVEL * signal_rms
            and s.real_std < QUIET_RELATIVE_LEVEL * max(scale, signal_rms)
            and abs(s.bias) > 0.15 * max(scale, signal_rms)
        ]
        if quiet_drift:
            worst = max(quiet_drift, key=lambda s: abs(s.bias))
            pair_findings.append(
                Finding(
                    severity="issue",
                    kind="static_offset",
                    pair=pair.name,
                    phase=worst.phase,
                    value=worst.bias,
                    message=(
                        f"{pair.name}: sim drifts while real stays near zero in "
                        f"{len(quiet_drift)} quiet phase(s); worst {worst.bias:+.3g} "
                        f"{pair.unit} during {worst.phase!r}"
                    ),
                )
            )

        # Whole-run bias domination.
        if overall_rms > 1e-12 and abs(overall_bias) > 0.7 * overall_rms:
            pair_findings.append(
                Finding(
                    severity="issue",
                    kind="bias",
                    pair=pair.name,
                    phase=None,
                    value=overall_bias,
                    message=(
                        f"{pair.name}: error is mostly a constant offset "
                        f"({overall_bias:+.4g} {pair.unit}) - check frames, zeroing, "
                        "or a missing constant term"
                    ),
                )
            )

        # Gain and lag are phase-local when phase data exists. This prevents
        # unrelated scripted motions from cancelling each other in one fit.
        phase_gain_issues = [
            s
            for s in stats
            if s.gain is not None
            and s.correlation is not None
            and abs(s.correlation) >= CORRELATION_FLOOR
            and not (GAIN_BAND[0] <= s.gain <= GAIN_BAND[1])
        ]
        if phase_gain_issues:
            worst = sorted(
                phase_gain_issues,
                key=lambda s: abs((s.gain or 1.0) - 1.0),
                reverse=True,
            )
            examples = ", ".join(
                f"{s.phase} ({s.gain:.2f}x)" for s in worst[:3] if s.gain is not None
            )
            pair_findings.append(
                Finding(
                    severity="issue",
                    kind="gain",
                    pair=pair.name,
                    phase=worst[0].phase,
                    value=float(worst[0].gain),
                    message=(
                        f"{pair.name}: gain is outside {GAIN_BAND[0]:.2f}-{GAIN_BAND[1]:.2f}x "
                        f"in {len(worst)} phase(s); worst: {examples}"
                    ),
                )
            )
        phase_lag_issues = [
            s for s in stats if s.lag_s is not None and abs(s.lag_s) >= LAG_SIGNIFICANT_S
        ]
        if phase_lag_issues:
            worst = sorted(phase_lag_issues, key=lambda s: abs(s.lag_s or 0.0), reverse=True)
            examples = ", ".join(
                f"{s.phase} ({(s.lag_s or 0.0) * 1000:+.0f} ms)" for s in worst[:3]
            )
            pair_findings.append(
                Finding(
                    severity="issue",
                    kind="lag",
                    pair=pair.name,
                    phase=worst[0].phase,
                    value=float(worst[0].lag_s),
                    message=(
                        f"{pair.name}: |lag| is at least {LAG_SIGNIFICANT_S * 1000:.0f} ms "
                        f"in {len(worst)} phase(s); positive means sim is later; worst: {examples}"
                    ),
                )
            )

        if not stats:
            gain, correlation = estimate_gain(pair)
            if (
                gain is not None
                and correlation is not None
                and abs(correlation) >= CORRELATION_FLOOR
                and not (GAIN_BAND[0] <= gain <= GAIN_BAND[1])
            ):
                pair_findings.append(
                    Finding(
                        severity="issue",
                        kind="gain",
                        pair=pair.name,
                        phase=None,
                        value=gain,
                        message=(
                            f"{pair.name}: sim responds with gain {gain:.2f}x of real "
                            f"(correlation {correlation:.2f}) - scaling-type mismatch"
                        ),
                    )
                )
            lag_s = estimate_lag_s(pair)
            if (
                correlation is not None
                and abs(correlation) >= CORRELATION_FLOOR
                and abs(lag_s) >= LAG_SIGNIFICANT_S
            ):
                direction = "later" if lag_s > 0 else "earlier"
                pair_findings.append(
                    Finding(
                        severity="issue",
                        kind="lag",
                        pair=pair.name,
                        phase=None,
                        value=lag_s,
                        message=(
                            f"{pair.name}: sim responds {abs(lag_s) * 1000:.0f} ms "
                            f"{direction} than real - timing/delay mismatch"
                        ),
                    )
                )

        if not pair_findings:
            relative_rmse = overall_rms / signal_rms if signal_rms > 1e-12 else np.inf
            is_clean = relative_rmse <= CLEAN_RELATIVE_RMSE
            pair_findings.append(
                Finding(
                    severity="ok" if is_clean else "note",
                    kind="clean" if is_clean else "unclassified",
                    pair=pair.name,
                    phase=None,
                    value=overall_rms,
                    message=(
                        f"{pair.name}: gap is small relative to the real signal "
                        if is_clean
                        else f"{pair.name}: no specific bias/gain/lag pattern identified "
                    )
                    + f"(rmse {overall_rms:.4g} {pair.unit})",
                )
            )
        findings.extend(pair_findings)

    severity_order = {"issue": 0, "note": 1, "ok": 2}
    findings.sort(key=lambda f: (severity_order[f.severity], f.pair))
    return all_stats, findings


def stats_markdown(stats: list[PhaseStats], findings: list[Finding]) -> str:
    lines = ["# Diagnosis", ""]
    if findings:
        lines.append("## Findings")
        lines.append("")
        for finding in findings:
            marker = {"issue": "⚠", "note": "•", "ok": "✓"}[finding.severity]
            lines.append(f"- {marker} {finding.message}")
        lines.append("")
    if stats:
        lines += [
            "## Per-phase residuals",
            "",
            "| pair | phase | t (s) | n | bias | rms | real rms | gain | corr | lag (ms) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in stats:
            gain = "-" if s.gain is None else f"{s.gain:.3g}"
            correlation = "-" if s.correlation is None else f"{s.correlation:.3g}"
            lag_ms = "-" if s.lag_s is None else f"{s.lag_s * 1000:+.1f}"
            lines.append(
                f"| {s.pair} | {s.phase} | {s.start_s:.1f}-{s.end_s:.1f} "
                f"| {s.sample_count} | {s.bias:+.4g} | {s.rms:.4g} | {s.real_rms:.4g} "
                f"| {gain} | {correlation} | {lag_ms} |"
            )
    if len(lines) == 2:
        lines.append("No phase information available (no phase column in this run).")
    return "\n".join(lines)


def stats_json(stats: list[PhaseStats], findings: list[Finding]) -> dict[str, object]:
    return {
        "phase_stats": [asdict(s) for s in stats],
        "findings": [asdict(f) for f in findings],
    }
