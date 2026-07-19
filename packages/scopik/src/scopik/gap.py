"""The gap pipeline: real run -> model replay -> residuals -> Rerun dashboard.

Pure orchestration; individually testable pieces live in sources/, reconstruct/,
metrics, and rrlog/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from scopik.datamodel import RunData
from scopik.diagnose import Finding, PhaseStats, analyze, phase_segments, stats_markdown
from scopik.metrics import PairResult, compare_runs, summary_json, summary_markdown
from scopik.profile import Profile
from scopik.sources import load_events, load_run, phase_changes

SCOPE_NOTE = (
    "Open-loop onboard-signal replay: recorded commands drive the model; rate and "
    "velocity signals are the honest comparison. Absolute pose diverges by design."
)


@dataclass
class GapReport:
    real_run: RunData
    sim_run: RunData | None
    results: list[PairResult]
    events: list[tuple[float, str]]
    model_path: Path | None
    phase_stats: list[PhaseStats] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    summary: dict[str, dict[str, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.summary = summary_json(self.results)

    def markdown(self) -> str:
        note = SCOPE_NOTE
        if self.sim_run is not None and self.sim_run.meta.get("hold_upright"):
            note += (
                f" Replay holds {self.sim_run.meta['hold_upright']} upright "
                "(ideal external support: roll/pitch constrained, yaw and translation free)."
            )
        return summary_markdown(self.results, scope_note=note)

    def diagnosis_markdown(self) -> str:
        return stats_markdown(self.phase_stats, self.findings)


def run_gap(
    profile: Profile,
    run_dir: Path,
    *,
    replay: bool = True,
    window_s: float = 1.0,
    model_path: Path | None = None,
) -> GapReport:
    real_run = load_run(run_dir, profile)

    events = load_events(run_dir, profile)
    if profile.events is not None and profile.events.phase_column:
        events = sorted(events + phase_changes(real_run, profile.events.phase_column))

    sim_run: RunData | None = None
    results: list[PairResult] = []
    resolved_model = model_path or profile.model_path
    if replay and profile.replay is not None:
        from scopik.reconstruct.replay import replay_commands

        sim_run = replay_commands(profile, real_run, model_path=resolved_model)
        results = compare_runs(real_run, sim_run, profile.compare, window_s=window_s)

    segments = []
    if profile.events is not None and profile.events.phase_column:
        segments = phase_segments(real_run, profile.events.phase_column)
    phase_stats_list, findings = analyze(results, segments)

    return GapReport(
        real_run=real_run,
        sim_run=sim_run,
        results=results,
        events=events,
        model_path=resolved_model if sim_run is not None else None,
        phase_stats=phase_stats_list,
        findings=findings,
    )


def log_to_rerun(report: GapReport, profile: Profile) -> None:
    """Log a complete gap report into the active Rerun recording."""

    from scopik.rrlog import DEFAULT_SIM_COLOR, blueprint, events, series

    real_color = profile.scene.real_color
    sim_color = profile.scene.sim_color or DEFAULT_SIM_COLOR

    series.log_run_signals(report.real_run, "/real", color=None)
    if report.sim_run is not None:
        series.log_run_signals(report.sim_run, "/sim", color=sim_color)
    for pair in report.results:
        series.log_pair(
            pair,
            real_color=real_color or (230, 110, 60),
            sim_color=sim_color,
        )
    if report.events:
        events.log_events(report.events)
    series.log_summary_document(report.markdown())
    if report.results:
        series.log_summary_document(report.diagnosis_markdown(), entity_path="/diagnosis")

    blueprint.send(
        blueprint.gap_blueprint(
            pair_names=[pair.name for pair in report.results],
            signal_groups=[
                group for group in profile.signal_groups if group_present(report, group)
            ],
            has_events=bool(report.events),
            has_diagnosis=bool(report.results),
        )
    )


def group_present(report: GapReport, group: str) -> bool:
    return any(signal.group == group for signal in report.real_run.signals.values())
