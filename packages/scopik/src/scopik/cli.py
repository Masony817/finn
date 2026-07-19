"""scopik command line for recorded sim-to-real gap analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scopik.datamodel import ScopikError
from scopik.profile import load_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scopik",
        description=(
            "Sim-to-real gap profiler on Rerun: replay a recorded robot run through "
            "its MuJoCo model and see where they disagree."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gap = subparsers.add_parser(
        "gap", help="Compare a recorded run against the model; open a Rerun dashboard."
    )
    gap.add_argument("--profile", type=Path, required=True, help="Robot profile YAML.")
    gap.add_argument("--run", type=Path, required=True, help="Run directory (or telemetry file).")
    gap.add_argument("--model", type=Path, help="Override the profile's model XML.")
    gap.add_argument(
        "--no-replay",
        action="store_true",
        help="Skip model replay; just visualize the recorded run.",
    )
    gap.add_argument("--window-s", type=float, default=1.0, help="Rolling RMSE window.")
    sink = gap.add_mutually_exclusive_group()
    sink.add_argument("--save", type=Path, help="Write a shareable .rrd instead of a viewer.")
    sink.add_argument("--serve", action="store_true", help="Serve to a connecting viewer.")
    gap.add_argument(
        "--out",
        type=Path,
        help="Where to write gap.json (default: <run>/scopik_gap.json).",
    )
    gap.add_argument(
        "--history",
        type=Path,
        help="Gap history JSONL (default: <profile name>_gap_history.jsonl next to profile).",
    )
    gap.add_argument("--no-history", action="store_true", help="Do not append to gap history.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_gap_command(args)
    except ScopikError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def run_gap_command(args: argparse.Namespace) -> int:
    from scopik import gap as gap_module
    from scopik import history as history_module
    from scopik import rrlog

    profile = load_profile(args.profile)
    report = gap_module.run_gap(
        profile,
        args.run,
        replay=not args.no_replay,
        window_s=args.window_s,
        model_path=args.model,
    )

    rrlog.init(f"scopik-gap-{profile.name}")
    serve_uri: str | None = None
    if args.save:
        rrlog.sink_save(args.save)
    elif args.serve:
        serve_uri = rrlog.sink_serve()
    else:
        rrlog.sink_spawn()

    gap_module.log_to_rerun(report, profile)

    out_path = args.out or (args.run / "scopik_gap.json" if args.run.is_dir() else None)
    if out_path is not None:
        from scopik.diagnose import stats_json

        payload = {
            "scope": gap_module.SCOPE_NOTE,
            "run_dir": str(args.run),
            "model": None if report.model_path is None else str(report.model_path),
            "metrics": report.summary,
            **stats_json(report.phase_stats, report.findings),
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out_path}")

    if report.results and not args.no_history and report.model_path is not None:
        history_path = args.history or history_module.default_history_path(profile)
        history_module.append_history(
            history_path,
            history_module.build_record(profile, args.run, report.model_path, report.results),
        )
        print(f"appended gap history: {history_path}")

    if report.results:
        print()
        print(f"sim-to-real gap  ({len(report.results)} signals, model: {report.model_path})")
        header = f"  {'signal':<16} {'unit':<6} {'rmse':>10} {'mae':>10} {'max |err|':>10} {'n':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for pair in report.results:
            s = pair.summary
            print(
                f"  {pair.name:<16} {pair.unit:<6} {s['rmse']:>10.4g} {s['mae']:>10.4g} "
                f"{s['max_abs']:>10.4g} {s['sample_count']:>7d}"
            )
        print()
        if report.findings:
            print("findings")
            marker = {"issue": "!", "note": "-", "ok": "+"}
            for finding in report.findings:
                print(f"  {marker[finding.severity]} {finding.message}")
            print()
        print(
            "  viewer tabs: Overview (overlays + diagnosis + summary + events), one tab "
            "per signal (residual + rolling RMSE), Telemetry (raw groups)."
        )
        print("  residual spikes locate WHERE the model diverges; scrub the timeline to them.")
    else:
        print("no compare pairs computed (no replay or no matching signals)")
    if args.save:
        print(f"saved recording: {args.save}  (open with: rerun {args.save})")
    if serve_uri:
        print(f"serving at {serve_uri}; press Ctrl-C to stop")
        try:
            import time

            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
