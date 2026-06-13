#!/usr/bin/env python3
"""Summarize a captured Batch 1 sysid run. No fitting yet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_batch1_rows(telemetry: Path) -> list[dict[str, str]]:
    if not telemetry.exists():
        return []

    lines = telemetry.read_text().splitlines()
    data_lines = [line for line in lines if line.startswith("data,")]
    if not data_lines:
        return []

    header_index = next(
        (index for index, line in enumerate(data_lines) if line.startswith("data,t_us,")),
        None,
    )
    if header_index is None:
        return []

    csv_lines = data_lines[header_index:]
    return list(csv.DictReader(csv_lines))


def write_derived_yaml(out_path: Path, rows: list[dict[str, str]], phases: list[str]) -> None:
    out_path.write_text(
        "# Placeholder derived values for Batch 1. No model parameters are fitted yet.\n"
        f"telemetry_rows: {len(rows)}\n"
        f"phase_count: {len(phases)}\n"
        "phases:\n"
        + "".join(f"  - {phase}\n" for phase in phases)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    telemetry = args.run_dir / "telemetry.csv"
    rows = read_batch1_rows(telemetry)
    phases = sorted({row["phase"] for row in rows if row.get("phase")})

    out_dir = args.run_dir / "postprocess"
    out_dir.mkdir(exist_ok=True)
    report = out_dir / "report.md"
    derived = out_dir / "derived.yaml"
    report.write_text(
        f"# Batch 1 sysid summary\n\n"
        f"Run: `{args.run_dir}`\n"
        f"Telemetry rows: {len(rows)}\n"
        f"Phases: {', '.join(phases) if phases else 'none'}\n"
    )
    write_derived_yaml(derived, rows, phases)
    print(f"Wrote {report} and {derived} ({len(rows)} rows, {len(phases)} phases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
