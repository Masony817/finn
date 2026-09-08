"""Read Finn sysid telemetry and lifecycle events with shared wire-format rules."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any


def read_schema(telemetry: Path) -> str | None:
    if not telemetry.exists():
        return None
    for line in telemetry.read_text().splitlines():
        if line.startswith("schema,"):
            parts = line.split(",", 1)
            return parts[1] if len(parts) == 2 else None
    return None


def read_rows(telemetry: Path, *, strict: bool = True) -> list[dict[str, str]]:
    """Rows after the embedded header. strict=False tolerates a capture truncated
    mid-row (a power-off ends Batch 1 runs) instead of refusing the whole file."""

    if not telemetry.exists():
        return []
    data_lines = [line for line in telemetry.read_text().splitlines() if line.startswith("data,")]
    header_index = next(
        (index for index, line in enumerate(data_lines) if line.startswith("data,t_us,")),
        None,
    )
    if header_index is None:
        return []
    header = data_lines[header_index].split(",")
    if strict:
        for line_number, line in enumerate(data_lines[header_index + 1 :], start=header_index + 2):
            field_count = len(line.split(","))
            if field_count != len(header):
                raise ValueError(
                    f"Malformed telemetry row in {telemetry} at data line {line_number}: "
                    f"expected {len(header)} fields, got {field_count}"
                )
    return list(csv.DictReader(data_lines[header_index:]))


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("event,"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        event: dict[str, Any] = {
            "t_us": _to_float(parts[1]),
            "event": parts[2],
            "state": parts[3],
            "detail": ",".join(parts[4:]) if len(parts) > 4 else "",
            "fields": parts[4:],
        }
        if parts[2] == "segment_start" and len(parts) >= 5:
            event["phase"] = parts[4]
        if parts[2] == "segment_end":
            if len(parts) >= 7:
                event["phase"] = parts[4]
                event["reason"] = parts[5]
                event["elapsed_ms"] = _to_float(parts[6])
            else:
                parsed = dict(re.findall(r"([a-zA-Z_]+)=([^,]+)", event["detail"]))
                event.update(parsed)
                if "elapsed_ms" in event:
                    event["elapsed_ms"] = _to_float(str(event["elapsed_ms"]))
        events.append(event)
    return events


def _to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
