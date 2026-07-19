"""Timeline annotations: lifecycle events and phase changes as text logs."""

from __future__ import annotations

import rerun as rr

from scopik.rrlog import TIMELINE


def log_events(events: list[tuple[float, str]], entity_path: str = "/events") -> None:
    for time_s, message in events:
        rr.set_time(TIMELINE, duration=time_s)
        level = rr.TextLogLevel.WARN if "fault" in message.lower() else rr.TextLogLevel.INFO
        rr.log(entity_path, rr.TextLog(message, level=level))
