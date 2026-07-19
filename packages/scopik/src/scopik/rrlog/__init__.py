"""Rerun logging layer. Every rerun API call in scopik lives under rrlog/.

Rerun's Python API has renamed core symbols more than once; keeping the usage
surface in one small package makes version bumps a local diff instead of a
codebase sweep. rerun-sdk is pinned accordingly in pyproject.
"""

from __future__ import annotations

from pathlib import Path

import rerun as rr

TIMELINE = "time"

DEFAULT_REAL_COLOR = (230, 110, 60)
DEFAULT_SIM_COLOR = (90, 140, 255)
RESIDUAL_COLOR = (200, 60, 90)
ROLLING_COLOR = (140, 140, 150)


def init(app_id: str = "scopik") -> None:
    rr.init(app_id)


def sink_save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(path))


def sink_spawn(port: int = 9876) -> None:
    # rr.spawn() silently connects to ANY process already listening on the
    # port (even a stale non-viewer rerun server), which looks like "nothing
    # happened". Surface that so the failure mode is diagnosable.
    if _port_in_use(port):
        print(
            f"note: port {port} is already in use - sending to the existing listener "
            "instead of spawning a new viewer. If no viewer window shows your data, "
            f"kill the stale process (lsof -i :{port}) and rerun."
        )
    rr.spawn(port=port)


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def sink_serve() -> str:
    return str(rr.serve_grpc())


def set_time(time_s: float) -> None:
    rr.set_time(TIMELINE, duration=time_s)
