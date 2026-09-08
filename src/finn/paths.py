"""Repository paths and portable artifact names."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "sim/generated/seeded/latest/finn.seeded.sim.xml"
DEFAULT_OUT_ROOT = REPO_ROOT / "logs/lqr_sim"
DEFAULT_CONVENTIONS = REPO_ROOT / "config/finn_conventions.yaml"


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)
