"""Unit/column transforms declared in profiles.

A transform is either a named shorthand ("rev_to_rad", "us_to_s", "negate",
"deg_to_rad") or a mapping {scale: float, offset: float} applied as
value * scale + offset. Profiles may chain them as a list.
"""

from __future__ import annotations

import math

import numpy as np

from scopik.datamodel import ScopikError

NAMED_TRANSFORMS: dict[str, tuple[float, float]] = {
    # name -> (scale, offset)
    "rev_to_rad": (2.0 * math.pi, 0.0),
    "rad_to_rev": (1.0 / (2.0 * math.pi), 0.0),
    "deg_to_rad": (math.pi / 180.0, 0.0),
    "rad_to_deg": (180.0 / math.pi, 0.0),
    "us_to_s": (1e-6, 0.0),
    "ms_to_s": (1e-3, 0.0),
    "negate": (-1.0, 0.0),
    "identity": (1.0, 0.0),
}

TransformSpec = str | dict[str, float] | list["TransformSpec"] | None


def resolve_transform(spec: TransformSpec) -> tuple[float, float]:
    """Collapse a transform spec into a single (scale, offset) pair."""

    if spec is None:
        return (1.0, 0.0)
    if isinstance(spec, str):
        if spec not in NAMED_TRANSFORMS:
            known = ", ".join(sorted(NAMED_TRANSFORMS))
            raise ScopikError(f"unknown transform {spec!r} (known: {known})")
        return NAMED_TRANSFORMS[spec]
    if isinstance(spec, dict):
        unknown = set(spec) - {"scale", "offset"}
        if unknown:
            raise ScopikError(f"transform mapping has unknown keys: {sorted(unknown)}")
        return (float(spec.get("scale", 1.0)), float(spec.get("offset", 0.0)))
    if isinstance(spec, list):
        scale, offset = 1.0, 0.0
        for part in spec:
            part_scale, part_offset = resolve_transform(part)
            # Composing y = b + B*(a + A*x): scale = A*B, offset = a*B + b.
            scale *= part_scale
            offset = offset * part_scale + part_offset
        return (scale, offset)
    raise ScopikError(f"invalid transform spec: {spec!r}")


def apply_transform(values: np.ndarray, spec: TransformSpec) -> np.ndarray:
    scale, offset = resolve_transform(spec)
    if scale == 1.0 and offset == 0.0:
        return values
    return values * scale + offset
