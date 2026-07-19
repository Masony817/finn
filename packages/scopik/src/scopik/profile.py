"""Robot profile: the single YAML file that adapts scopik to a robot.

A profile declares where the MuJoCo model lives, how to parse the telemetry
log, which columns become named signals (with units and transforms), how to
replay recorded commands through the model, and which real/sim signals to compare.

Paths inside a profile are resolved relative to the profile file itself so a
profile can live anywhere in a robot's repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scopik.datamodel import ScopikError
from scopik.transforms import TransformSpec, resolve_transform


class ProfileError(ScopikError):
    """Profile file is missing, malformed, or references unknown things."""


def validate_transform(spec: TransformSpec, path: Path, context: str) -> None:
    """Validate a transform eagerly, attributing errors to the profile file."""

    try:
        resolve_transform(spec)
    except ScopikError as exc:
        raise ProfileError(f"{path}: {context}: {exc}") from exc


@dataclass(frozen=True)
class SourceProfile:
    type: str
    file: str
    line_prefix: str = "data,"
    header_marker: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeProfile:
    column: str
    transform: TransformSpec = None


@dataclass(frozen=True)
class SignalProfile:
    column: str
    name: str
    unit: str
    group: str
    transform: TransformSpec = None


@dataclass(frozen=True)
class EventsProfile:
    """Where lifecycle/segment events live and how to timestamp them.

    phase_column names a text column in the telemetry itself whose value
    changes are logged as timeline annotations (e.g. sysid segment names).
    """

    file: str
    prefix: str = "event,"
    time_index: int = 1
    time_transform: TransformSpec = None
    phase_column: str | None = None


@dataclass(frozen=True)
class ReplaySample:
    """One sim sensor sampled during command replay, emitted as a signal.

    gravity_compensated subtracts the gravity-reaction term (R_site^T · -g)
    from a site-attached 3-axis sensor before indexing, turning a MuJoCo
    accelerometer (specific force, gravity included) into a linear
    acceleration comparable to an IMU's gravity-removed output.
    """

    name: str
    sensor: str
    index: int | None = None
    transform: TransformSpec = None
    unit: str = ""
    group: str = "replay"
    gravity_compensated: bool = False


@dataclass(frozen=True)
class ReplayProfile:
    """hold_upright names a free joint to constrain like an ideal gantry:
    roll/pitch are projected out every step (yaw and translation stay free).
    Use when the real run was externally supported and the model is not."""

    actuators: dict[str, str]  # actuator name -> command column
    samples: tuple[ReplaySample, ...]
    hold_upright: str | None = None


@dataclass(frozen=True)
class ComparePair:
    name: str
    real: str
    sim: str
    unit: str = ""


@dataclass(frozen=True)
class SceneProfile:
    geom_filter: str = "visual_if_any"  # or "all"
    real_color: tuple[int, int, int] | None = (230, 110, 60)
    sim_color: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class Profile:
    name: str
    path: Path
    model_path: Path
    source: SourceProfile
    time: TimeProfile
    signals: tuple[SignalProfile, ...]
    events: EventsProfile | None
    replay: ReplayProfile | None
    compare: tuple[ComparePair, ...]
    scene: SceneProfile

    @property
    def signal_groups(self) -> list[str]:
        seen: dict[str, None] = {}
        for signal in self.signals:
            seen.setdefault(signal.group, None)
        return list(seen)


def load_profile(path: Path | str) -> Profile:
    path = Path(path).resolve()
    if not path.exists():
        raise ProfileError(f"profile not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"profile {path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"profile {path} must be a YAML mapping")
    return parse_profile(raw, path)


def parse_profile(raw: dict[str, Any], path: Path) -> Profile:
    name = require_str(raw, "name", path)
    model_rel = require_str(raw, "model", path)
    model_path = (path.parent / model_rel).resolve()

    source = parse_source(require_map(raw, "source", path), path)
    time = parse_time(require_map(raw, "time", path), path)
    signals = parse_signals(raw.get("signals") or {}, path)
    events = parse_events(raw.get("events"), path)
    replay = parse_replay(raw.get("replay"), path)
    compare = parse_compare(raw.get("compare"), path)
    scene = parse_scene(raw.get("scene"), path)

    signal_names = {signal.name for signal in signals}
    for pair in compare:
        if pair.real not in signal_names:
            raise ProfileError(
                f"{path}: compare pair {pair.name!r} references unknown real signal {pair.real!r}"
            )
    if replay is not None:
        replay_names = {sample.name for sample in replay.samples}
        for pair in compare:
            if pair.sim not in replay_names and pair.sim not in signal_names:
                raise ProfileError(
                    f"{path}: compare pair {pair.name!r} references sim signal {pair.sim!r} "
                    "that neither replay.sample nor signals defines"
                )

    return Profile(
        name=name,
        path=path,
        model_path=model_path,
        source=source,
        time=time,
        signals=signals,
        events=events,
        replay=replay,
        compare=compare,
        scene=scene,
    )


def parse_source(raw: dict[str, Any], path: Path) -> SourceProfile:
    source_type = require_str(raw, "type", path, context="source")
    file = require_str(raw, "file", path, context="source")
    known = {"type", "file", "line_prefix", "header_marker"}
    options = {key: value for key, value in raw.items() if key not in known}
    return SourceProfile(
        type=source_type,
        file=file,
        line_prefix=str(raw.get("line_prefix", "data,")),
        header_marker=raw.get("header_marker"),
        options=options,
    )


def parse_time(raw: dict[str, Any], path: Path) -> TimeProfile:
    column = require_str(raw, "column", path, context="time")
    transform = raw.get("transform")
    validate_transform(transform, path, "time.transform")
    return TimeProfile(column=column, transform=transform)


def parse_signals(raw: dict[str, Any], path: Path) -> tuple[SignalProfile, ...]:
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: signals must be a mapping of column -> settings")
    signals: list[SignalProfile] = []
    for column, settings in raw.items():
        settings = settings or {}
        if not isinstance(settings, dict):
            raise ProfileError(f"{path}: signals.{column} must be a mapping")
        transform = settings.get("transform")
        validate_transform(transform, path, f"signals.{column}.transform")
        signals.append(
            SignalProfile(
                column=str(column),
                name=str(settings.get("rename", column)),
                unit=str(settings.get("unit", "")),
                group=str(settings.get("group", "signals")),
                transform=transform,
            )
        )
    names = [signal.name for signal in signals]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ProfileError(f"{path}: duplicate signal names after rename: {sorted(duplicates)}")
    return tuple(signals)


def parse_events(raw: Any, path: Path) -> EventsProfile | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: events must be a mapping")
    file = require_str(raw, "file", path, context="events")
    transform = raw.get("time_transform")
    validate_transform(transform, path, "events.time_transform")
    phase_column = raw.get("phase_column")
    return EventsProfile(
        file=file,
        prefix=str(raw.get("prefix", "event,")),
        time_index=int(raw.get("time_index", 1)),
        time_transform=transform,
        phase_column=None if phase_column is None else str(phase_column),
    )


def parse_replay(raw: Any, path: Path) -> ReplayProfile | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: replay must be a mapping")
    actuators = raw.get("actuators")
    if not isinstance(actuators, dict) or not actuators:
        raise ProfileError(f"{path}: replay.actuators must map actuator names to command columns")
    samples_raw = raw.get("sample") or {}
    if not isinstance(samples_raw, dict):
        raise ProfileError(f"{path}: replay.sample must be a mapping of signal name -> settings")
    samples: list[ReplaySample] = []
    for name, settings in samples_raw.items():
        if not isinstance(settings, dict):
            raise ProfileError(f"{path}: replay.sample.{name} must be a mapping")
        sensor = require_str(settings, "sensor", path, context=f"replay.sample.{name}")
        transform = settings.get("transform")
        validate_transform(transform, path, f"replay.sample.{name}.transform")
        index = settings.get("index")
        samples.append(
            ReplaySample(
                name=str(name),
                sensor=sensor,
                index=None if index is None else int(index),
                transform=transform,
                unit=str(settings.get("unit", "")),
                group=str(settings.get("group", "replay")),
                gravity_compensated=bool(settings.get("gravity_compensated", False)),
            )
        )
    hold_upright = raw.get("hold_upright")
    return ReplayProfile(
        actuators={str(k): str(v) for k, v in actuators.items()},
        samples=tuple(samples),
        hold_upright=None if hold_upright is None else str(hold_upright),
    )


def parse_compare(raw: Any, path: Path) -> tuple[ComparePair, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ProfileError(f"{path}: compare must be a list of pairs")
    pairs: list[ComparePair] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ProfileError(f"{path}: compare[{i}] must be a mapping")
        pairs.append(
            ComparePair(
                name=require_str(entry, "name", path, context=f"compare[{i}]"),
                real=require_str(entry, "real", path, context=f"compare[{i}]"),
                sim=require_str(entry, "sim", path, context=f"compare[{i}]"),
                unit=str(entry.get("unit", "")),
            )
        )
    return tuple(pairs)


def parse_scene(raw: Any, path: Path) -> SceneProfile:
    if raw is None:
        return SceneProfile()
    if not isinstance(raw, dict):
        raise ProfileError(f"{path}: scene must be a mapping")
    geom_filter = str(raw.get("geom_filter", "visual_if_any"))
    if geom_filter not in ("visual_if_any", "all"):
        raise ProfileError(f"{path}: scene.geom_filter must be 'visual_if_any' or 'all'")

    def parse_color(
        value: Any, default: tuple[int, int, int] | None
    ) -> tuple[int, int, int] | None:
        if value is None:
            return default
        if not isinstance(value, list) or len(value) != 3:
            raise ProfileError(f"{path}: scene colors must be [r, g, b] lists")
        return (int(value[0]), int(value[1]), int(value[2]))

    return SceneProfile(
        geom_filter=geom_filter,
        real_color=parse_color(raw.get("real_color"), SceneProfile.real_color),
        sim_color=parse_color(raw.get("sim_color"), SceneProfile.sim_color),
    )


def require_str(raw: dict[str, Any], key: str, path: Path, context: str = "") -> str:
    prefix = f"{context}." if context else ""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{path}: {prefix}{key} must be a non-empty string")
    return value


def require_map(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ProfileError(f"{path}: {key} must be a mapping")
    return value
