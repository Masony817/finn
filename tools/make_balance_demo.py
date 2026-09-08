#!/usr/bin/env python3
"""Render a shareable balance demo from one LQR rollout: a chart and a GIF.

Runs the same controller `tools/run_lqr_sim.py` validates, shoves the robot on a
schedule, and writes a blog-ready PNG plus an animated GIF of each recovery.  The
pushes are the point: a standing rollout and a balancing rollout look identical
until something disturbs them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from finn import control, paths, reporting
from finn import lqr as cli
from finn import simulation as sim

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "sim/generated/seeded/latest/finn.seeded.sim.xml"
DEFAULT_OUT_ROOT = REPO_ROOT / "logs/lqr_demo"

# Palette slots 1-3 of the validated categorical set, plus its text/surface tokens.
LIGHT_THEME = {
    "surface": "#fcfcfb",
    "panel": "#f2f2f0",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#d9d9d6",
    "pitch": "#2a78d6",
    "travel": "#eb6834",
    "torque": "#1baf7a",
    "push": "#8a8880",
}
DARK_THEME = {
    "surface": "#1a1a19",
    "panel": "#262624",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "grid": "#3b3b38",
    "pitch": "#3987e5",
    "travel": "#d95926",
    "torque": "#199e70",
    "push": "#8a8880",
}

# Window after a push in which the recovery is measured.  Long enough for the
# outer position loop to bring the wheels back under the mast at these gains.
RECOVERY_WINDOW_S = 3.0


class DemoError(Exception):
    """Expected failure with a concise user-facing message."""


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class Push:
    """One scripted shove against the chassis COM."""

    start_s: float
    duration_s: float
    force_n: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def force_at(self, time_s: float) -> float:
        return self.force_n if self.start_s <= time_s < self.end_s else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Finn LQR balance controller in sim and render shareable artifacts."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--control-dt-s", type=float, default=0.01)
    parser.add_argument("--initial-pitch-rad", type=float, default=0.03)
    parser.add_argument(
        "--push-at-s",
        type=float,
        nargs="*",
        default=(2.0, 5.5, 9.0),
        help="Times to shove the chassis. Direction alternates, starting forward.",
    )
    parser.add_argument(
        "--push-force-n",
        type=float,
        default=22.0,
        help=(
            "Push magnitude at the chassis COM. The default leans the robot about 8 "
            "degrees; it stops recovering somewhere above 32 N at the shipped gains."
        ),
    )
    parser.add_argument("--push-duration-s", type=float, default=0.12)
    parser.add_argument("--q-diag", type=float, nargs=3, default=(160.0, 16.0, 2.0))
    parser.add_argument("--r", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=460)
    parser.add_argument(
        "--camera-distance-m",
        type=float,
        help="Camera distance. Default fits the robot's rest-pose height in frame.",
    )
    parser.add_argument("--camera-azimuth-deg", type=float, default=115.0)
    parser.add_argument("--camera-elevation-deg", type=float, default=-10.0)
    parser.add_argument(
        "--camera-margin",
        type=float,
        default=1.5,
        help="Headroom multiplier on the auto-fit distance, for lean and travel.",
    )
    parser.add_argument("--gif-colors", type=int, default=128)
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument(
        "--conventions",
        type=Path,
        default=REPO_ROOT / "config/finn_conventions.yaml",
        help="Machine-readable Finn frame/sign convention contract.",
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except DemoError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"wrote balance demo bundle: {result['out_dir']}")
    for name, path in result["artifacts"].items():
        print(f"  {name}: {path}")
    print(f"status: {result['status']}")
    for recovery in result["recoveries"]:
        settle = recovery["settle_s"]
        settle_text = f"{settle:.2f} s" if settle is not None else "not settled"
        print(
            f"  push {recovery['force_n']:+.1f} N at t={recovery['start_s']:.1f}s -> "
            f"peak lean {recovery['peak_lean_deg']:.2f} deg, recovered in {settle_text}"
        )
    missed = len(result["pushes"]) - len(result["recoveries"])
    if missed:
        print(f"  {missed} push(es) never happened: the rollout ended when the robot fell")
    return 0 if result["status"] == "pass" else 1


def run(args: argparse.Namespace) -> dict[str, object]:

    if not args.model.exists():
        raise DemoError(f"missing model XML: {args.model}")
    if args.fps <= 0:
        raise DemoError("--fps must be positive")

    out_dir = args.out_dir or DEFAULT_OUT_ROOT / dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    handles = sim.inspect_model(model)
    trim_pitch_rad = sim.estimate_balance_trim_pitch_rad(model)

    config = sim.SimConfig(
        control_dt_s=args.control_dt_s,
        duration_s=args.duration_s,
        initial_pitch_rad=args.initial_pitch_rad,
        target_pitch_rad=trim_pitch_rad,
        target_forward_vel_m_s=0.0,
        position_hold_kp_s=0.15,
        max_position_correction_m_s=0.15,
        linearization_torque_eps_nm=0.18,
        linearization_vel_eps_m_s=0.05,
        fall_pitch_rad=0.45,
        pitch_axis=0,
        pitch_sign=1.0,
        forward_sign=1.0,
        yaw_axis=1,
        yaw_sign=1.0,
        yaw_left_actuator_sign=cli.yaw_left_actuator_sign(args),
        drive=control.DriveLimits(),
    )
    sim.validate_timing(model, config)
    sim.validate_linearization_torque(config, handles)

    estimator = sim.calibrated_estimator(model, handles, config)
    a_matrix, b_matrix = sim.linearize_balance_dynamics(model, handles, estimator, config)
    gain = sim.discrete_lqr(
        a_matrix,
        b_matrix,
        np.diag(np.array(args.q_diag, dtype=float)),
        np.array([[float(args.r)]], dtype=float),
    )

    pushes = build_pushes(args)
    capture_every = frames_decimation(args.fps, config.control_dt_s)
    recorder = Recorder(
        model=model,
        base_body_id=sim.require_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"),
        pushes=pushes,
        capture_every=0 if args.no_gif else capture_every,
        width=args.width,
        height=args.height,
        camera_distance_m=args.camera_distance_m,
        camera_azimuth_deg=args.camera_azimuth_deg,
        camera_elevation_deg=args.camera_elevation_deg,
        camera_margin=args.camera_margin,
    )

    with recorder:
        rows, metrics = cli.run_closed_loop(
            model, handles, estimator, config, gain, on_tick=recorder
        )

    measured = [measure_recovery(rows, push, trim_pitch_rad) for push in pushes]
    recoveries = [recovery for recovery in measured if recovery is not None]
    status = demo_status(metrics, recoveries, rows, trim_pitch_rad)
    if len(recoveries) < len(pushes):
        status = "failed"

    csv_path = out_dir / "timeseries.csv"
    reporting.write_timeseries(csv_path, rows)
    artifacts = {"timeseries_csv": portable_path(csv_path)}

    theme = LIGHT_THEME if args.theme == "light" else DARK_THEME
    if not args.no_plot:
        plot_path = out_dir / "balance_chart.png"
        write_chart(
            plot_path,
            rows=rows,
            pushes=pushes,
            recoveries=recoveries,
            trim_pitch_rad=trim_pitch_rad,
            torque_limit_nm=handles.torque_limit_nm,
            theme=theme,
        )
        artifacts["chart_png"] = portable_path(plot_path)

    if not args.no_gif:
        gif_path = out_dir / "balance.gif"
        write_gif(
            gif_path,
            recorder=recorder,
            pushes=pushes,
            trim_pitch_rad=trim_pitch_rad,
            duration_s=config.duration_s,
            fps=args.fps,
            colors=args.gif_colors,
            theme=theme,
        )
        artifacts["gif"] = portable_path(gif_path)
        artifacts["gif_mb"] = round(gif_path.stat().st_size / 1e6, 2)

    result: dict[str, object] = {
        "status": status,
        "out_dir": portable_path(out_dir),
        "model": portable_path(args.model),
        "model_sha256_12": paths.sha256_12(args.model),
        "gain": gain.tolist(),
        "trim_pitch_rad": trim_pitch_rad,
        "torque_limit_nm": handles.torque_limit_nm,
        "pushes": [
            {"start_s": push.start_s, "duration_s": push.duration_s, "force_n": push.force_n}
            for push in pushes
        ],
        "recoveries": recoveries,
        "metrics": metrics,
        "artifacts": artifacts,
        "scope": (
            "Sim-only. The controller reads IMU attitude/rate and wheel odometry, never "
            "MuJoCo ground truth, but this is not evidence of sim-to-real transfer."
        ),
    }
    (out_dir / "demo.json").write_text(json.dumps(result, indent=2, sort_keys=True), "utf-8")
    return result


def build_pushes(args: argparse.Namespace) -> list[Push]:
    pushes: list[Push] = []
    for index, start_s in enumerate(sorted(args.push_at_s)):
        if start_s < 0.0 or start_s + args.push_duration_s > args.duration_s:
            raise DemoError(f"push at t={start_s}s does not fit inside --duration-s")
        direction = 1.0 if index % 2 == 0 else -1.0
        pushes.append(
            Push(
                start_s=float(start_s),
                duration_s=float(args.push_duration_s),
                force_n=direction * abs(float(args.push_force_n)),
            )
        )
    return pushes


def frames_decimation(fps: int, control_dt_s: float) -> int:
    return max(1, round(1.0 / (fps * control_dt_s)))


class Recorder:
    """Applies the scripted pushes and captures render frames during the rollout."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        base_body_id: int,
        pushes: list[Push],
        capture_every: int,
        width: int,
        height: int,
        camera_distance_m: float | None,
        camera_azimuth_deg: float,
        camera_elevation_deg: float,
        camera_margin: float,
    ) -> None:
        self.model = model
        self.base_body_id = base_body_id
        self.pushes = pushes
        self.capture_every = capture_every
        self.width = width
        self.height = height
        self.frames: list[np.ndarray] = []
        self.frame_rows: list[dict[str, float]] = []
        self.renderer: mujoco.Renderer | None = None

        lookat, fit_distance_m = frame_robot(model, camera_margin)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = camera_distance_m or fit_distance_m
        self.camera.azimuth = camera_azimuth_deg
        self.camera.elevation = camera_elevation_deg
        self.camera.lookat[:] = lookat

    @property
    def forward_points_right(self) -> bool:
        """Whether world +x renders toward the right of frame, for the push arrow."""

        return math.sin(math.radians(self.camera.azimuth)) >= 0.0

    def __enter__(self) -> Recorder:
        if self.capture_every > 0:
            # The offscreen framebuffer defaults to 640x480 and the Renderer refuses to
            # exceed it. Widen the loaded model rather than the generated XML on disk.
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __call__(self, tick: int, row: dict[str, float], data: mujoco.MjData) -> None:
        force_n = sum(push.force_at(row["time_s"]) for push in self.pushes)
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[self.base_body_id, 0] = force_n
        row["push_force_n"] = force_n

        if self.renderer is not None and tick % self.capture_every == 0:
            self.renderer.update_scene(data, self.camera)
            self.frames.append(self.renderer.render().copy())
            self.frame_rows.append(dict(row))


def frame_robot(model: mujoco.MjModel, margin: float) -> tuple[np.ndarray, float]:
    """Return the camera lookat and the distance that fits the robot at rest.

    Neither model.stat nor geom_rbound frames this robot: stat is dominated by the
    floor plane, and the bounding spheres of the wheel meshes reach below it. Mesh
    vertices give the real silhouette, which is nearly all mast.
    """

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) == 0 or int(model.geom_group[geom_id]) == 3:
            continue
        corners = geom_world_extent(model, data, geom_id)
        lower = np.minimum(lower, corners.min(axis=0))
        upper = np.maximum(upper, corners.max(axis=0))
    if not np.all(np.isfinite(lower)):
        raise DemoError("model has no visual geoms to frame the camera on")

    top_m = float(upper[2])
    lookat = np.array([0.0, float(0.5 * (lower[1] + upper[1])), 0.5 * top_m], dtype=float)
    half_fov_rad = math.radians(float(model.vis.global_.fovy)) / 2.0
    return lookat, margin * (top_m / 2.0) / math.tan(half_fov_rad)


def geom_world_extent(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    """World-space points bounding one geom: mesh vertices, or the bounding sphere."""

    position = data.geom_xpos[geom_id]
    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        return vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + position
    radius = float(model.geom_rbound[geom_id])
    return np.array([position - radius, position + radius], dtype=float)


def measure_recovery(
    rows: list[dict[str, float]], push: Push, trim_pitch_rad: float
) -> dict[str, object] | None:
    """Peak lean caused by one push and how long the controller took to null it.

    None means the rollout never reached this push, which is what an earlier push
    knocking the robot past run_closed_loop's fall threshold looks like from here.
    """

    window = [
        row for row in rows if push.start_s <= row["time_s"] <= push.end_s + RECOVERY_WINDOW_S
    ]
    if not window:
        return None

    lean = [abs(row["pitch_rad"] - trim_pitch_rad) for row in window]
    peak_index = int(np.argmax(lean))
    peak_lean_rad = lean[peak_index]

    # "Settled" means the lean stays inside the band for the rest of the window, so a
    # single zero crossing on the way through an oscillation does not count.
    settle_band_rad = max(0.2 * peak_lean_rad, math.radians(0.2))
    settle_s: float | None = None
    for index in range(peak_index, len(window)):
        if all(value <= settle_band_rad for value in lean[index:]):
            settle_s = window[index]["time_s"] - push.start_s
            break

    return {
        "start_s": push.start_s,
        "force_n": push.force_n,
        "peak_lean_deg": math.degrees(peak_lean_rad),
        "peak_lean_rad": peak_lean_rad,
        "settle_s": settle_s,
        "settle_band_deg": math.degrees(settle_band_rad),
        "peak_torque_nm": max(abs(row["tau_balance_nm"]) for row in window),
        "travel_excursion_m": max(row["forward_pos_m"] for row in window)
        - min(row["forward_pos_m"] for row in window),
    }


def demo_status(
    metrics: dict[str, float | bool],
    recoveries: list[dict[str, object]],
    rows: list[dict[str, float]],
    target_pitch_rad: float,
) -> str:
    """Pass only if the robot survived every push and ended upright and stopped.

    run_closed_loop's own verdict is not reusable here: its position-hold gate assumes
    an undisturbed rollout, and being shoved is the entire experiment.
    """

    if not metrics["finite"] or metrics["fell"]:
        return "failed"
    if any(recovery["settle_s"] is None for recovery in recoveries):
        return "failed"
    final = rows[-1]
    upright = abs(final["pitch_rad"] - target_pitch_rad) < math.radians(2.0)
    stopped = abs(final["forward_vel_m_s"]) < 0.05
    return "pass" if upright and stopped else "failed"


def write_chart(
    path: Path,
    *,
    rows: list[dict[str, float]],
    pushes: list[Push],
    recoveries: list[dict[str, object]],
    trim_pitch_rad: float,
    torque_limit_nm: float,
    theme: dict[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    time_s = [row["time_s"] for row in rows]
    lean_deg = [math.degrees(row["pitch_rad"] - trim_pitch_rad) for row in rows]
    travel_m = [row["forward_pos_m"] for row in rows]
    torque_nm = [row["tau_balance_nm"] for row in rows]

    fig = plt.figure(figsize=(9.5, 8.6), dpi=170)
    fig.patch.set_facecolor(theme["surface"])
    grid = GridSpec(4, 3, figure=fig, height_ratios=[0.75, 1.5, 1.0, 1.0], hspace=0.42, wspace=0.16)

    settled = [r["settle_s"] for r in recoveries if r["settle_s"] is not None]
    peak_lean = [r["peak_lean_deg"] for r in recoveries]
    tiles = [
        (
            f"{max(peak_lean):.1f}°" if peak_lean else "fell",
            "peak lean under push",
            theme["pitch"],
        ),
        (
            f"{max(settled):.1f} s" if settled else "n/a",
            "slowest recovery to upright",
            theme["travel"],
        ),
        (
            f"{max(abs(value) for value in torque_nm):.2f} / {torque_limit_nm:.2f} N·m",
            "peak torque vs. motor limit",
            theme["torque"],
        ),
    ]
    for column, (value, label, accent) in enumerate(tiles):
        axis = fig.add_subplot(grid[0, column])
        draw_stat_tile(axis, value, label, accent, theme)

    axes = [fig.add_subplot(grid[index + 1, :]) for index in range(3)]
    for axis in axes:
        style_axis(axis, theme)
        for push in pushes:
            axis.axvspan(push.start_s, push.end_s, color=theme["push"], alpha=0.22, linewidth=0)

    axes[0].axhline(0.0, color=theme["grid"], linewidth=1.0)
    axes[0].plot(time_s, lean_deg, color=theme["pitch"], linewidth=2.0)
    axes[0].set_ylabel("lean from balance point (deg)", color=theme["text_secondary"])
    label_panel(axes[0], "Body lean - every spike is a shove it did not fall over from", theme)
    # Headroom so the peak callouts sit inside the panel instead of over the next title.
    lean_limit = 1.55 * max(abs(value) for value in lean_deg) or 1.0
    axes[0].set_ylim(-lean_limit, lean_limit)
    for recovery in recoveries:
        annotate_recovery(axes[0], rows, recovery, trim_pitch_rad, theme)

    axes[1].axhline(0.0, color=theme["grid"], linewidth=1.0)
    axes[1].plot(time_s, travel_m, color=theme["travel"], linewidth=2.0)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    axes[1].set_ylabel("wheel travel (m)", color=theme["text_secondary"])
    label_panel(axes[1], "Ground position - the wheels drive under the mast to catch it", theme)

    axes[2].axhline(0.0, color=theme["grid"], linewidth=1.0)
    for limit in (-torque_limit_nm, torque_limit_nm):
        axes[2].axhline(limit, color=theme["text_secondary"], linewidth=1.0, linestyle=(0, (4, 4)))
    axes[2].plot(time_s, torque_nm, color=theme["torque"], linewidth=2.0)
    axes[2].set_ylim(-1.15 * torque_limit_nm, 1.15 * torque_limit_nm)
    axes[2].set_ylabel("torque per wheel (N·m)", color=theme["text_secondary"])
    axes[2].set_xlabel("time (s)", color=theme["text_secondary"])
    label_panel(axes[2], "Motor command (dashed = actuator limit)", theme)

    for axis in axes:
        axis.set_xlim(time_s[0], time_s[-1])
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)

    push_text = (
        f"{abs(pushes[0].force_n):.0f} N shove for {pushes[0].duration_s * 1000:.0f} ms"
        if pushes
        else "no disturbance"
    )
    fig.text(
        0.055,
        0.972,
        "Finn holds itself up in simulation",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=theme["text_primary"],
    )
    fig.text(
        0.055,
        0.941,
        f"LQR balance controller. Shaded bands are a {push_text} against the chassis.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=theme["text_secondary"],
    )
    fig.text(
        0.055,
        0.014,
        "MuJoCo model seeded from bench measurements. The controller sees only IMU "
        "attitude, IMU rate, and wheel odometry.",
        ha="left",
        fontsize=8.5,
        color=theme["text_secondary"],
    )
    fig.subplots_adjust(left=0.093, right=0.972, top=0.895, bottom=0.082)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def draw_stat_tile(axis, value: str, label: str, accent: str, theme: dict[str, str]) -> None:
    axis.set_facecolor(theme["panel"])
    axis.set_xticks([])
    axis.set_yticks([])
    for side, spine in axis.spines.items():
        spine.set_visible(side == "left")
        spine.set_color(accent)
        spine.set_linewidth(3.0)
    axis.text(
        0.055,
        0.62,
        value,
        transform=axis.transAxes,
        fontsize=19,
        fontweight="bold",
        color=theme["text_primary"],
        va="center",
    )
    axis.text(
        0.055,
        0.24,
        label,
        transform=axis.transAxes,
        fontsize=9,
        color=theme["text_secondary"],
        va="center",
    )


def style_axis(axis, theme: dict[str, str]) -> None:
    axis.set_facecolor(theme["surface"])
    axis.grid(axis="y", color=theme["grid"], linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(theme["grid"])
    axis.tick_params(colors=theme["text_secondary"], labelsize=9)


def label_panel(axis, text: str, theme: dict[str, str]) -> None:
    axis.set_title(
        text, loc="left", fontsize=11, fontweight="bold", color=theme["text_primary"], pad=6
    )


def annotate_recovery(
    axis,
    rows: list[dict[str, float]],
    recovery: dict[str, object],
    trim_pitch_rad: float,
    theme: dict[str, str],
) -> None:
    start_s = float(recovery["start_s"])
    window = [row for row in rows if start_s <= row["time_s"] <= start_s + RECOVERY_WINDOW_S]
    peak_row = max(window, key=lambda row: abs(row["pitch_rad"] - trim_pitch_rad))
    peak_deg = math.degrees(peak_row["pitch_rad"] - trim_pitch_rad)
    settle_s = recovery["settle_s"]
    text = f"{abs(peak_deg):.1f}°"
    if settle_s is not None:
        text += f", back in {float(settle_s):.1f}s"
    axis.annotate(
        text,
        xy=(peak_row["time_s"], peak_deg),
        xytext=(0, 14 if peak_deg >= 0 else -20),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=theme["text_primary"],
    )


def write_gif(
    path: Path,
    *,
    recorder: Recorder,
    pushes: list[Push],
    trim_pitch_rad: float,
    duration_s: float,
    fps: int,
    colors: int,
    theme: dict[str, str],
) -> None:
    from PIL import Image

    if not recorder.frames:
        raise DemoError("no frames were captured; rendering produced an empty animation")

    strip_height = max(80, round(0.22 * recorder.height))
    lean_deg = [math.degrees(row["pitch_rad"] - trim_pitch_rad) for row in recorder.frame_rows]
    lean_span_deg = max(2.0, 1.25 * max(abs(value) for value in lean_deg))

    frames = [
        compose_frame(
            pixels=pixels,
            row=row,
            index=index,
            lean_deg=lean_deg,
            lean_span_deg=lean_span_deg,
            strip_height=strip_height,
            duration_s=duration_s,
            pushes=pushes,
            theme=theme,
            forward_points_right=recorder.forward_points_right,
        )
        for index, (pixels, row) in enumerate(
            zip(recorder.frames, recorder.frame_rows, strict=True)
        )
    ]

    # One shared palette with dithering off, so unchanged background pixels stay
    # bit-identical between frames and the encoder can ship inter-frame diffs. Local
    # palettes cost roughly 3x the file size on this scene.
    push_frames = [index for index, row in enumerate(recorder.frame_rows) if row["push_force_n"]]
    palette = global_palette(frames, colors, push_frames, theme)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000.0 / fps),
        loop=0,
        optimize=True,
    )


def global_palette(frames: list, colors: int, required_indices: list[int], theme: dict[str, str]):
    """Derive one palette from a sample of frames plus a band of the theme colors.

    Median-cut allocates entries by pixel count, so against a scene of blue floor it
    spends nothing on a 2px trace or a badge that shows for two frames out of a few
    hundred. The push frames are sampled explicitly and the theme colors are given a
    block each, large enough to survive the cut at their authored values.
    """

    from PIL import Image, ImageDraw

    spaced = np.linspace(0, len(frames) - 1, num=min(12, len(frames)), dtype=int).tolist()
    sample_indices = sorted({int(index) for index in spaced + required_indices})
    width, height = frames[0].size

    swatch_height = max(24, height // 8)
    montage = Image.new("RGB", (width, height * len(sample_indices) + swatch_height))
    for slot, frame_index in enumerate(sample_indices):
        montage.paste(frames[frame_index], (0, slot * height))

    draw = ImageDraw.Draw(montage)
    swatch_top = height * len(sample_indices)
    swatch_width = width / len(theme)
    for slot, color in enumerate(theme.values()):
        draw.rectangle(
            [slot * swatch_width, swatch_top, (slot + 1) * swatch_width, montage.height],
            fill=color,
        )
    return montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def compose_frame(
    *,
    pixels: np.ndarray,
    row: dict[str, float],
    index: int,
    lean_deg: list[float],
    lean_span_deg: float,
    strip_height: int,
    duration_s: float,
    pushes: list[Push],
    theme: dict[str, str],
    forward_points_right: bool,
):
    from PIL import Image, ImageDraw

    render = Image.fromarray(pixels)
    width, height = render.size
    canvas = Image.new("RGB", (width, height + strip_height), theme["surface"])
    canvas.paste(render, (0, 0))

    draw_hud(canvas, row, theme, forward_points_right, lean_deg[index])

    draw = ImageDraw.Draw(canvas)
    top = height
    draw.rectangle([0, top, width, top + strip_height], fill=theme["panel"])
    inset = 12.0
    trace_half_height = strip_height / 2.0 - 20.0

    def x_of(time_s: float) -> float:
        return inset + (width - 2 * inset) * min(1.0, max(0.0, time_s / duration_s))

    def y_of(value_deg: float) -> float:
        return top + strip_height / 2.0 - trace_half_height * (value_deg / lean_span_deg)

    now_s = row["time_s"]
    for push in pushes:
        if push.start_s > now_s:
            continue
        draw.rectangle(
            [
                x_of(push.start_s),
                top + 18,
                max(x_of(min(push.end_s, now_s)), x_of(push.start_s) + 2),
                top + strip_height - 4,
            ],
            fill=blend(theme["push"], theme["panel"], 0.45),
        )
    draw.line([(inset, y_of(0.0)), (width - inset, y_of(0.0))], fill=theme["grid"], width=1)

    trace = [
        (x_of(row_index / max(1, len(lean_deg) - 1) * duration_s), y_of(value))
        for row_index, value in enumerate(lean_deg[: index + 1])
    ]
    if len(trace) > 1:
        draw.line(trace, fill=theme["pitch"], width=2, joint="curve")
    if trace:
        head_x, head_y = trace[-1]
        draw.ellipse([head_x - 4, head_y - 4, head_x + 4, head_y + 4], fill=theme["pitch"])

    font = load_font(12)
    draw.text((inset, top + 4), "body lean", font=font, fill=theme["text_secondary"])
    draw.text(
        (width - inset, top + 4),
        f"full scale ±{lean_span_deg:.1f}°",
        font=font,
        fill=theme["text_secondary"],
        anchor="ra",
    )
    return canvas


def draw_hud(
    canvas,
    row: dict[str, float],
    theme: dict[str, str],
    forward_points_right: bool,
    lean_deg: float,
) -> None:
    from PIL import Image, ImageDraw

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle([16, 16, 228, 142], radius=12, fill=(10, 10, 10, 155))

    label_font = load_font(12)
    value_font = load_font(15, bold=True)
    entries = [
        ("time", f"{row['time_s']:.2f} s"),
        ("lean", f"{lean_deg:+.2f}°"),
        ("lean rate", f"{math.degrees(row['pitch_rate_rad_s']):+.1f} °/s"),
        ("wheel travel", f"{row['forward_pos_m']:+.3f} m"),
        ("torque", f"{row['tau_balance_nm']:+.2f} N·m"),
    ]
    for line, (label, value) in enumerate(entries):
        top = 26 + line * 22
        draw.text((30, top), label, font=label_font, fill=(208, 208, 202, 235))
        draw.text((214, top - 2), value, font=value_font, fill=(255, 255, 255, 245), anchor="ra")

    if row.get("push_force_n"):
        pushed_right = (row["push_force_n"] > 0) == forward_points_right
        arrow = "→" if pushed_right else "←"
        draw.rounded_rectangle([16, 152, 228, 188], radius=12, fill=hex_rgba(theme["travel"], 240))
        draw.text(
            (122, 170),
            f"PUSH  {abs(row['push_force_n']):.0f} N  {arrow}",
            font=value_font,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )

    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"), (0, 0))


def blend(color: str, background: str, weight: float) -> tuple[int, int, int]:
    front = hex_rgba(color, 255)[:3]
    back = hex_rgba(background, 255)[:3]
    return tuple(round(weight * f + (1.0 - weight) * b) for f, b in zip(front, back, strict=True))


def hex_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha)


def load_font(size: int, *, bold: bool = False):
    """Use the font matplotlib ships so the GIF renders the same on any machine."""

    from matplotlib import get_data_path
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf" / name), size)


if __name__ == "__main__":
    sys.exit(main())
