"""Shared fixtures: a tiny synthetic robot (model + profile + run dir)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

SPINNER_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <body name="spinner">
      <joint name="hinge" type="hinge" axis="0 0 1" damping="0.01"/>
      <geom type="box" size="0.05 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
  <actuator><motor name="motor_hinge" joint="hinge" ctrlrange="-1 1"/></actuator>
  <sensor><jointvel name="hinge_vel" joint="hinge"/></sensor>
</mujoco>
"""

PROFILE_YAML = """
name: spinner
model: spinner.xml

source:
  type: prefixed_csv
  file: telemetry.csv
  line_prefix: "data,"
  header_marker: "data,t_us,"

time:
  column: t_us
  transform: us_to_s

events:
  file: events.log
  prefix: "event,"
  time_index: 1
  time_transform: us_to_s
  phase_column: phase

signals:
  vel_rad_s: {unit: rad/s, group: motion}
  cmd_nm: {unit: "N·m", group: commands}

replay:
  actuators:
    motor_hinge: cmd_nm
  sample:
    vel_rad_s: {sensor: hinge_vel, unit: rad/s}

compare:
  - {name: vel, real: vel_rad_s, sim: vel_rad_s, unit: rad/s}
"""


def write_spinner_run(root: Path, rows: int = 200, hz: float = 100.0) -> Path:
    """Write model + profile + a synthetic run dir; returns the run dir."""

    (root / "spinner.xml").write_text(SPINNER_XML, encoding="utf-8")
    (root / "profile.yaml").write_text(PROFILE_YAML, encoding="utf-8")

    run_dir = root / "run"
    run_dir.mkdir(exist_ok=True)
    lines = ["schema,spinner_v1", "data,t_us,state,phase,cmd_nm,vel_rad_s"]
    for i in range(rows):
        t_us = int(i / hz * 1e6)
        phase = "pulse" if i < rows // 2 else "coast"
        cmd = 0.2 if i < rows // 2 else 0.0
        vel = 2.0 * math.sin(2 * math.pi * 0.5 * i / hz)
        lines.append(f"data,{t_us},running,{phase},{cmd},{vel:.5f}")
    (run_dir / "telemetry.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "events.log").write_text(
        "event,0,run_started,running,ok\n"
        f"event,{int(rows // 2 / hz * 1e6)},coast_started,running,ok\n",
        encoding="utf-8",
    )
    return run_dir


@pytest.fixture
def spinner(tmp_path: Path) -> dict[str, Path]:
    run_dir = write_spinner_run(tmp_path)
    return {
        "root": tmp_path,
        "run_dir": run_dir,
        "profile": tmp_path / "profile.yaml",
        "model": tmp_path / "spinner.xml",
    }
