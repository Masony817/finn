"""Physics-honesty features: gravity compensation and the upright hold."""

from pathlib import Path

import numpy as np
import pytest
from scopik.profile import load_profile
from scopik.reconstruct.replay import replay_commands
from scopik.sources import load_run

pytest.importorskip("mujoco")

# A top-heavy box on a freejoint with an offset COM: it tips over within a
# couple of seconds unless held. The imu site starts with z pointing world +x.
TIPPY_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom type="box" size="0.04 0.04 0.3" mass="0.2"/>
      <geom type="sphere" size="0.03" mass="2.0" pos="0.2 0 0.1"/>
      <site name="imu" pos="0 0 0.2" zaxis="1 0 0"/>
      <body name="rotor" pos="0 0 0.32">
        <joint name="spin" type="hinge" axis="0 0 1"/>
        <geom type="cylinder" size="0.03 0.01" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="motor_spin" joint="spin" ctrlrange="-1 1"/></actuator>
  <sensor>
    <accelerometer name="acc" site="imu"/>
    <framequat name="quat" objtype="xbody" objname="base"/>
  </sensor>
</mujoco>
"""

PROFILE_TEMPLATE = """
name: tippy
model: tippy.xml
source: {{type: prefixed_csv, file: telemetry.csv}}
time: {{column: t_us, transform: us_to_s}}
signals:
  cmd_nm: {{unit: "N·m", group: commands}}
replay:
  actuators: {{motor_spin: cmd_nm}}
{hold_line}
  sample:
    acc_raw: {{sensor: acc, index: 2, unit: "m/s²"}}
    acc_comp: {{sensor: acc, index: 2, unit: "m/s²", gravity_compensated: true}}
    quat_x: {{sensor: quat, index: 1}}
    quat_y: {{sensor: quat, index: 2}}
"""


def write_tippy(root: Path, hold: bool, rows: int = 300) -> Path:
    (root / "tippy.xml").write_text(TIPPY_XML, encoding="utf-8")
    hold_line = "  hold_upright: root" if hold else ""
    (root / "profile.yaml").write_text(
        PROFILE_TEMPLATE.format(hold_line=hold_line), encoding="utf-8"
    )
    run_dir = root / "run"
    run_dir.mkdir(exist_ok=True)
    lines = ["data,t_us,cmd_nm"]
    for i in range(rows):
        lines.append(f"data,{int(i * 1e4)},0.0")
    (run_dir / "telemetry.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def run_replay(tmp_path: Path, hold: bool):
    run_dir = write_tippy(tmp_path, hold=hold)
    profile = load_profile(tmp_path / "profile.yaml")
    real_run = load_run(run_dir, profile)
    return replay_commands(profile, real_run)


def test_gravity_compensation_zeroes_static_reading_even_when_fallen(tmp_path):
    sim = run_replay(tmp_path, hold=False)
    raw_tail = sim.signals["acc_raw"].values[-50:]
    comp_tail = sim.signals["acc_comp"].values[-50:]
    # The body tips over and comes to rest: the raw site-z reading picks up a
    # large gravity component, while the compensated reading stays near zero.
    assert np.abs(raw_tail).max() > 5.0
    assert np.abs(comp_tail).max() < 0.5
    # Sanity: it really did tip (quaternion far from yaw-only).
    assert np.abs(sim.signals["quat_y"].values[-1]) > 0.5


def test_hold_upright_prevents_tipping(tmp_path):
    sim = run_replay(tmp_path, hold=True)
    # Held upright, the orientation must stay yaw-only for the whole run even
    # though this body tips within a second when unheld.
    assert np.abs(sim.signals["quat_x"].values).max() < 0.05
    assert np.abs(sim.signals["quat_y"].values).max() < 0.05


def test_hold_upright_unknown_joint_raises(tmp_path):
    run_dir = write_tippy(tmp_path, hold=False)
    profile_text = (tmp_path / "profile.yaml").read_text()
    (tmp_path / "bad.yaml").write_text(
        profile_text.replace("replay:", "replay:\n  hold_upright: nope")
    )
    bad_profile = load_profile(tmp_path / "bad.yaml")
    good_profile = load_profile(tmp_path / "profile.yaml")
    real_run = load_run(run_dir, good_profile)
    from scopik.datamodel import ScopikError

    with pytest.raises(ScopikError, match="no joint"):
        replay_commands(bad_profile, real_run)


def test_gravity_compensation_rejects_scalar_sensor(tmp_path):
    run_dir = write_tippy(tmp_path, hold=False)
    profile_text = (tmp_path / "profile.yaml").read_text()
    (tmp_path / "bad.yaml").write_text(
        profile_text.replace(
            'acc_comp: {sensor: acc, index: 2, unit: "m/s²", gravity_compensated: true}',
            "acc_comp: {sensor: quat, index: 0, gravity_compensated: true}",
        )
    )
    bad_profile = load_profile(tmp_path / "bad.yaml")
    good_profile = load_profile(tmp_path / "profile.yaml")
    real_run = load_run(run_dir, good_profile)
    from scopik.datamodel import ScopikError

    with pytest.raises(ScopikError, match="accelerometer"):
        replay_commands(bad_profile, real_run)
