"""Live streaming: named scalars pushed while the robot is still moving.

Deliberately smaller than the gap pipeline - no profile, no model, no comparison.
These tests pin the contract a control loop relies on, above all that a live sink
can never interrupt the loop feeding it.
"""

from __future__ import annotations

import pytest
from scopik.live import DEFAULT_COLORS, LiveSession, LiveSignal


def test_from_columns_groups_signals_and_cycles_colors():
    session = LiveSession.from_columns({"attitude": ("pitch_rad", "yaw_rate_rad_s")})

    assert [s.column for s in session.signals] == ["pitch_rad", "yaw_rate_rad_s"]
    assert all(s.group == "attitude" for s in session.signals)
    assert session.signals[0].color == DEFAULT_COLORS[0]
    assert session.signals[1].color == DEFAULT_COLORS[1]


def test_entity_paths_put_a_group_on_one_plot():
    signal = LiveSignal("pitch_rad", "attitude")

    assert signal.entity_path("/live") == "/live/attitude/pitch_rad"


def test_logging_before_a_sink_is_opened_is_a_clear_error():
    session = LiveSession.from_columns({"attitude": ("pitch_rad",)})

    with pytest.raises(RuntimeError, match="spawn"):
        session.log_row(0.0, {"pitch_rad": 0.0})


def test_a_row_missing_a_declared_column_is_skipped_not_raised(tmp_path):
    """A control loop must not fall over because a plot lost a column."""

    session = LiveSession.from_columns({"attitude": ("pitch_rad", "absent_column")})
    session.save(tmp_path / "live.rrd")

    session.log_row(0.0, {"pitch_rad": 0.5})
    session.log_row(0.01, {})
    session.close()


def test_streaming_writes_a_readable_recording(tmp_path):
    path = tmp_path / "live.rrd"
    session = LiveSession.from_columns({"attitude": ("pitch_rad",)}, app_id="test-live")
    session.save(path)

    for tick in range(50):
        session.log_row(tick * 0.01, {"pitch_rad": tick * 0.001})
    session.close()

    assert path.exists() and path.stat().st_size > 0
    assert b"/live/attitude/pitch_rad" in path.read_bytes()
