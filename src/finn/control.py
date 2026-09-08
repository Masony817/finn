"""Bounded drive references and balance-first torque allocation."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class DriveCommand:
    """The entire vocabulary a command source may speak."""

    forward_vel_m_s: float = 0.0
    yaw_rate_rad_s: float = 0.0

    def is_finite(self) -> bool:
        return math.isfinite(self.forward_vel_m_s) and math.isfinite(self.yaw_rate_rad_s)


@dataclass(frozen=True)
class DriveLimits:
    """Envelope every command is clamped and rate-limited into.

    The acceleration limits are a safety element, not smoothing.  Holding a
    forward acceleration `a` costs a steady lean of atan(a/g), and the reviewed
    firmware pitch fault trips at 10 degrees, so acceleration is physically capped
    at g*tan(10 deg) = 1.73 m/s^2 before Finn faults out.  The 0.5 m/s^2 default
    leans 2.9 degrees and leaves the rest of that budget for disturbance rejection.
    """

    max_forward_vel_m_s: float = 0.6
    # Measured knee: spinning in place, |forward_vel| p95 drift grows from 0.22 to
    # 0.31 m/s between 1.0 and 1.5 rad/s, which the balance loop then has to reject.
    max_yaw_rate_rad_s: float = 1.0
    forward_accel_limit_m_s2: float = 0.5
    yaw_accel_limit_rad_s2: float = 3.0
    command_timeout_s: float = 0.5
    reference_position_band_m: float = 0.30


CommandSource = Callable[[float], DriveCommand | None]
STOPPED = DriveCommand()


@dataclass
class CommandArbiter:
    """Turn intermittent, untrusted intent into a bounded, continuous reference.

    Key release, a dropped serial link, a policy that stops publishing, and a
    policy that raises all arrive here as the same event -- no fresh command --
    and all four ramp to zero through the same slew limiter.  Ramping matters:
    stepping the reference to zero is itself a disturbance the balance loop would
    have to reject.
    """

    limits: DriveLimits
    shaped: DriveCommand = STOPPED
    requested: DriveCommand = STOPPED
    last_good_s: float | None = None
    stale: bool = False
    rejected_samples: int = 0
    _warned: bool = False

    def sample(self, source: CommandSource | None, time_s: float) -> DriveCommand:
        """Sources run synchronously and must return immediately; None means no new sample."""

        if source is None:
            return STOPPED
        try:
            requested = source(time_s)
            valid = isinstance(requested, DriveCommand) and requested.is_finite()
        except Exception:
            valid = False
        if not valid:
            self.rejected_samples += 1
            if not self._warned:
                self._warned = True
                print(
                    "warning: command source returned an unusable value; holding then "
                    "ramping to zero",
                    file=sys.stderr,
                )
            return self._held(time_s)
        self.last_good_s = time_s
        self.stale = False
        self.requested = requested
        return requested

    def step(self, source: CommandSource | None, time_s: float, dt_s: float) -> DriveCommand:
        target = self.sample(source, time_s)
        limits = self.limits
        forward_target = clamp(
            target.forward_vel_m_s, -limits.max_forward_vel_m_s, limits.max_forward_vel_m_s
        )
        yaw_target = clamp(
            target.yaw_rate_rad_s, -limits.max_yaw_rate_rad_s, limits.max_yaw_rate_rad_s
        )
        self.shaped = DriveCommand(
            slew(
                self.shaped.forward_vel_m_s, forward_target, limits.forward_accel_limit_m_s2 * dt_s
            ),
            slew(self.shaped.yaw_rate_rad_s, yaw_target, limits.yaw_accel_limit_rad_s2 * dt_s),
        )
        return self.shaped

    def _held(self, time_s: float) -> DriveCommand:
        """Hold the last accepted command briefly, then command a stop.

        The hold is what decouples a 100 Hz control tick from a policy publishing
        at 10 Hz with jitter: a missing sample is normal, a missing second is not.
        """

        if self.last_good_s is None:
            return STOPPED
        if time_s - self.last_good_s <= self.limits.command_timeout_s:
            return self.requested
        self.stale = True
        return STOPPED


def allocate_wheel_torques(
    tau_balance_nm: float, tau_yaw_nm: float, limit_nm: float
) -> tuple[float, float]:
    """Divide the per-wheel torque envelope between balance and steering.

    Clamping each wheel after summing the two channels would let a turn request
    eat balance authority asymmetrically, which is a fall.  Yaw instead gets only
    the headroom balance left behind, so every wheel command lands inside the
    envelope by construction and a saturated balance loop steers not at all.
    """

    tau_common = clamp(tau_balance_nm, -limit_nm, limit_nm)
    headroom_nm = limit_nm - abs(tau_common)
    return tau_common, clamp(tau_yaw_nm, -headroom_nm, headroom_nm)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def slew(current: float, target: float, max_step: float) -> float:
    return current + clamp(target - current, -max_step, max_step)
