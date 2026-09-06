#pragma once

// Human-reviewed limits for the first unsupported floor trials. These are kept
// separate from the generated LQR gain so changing the model never silently
// relaxes a real-robot safety boundary.
namespace FinnLqrSafety {
// Raised from the 0.25 N*m Batch 1 bench cap on 2026-08-05. That cap was carried
// into the model by accident rather than chosen, and it sits below the torque this
// robot physically needs to stay upright: with both wheels capped there, gravity
// wins past 2.7 degrees of lean. At 1.0 N*m per wheel the sim recovers roughly 11
// degrees, which covers kMaxStartPitchErrorRad with margin, while holding chassis
// acceleration near 3 m/s^2 so a fault cannot launch the robot across the room.
// The hub motors can deliver far more; that authority stays unused for now.
constexpr float kHardTorqueLimitNm = 1.0f;
// Keep the fault limit inside the recoverable envelope. A limit wider than what the
// controller can arrest is not protection, it just lets the robot flail before it
// gives up.
constexpr float kMaxAbsPitchRad = 10.0f * PI / 180.0f;
constexpr float kMaxStartPitchErrorRad = 8.0f * PI / 180.0f;
constexpr float kMaxStartWheelSpeedRevS = 0.15f;
constexpr float kMaxWheelSpeedRevS = 2.5f;
constexpr float kMaxWheelTravelRev = 2.0f;
constexpr float kMaxMoteusTempC = 60.0f;
constexpr unsigned long kArmTimeoutMs = 10000UL;
constexpr unsigned long kHeartbeatTimeoutMs = 300UL;
constexpr unsigned long kFirstTrialDurationMs = 3000UL;
// TRIAL <ms> clamps to this. Shipped equal to kFirstTrialDurationMs so the
// envelope an operator can select is exactly the envelope that was reviewed.
// Raising it is a deliberate decision made from completed short trials, not a
// convenience: every limit below was sized for a 3 s catch.
constexpr unsigned long kMaxTrialDurationMs = kFirstTrialDurationMs;
constexpr unsigned long kMinTrialDurationMs = 500UL;
constexpr unsigned long kConventionCheckDurationMs = 15000UL;

// ---- Arming preflight ----
// ARM FINN spends this window with the motors commanded stopped, sampling every
// subsystem at the control rate before it will arm. It measures the CAN and IMU
// paths the control loop actually uses rather than checking them once, because
// the failures that matter here are intermittent: a marginal I2C pull-up, a CAN
// termination fault, a pack that sags only under query load.
constexpr unsigned long kPreflightDurationMs = 2000UL;
// The BNO085 is configured for 100 Hz reports. Below 80 Hz the estimator is
// running on stale attitude for whole control ticks.
constexpr float kMinImuReportRateHz = 80.0f;
constexpr float kMinTelemetryRateHz = 80.0f;
constexpr float kMinMoteusReplyRateHz = 80.0f;
// The operator holds Finn still through preflight, so anything above this is
// either a shaking hand or a gyro that is not actually reporting rest.
constexpr float kMaxPreflightGyroRadS = 0.15f;
constexpr float kMaxQuatNormError = 0.02f;
// Start cold enough that the 60 C run limit is not reachable inside one trial.
constexpr float kMaxPreflightTempC = 50.0f;
// A stopped-motor bus should not sag at all; movement here means a failing pack,
// a loose lug, or a connector that will drop out under balance current.
constexpr float kMaxPreflightVoltageSagV = 0.5f;
// No bus voltage is specified anywhere in the repo, so this check reports the
// measured value and is skipped rather than guessed at. Set it from the pack
// spec to turn the floor into a gate.
constexpr float kMinBusVoltageV = 0.0f;
// Leaves 2 ms of the 10 ms control period. The two moteus SetPosition calls
// already dominate this budget; a third CAN transaction does not fit.
constexpr uint32_t kControlBudgetUs = 8000UL;

// ---- Drive layer ----
// The command layer below the balance loop is built and tested, but no drive
// command is accepted while this is false, and yaw torque is held at exactly
// zero. Three separate things gate driving and none of them are cleared:
// Finn has not balanced unsupported, yaw_direction_bench_verified is false, and
// kMaxWheelTravelRev faults at about 1.02 m of travel with kFirstTrialDurationMs
// ending the run at 3 s. Driving needs its own reviewed limits, set from balance
// trial evidence. See Stage 3 in docs/real_lqr_bringup.md.
constexpr bool kDriveEnabled = false;
}  // namespace FinnLqrSafety
