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
constexpr unsigned long kConventionCheckDurationMs = 15000UL;
}  // namespace FinnLqrSafety
