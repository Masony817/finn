#pragma once

// Human-reviewed limits for the first unsupported floor trials. These are kept
// separate from the generated LQR gain so changing the model never silently
// relaxes a real-robot safety boundary.
namespace FinnLqrSafety {
constexpr float kHardTorqueLimitNm = 0.25f;
constexpr float kMaxAbsPitchRad = 15.0f * PI / 180.0f;
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
