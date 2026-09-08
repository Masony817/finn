#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <MoteusTeensy.h>
#include <sh2.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "control_math.h"
#include "lqr_safety_config.h"
#include "lqr_seeded_config.h"

namespace {

// First unsupported-floor LQR bring-up firmware. The robot must be held by an
// operator for ZERO UPRIGHT, convention checking, release, and catch. Active
// control requires a host heartbeat and stops after one short trial.
//
// The balance loop runs every tick and depends on nothing above it. The command
// layer below it shapes a bounded reference and can never reach torque directly;
// docs/codebase-notes.md states the invariants it has to preserve.

constexpr uint32_t kSerialBaud = 115200;
constexpr uint8_t kImuSdaPin = 18;
constexpr uint8_t kImuSclPin = 19;
constexpr uint8_t kBno08xPrimaryAddress = 0x4A;
constexpr uint8_t kBno08xSecondaryAddress = 0x4B;
constexpr uint32_t kI2cClockHz = 100000;
constexpr uint32_t kI2cFallbackClockHz = 400000;
constexpr uint32_t kImuReportIntervalUs = 10000;
constexpr uint32_t kImuFreshTimeoutUs = 100000;
constexpr uint32_t kImuAliveTimeoutUs = 250000;
constexpr uint32_t kImuFirstSampleTimeoutUs = 2000000;

constexpr int8_t kLeftMoteusId = 1;
constexpr int8_t kRightMoteusId = 2;
constexpr uint32_t kCanArbitrationBitrate = 1000000;
constexpr uint16_t kMoteusMinReceiveWaitUs = 3000;
constexpr float kMoteusWatchdogTimeoutS = 0.05f;
constexpr uint8_t kMaxConsecutiveMoteusMisses = 3;

constexpr uint32_t kTelemetryPeriodUs = 10000;
constexpr uint32_t kIdleTelemetryPeriodUs = 1000000;
constexpr uint32_t kIdleStopPeriodUs = 100000;
constexpr uint32_t kMaxControlLagUs = 20000;
constexpr float kRadPerRev = 2.0f * PI;

static_assert(
    FinnLqrSafety::kHardTorqueLimitNm <= FinnLqrSeeded::kTorqueLimitNm,
    "real torque limit must not exceed the validated seeded-model limit");
static_assert(
    FinnLqrSeeded::kControlPeriodUs == 10000UL,
    "this firmware is intentionally reviewed for a 100 Hz controller");
static_assert(
    FinnLqrSafety::kMaxTrialDurationMs <= FinnLqrSafety::kFirstTrialDurationMs,
    "an operator must not be able to select a trial longer than the reviewed one");
static_assert(
    FinnLqrSafety::kControlBudgetUs < FinnLqrSeeded::kControlPeriodUs,
    "the control budget has to fit inside one control period");

float wrapPi(float value) {
  while (value > PI) value -= 2.0f * PI;
  while (value < -PI) value += 2.0f * PI;
  return value;
}

const FinnLqrControl::CommandLimits kCommandLimits = {
    FinnLqrSeeded::kMaxForwardVelMS,
    FinnLqrSeeded::kMaxYawRateRadS,
    FinnLqrSeeded::kDriveAccelLimitMS2,
    FinnLqrSeeded::kDriveYawAccelLimitRadS2,
    FinnLqrSeeded::kCommandTimeoutMs,
};

ACAN_T4FD_Settings can_settings(kCanArbitrationBitrate, DataBitRateFactor::x1);
MoteusTeensyCanFD can_bus(ACAN_T4::can3, can_settings);

Moteus::Options moteusOptions(const int8_t id) {
  Moteus::Options options;
  options.id = id;
  options.source = 0;
  options.disable_brs = true;
  options.default_query = true;
  options.min_rcv_wait_us = kMoteusMinReceiveWaitUs;
  options.query_format.mode = Moteus::kInt8;
  options.query_format.position = Moteus::kFloat;
  options.query_format.velocity = Moteus::kFloat;
  options.query_format.torque = Moteus::kFloat;
  options.query_format.voltage = Moteus::kFloat;
  options.query_format.temperature = Moteus::kFloat;
  options.query_format.fault = Moteus::kInt8;
  return options;
}

Moteus::PositionMode::Format torqueCommandFormat() {
  Moteus::PositionMode::Format format;
  format.position = Moteus::kFloat;
  format.velocity = Moteus::kFloat;
  format.feedforward_torque = Moteus::kFloat;
  format.kp_scale = Moteus::kFloat;
  format.kd_scale = Moteus::kFloat;
  format.maximum_torque = Moteus::kFloat;
  format.watchdog_timeout = Moteus::kFloat;
  return format;
}

Moteus left_moteus(can_bus, moteusOptions(kLeftMoteusId));
Moteus right_moteus(can_bus, moteusOptions(kRightMoteusId));
Moteus::PositionMode::Format torque_format = torqueCommandFormat();
using QueryValues = Moteus::Query::Result;

Adafruit_BNO08x bno08x(-1);
sh2_SensorValue_t imu_event;

struct ImuState {
  bool initialized = false;
  uint8_t address = 0;
  uint32_t clock_hz = 0;
  uint32_t last_any_event_us = 0;
  uint32_t last_quat_us = 0;
  float quat_real = 1.0f;
  float quat_i = 0.0f;
  float quat_j = 0.0f;
  float quat_k = 0.0f;
  float gyro_x_rad_s = 0.0f;
  float gyro_y_rad_s = 0.0f;
  float gyro_z_rad_s = 0.0f;
  float linear_accel_x_m_s2 = 0.0f;
  float linear_accel_y_m_s2 = 0.0f;
  float linear_accel_z_m_s2 = 0.0f;
  uint32_t reset_count = 0;
};

struct MoteusHealth {
  uint8_t consecutive_misses = 0;
};

ImuState imu;
MoteusHealth left_health;
MoteusHealth right_health;

enum class SystemState : uint8_t {
  kSafeIdle,
  kConventionCheck,
  kPreflight,
  kArmedIdle,
  kRunningLqr,
  kComplete,
  kFault,
};

SystemState state = SystemState::kSafeIdle;
char fault_reason[96] = "";
char serial_line[96] = "";
size_t serial_line_len = 0;

bool pitch_zero_valid = false;
float pitch_zero_raw_rad = 0.0f;
float start_left_pos_rev = 0.0f;
float start_right_pos_rev = 0.0f;
uint32_t state_start_ms = 0;
uint32_t arm_deadline_ms = 0;
uint32_t last_heartbeat_ms = 0;
uint32_t next_control_us = 0;
uint32_t next_telemetry_us = 0;
uint32_t next_idle_stop_us = 0;
uint32_t next_preflight_poll_us = 0;
uint32_t last_control_tick_us = 0;
uint32_t last_control_dt_us = 0;
uint32_t last_tick_duration_us = 0;
uint32_t trial_duration_ms = FinnLqrSafety::kFirstTrialDurationMs;

float last_left_command_nm = 0.0f;
float last_right_command_nm = 0.0f;
float last_balance_tau_raw_nm = 0.0f;
float last_balance_tau_nm = 0.0f;
float last_yaw_tau_nm = 0.0f;
float last_target_forward_vel_m_s = 0.0f;
float reference_forward_pos_m = 0.0f;
bool last_saturated = false;

// ---- Command layer (L2). Bounded reference only; never a torque. ----

// Held, clamped, and slew limited intent. A missing command and a crashed
// command source are the same event here: hold briefly, then ramp to zero.
// Ramping and not stepping is the point -- a step is itself a disturbance the
// balance loop would have to reject.
FinnLqrControl::CommandArbiter arbiter;

// ---- Arming preflight ----

struct PreflightStats {
  uint32_t start_ms = 0;
  uint32_t imu_quat_samples = 0;
  uint32_t imu_gyro_samples = 0;
  uint32_t imu_accel_samples = 0;
  uint32_t imu_reset_start = 0;
  float max_gyro_mag_rad_s = 0.0f;
  float max_quat_norm_error = 0.0f;
  float max_abs_pitch_error_rad = 0.0f;
  uint32_t poll_ticks = 0;
  uint32_t left_replies = 0;
  uint32_t right_replies = 0;
  uint32_t left_misses = 0;
  uint32_t right_misses = 0;
  uint8_t left_fault = 0;
  uint8_t right_fault = 0;
  float min_voltage_v = 1.0e9f;
  float max_voltage_v = -1.0e9f;
  float max_temp_c = -1.0e9f;
  float max_abs_wheel_vel_rev_s = 0.0f;
  bool encoders_finite = true;
  uint32_t telemetry_rows = 0;
  uint32_t max_tick_duration_us = 0;
};

PreflightStats preflight;
bool preflight_arms_on_pass = false;
uint8_t preflight_failures = 0;
uint8_t last_preflight_failures = 0;
bool last_preflight_valid = false;

const char* stateName() {
  switch (state) {
    case SystemState::kSafeIdle: return "safe_idle";
    case SystemState::kConventionCheck: return "convention_check";
    case SystemState::kPreflight: return "preflight";
    case SystemState::kArmedIdle: return "armed_idle";
    case SystemState::kRunningLqr: return "running_lqr";
    case SystemState::kComplete: return "complete";
    case SystemState::kFault: return "fault";
  }
  return "unknown";
}

const char* phaseName() {
  switch (state) {
    case SystemState::kConventionCheck: return "convention_check";
    case SystemState::kPreflight: return "preflight";
    case SystemState::kArmedIdle: return "armed_wait";
    case SystemState::kRunningLqr: return "balance";
    case SystemState::kComplete: return "complete";
    case SystemState::kFault: return "fault";
    default: return "idle";
  }
}

bool isRunning() { return state == SystemState::kRunningLqr; }
bool isArmed() { return state == SystemState::kArmedIdle || isRunning(); }

void printEvent(const char* event, const char* detail) {
  Serial.printf(
      "event,%lu,%s,%s,%s\n",
      static_cast<unsigned long>(micros()), event, stateName(), detail);
}

uint32_t imuQuatAgeUs() {
  return imu.last_quat_us == 0 ? UINT32_MAX : micros() - imu.last_quat_us;
}

uint32_t imuAnyAgeUs() {
  return imu.last_any_event_us == 0 ? UINT32_MAX : micros() - imu.last_any_event_us;
}

bool isImuFresh() {
  return imu.initialized && imu.last_quat_us != 0 && imuQuatAgeUs() <= kImuFreshTimeoutUs;
}

bool isImuAlive() {
  return imu.initialized && imu.last_any_event_us != 0 && imuAnyAgeUs() <= kImuAliveTimeoutUs;
}

float sensorXRotationRad() {
  const float w = imu.quat_real;
  const float x = imu.quat_i;
  const float y = imu.quat_j;
  const float z = imu.quat_k;
  return atan2f(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
}

float pitchRad() {
  const float raw = sensorXRotationRad();
  if (!pitch_zero_valid) return FinnLqrSeeded::kPitchSign * raw;
  return FinnLqrSeeded::kPitchSign * wrapPi(raw - pitch_zero_raw_rad);
}

float pitchRateRadS() {
  return FinnLqrSeeded::kPitchSign * imu.gyro_x_rad_s;
}

float yawRateRadS() {
  return FinnLqrSeeded::kYawSign * imu.gyro_y_rad_s;
}

float quatNormError() {
  const float w = imu.quat_real;
  const float x = imu.quat_i;
  const float y = imu.quat_j;
  const float z = imu.quat_k;
  return fabsf(sqrtf(w * w + x * x + y * y + z * z) - 1.0f);
}

float gyroMagnitudeRadS() {
  const float x = imu.gyro_x_rad_s;
  const float y = imu.gyro_y_rad_s;
  const float z = imu.gyro_z_rad_s;
  return sqrtf(x * x + y * y + z * z);
}

float leftForwardPositionRev() {
  const QueryValues& left = left_moteus.last_result().values;
  return FinnLqrSeeded::kRealLeftEncoderForwardSign *
         static_cast<float>(left.position - start_left_pos_rev);
}

float rightForwardPositionRev() {
  const QueryValues& right = right_moteus.last_result().values;
  return FinnLqrSeeded::kRealRightEncoderForwardSign *
         static_cast<float>(right.position - start_right_pos_rev);
}

float forwardPositionM() {
  return 0.5f * (leftForwardPositionRev() + rightForwardPositionRev()) *
         kRadPerRev * FinnLqrSeeded::kWheelRadiusM;
}

float forwardVelocityMS() {
  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  const float left_forward_rev_s = FinnLqrSeeded::kRealLeftEncoderForwardSign *
                                   static_cast<float>(left.velocity);
  const float right_forward_rev_s = FinnLqrSeeded::kRealRightEncoderForwardSign *
                                    static_cast<float>(right.velocity);
  return 0.5f * (left_forward_rev_s + right_forward_rev_s) *
         kRadPerRev * FinnLqrSeeded::kWheelRadiusM;
}

long heartbeatAgeMs() {
  return last_heartbeat_ms == 0 ? -1L : static_cast<long>(millis() - last_heartbeat_ms);
}

Moteus::PositionMode::Command torqueCommand(const float torque_nm) {
  Moteus::PositionMode::Command command;
  command.position = NaN;
  command.velocity = 0.0;
  command.feedforward_torque = FinnLqrControl::clampFloat(
      torque_nm, -FinnLqrSafety::kHardTorqueLimitNm, FinnLqrSafety::kHardTorqueLimitNm);
  command.kp_scale = 0.0;
  command.kd_scale = 0.0;
  command.maximum_torque = FinnLqrSafety::kHardTorqueLimitNm;
  command.watchdog_timeout = kMoteusWatchdogTimeoutS;
  return command;
}

void updateHealth(MoteusHealth& health, const bool ok) {
  if (ok) {
    health.consecutive_misses = 0;
  } else if (health.consecutive_misses < 255) {
    health.consecutive_misses++;
  }
}

bool sendAllStop() {
  last_left_command_nm = 0.0f;
  last_right_command_nm = 0.0f;
  const bool left_ok = left_moteus.SetStop();
  const bool right_ok = right_moteus.SetStop();
  updateHealth(left_health, left_ok);
  updateHealth(right_health, right_ok);
  return left_ok && right_ok;
}

void latchFault(const char* reason) {
  if (state == SystemState::kFault) return;
  strncpy(fault_reason, reason, sizeof(fault_reason) - 1U);
  fault_reason[sizeof(fault_reason) - 1U] = '\0';
  sendAllStop();
  state = SystemState::kFault;
  state_start_ms = millis();
  printEvent("failsafe", fault_reason);
}

bool sendWheelTorques(const float tau_common_nm, const float tau_yaw_nm) {
  if (!FinnLqrControl::isFinite(tau_common_nm) || !FinnLqrControl::isFinite(tau_yaw_nm)) {
    latchFault("non_finite_torque_command");
    return false;
  }
  const float yaw_left = FinnLqrSeeded::kRealLeftActuatorYawSign * tau_yaw_nm;
  const float left_request =
      FinnLqrSeeded::kRealLeftCommonTorqueSign * tau_common_nm + yaw_left;
  const float right_request =
      FinnLqrSeeded::kRealRightCommonTorqueSign * tau_common_nm - yaw_left;
  if (!FinnLqrControl::isFinite(left_request) || !FinnLqrControl::isFinite(right_request)) {
    latchFault("non_finite_torque_command");
    return false;
  }
  const bool left_ok = left_moteus.SetPosition(torqueCommand(left_request), &torque_format);
  const bool right_ok = right_moteus.SetPosition(torqueCommand(right_request), &torque_format);
  updateHealth(left_health, left_ok);
  updateHealth(right_health, right_ok);
  if (left_health.consecutive_misses >= kMaxConsecutiveMoteusMisses) {
    latchFault("left_moteus_no_reply");
    return false;
  }
  if (right_health.consecutive_misses >= kMaxConsecutiveMoteusMisses) {
    latchFault("right_moteus_no_reply");
    return false;
  }
  last_left_command_nm = left_request;
  last_right_command_nm = right_request;
  return left_ok && right_ok;
}

bool configureImuReports() {
  bool ok = true;
  ok = bno08x.enableReport(SH2_ROTATION_VECTOR, kImuReportIntervalUs) && ok;
  ok = bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, kImuReportIntervalUs) && ok;
  ok = bno08x.enableReport(SH2_LINEAR_ACCELERATION, kImuReportIntervalUs) && ok;
  printEvent(ok ? "imu_reports_enabled" : "imu_report_failed", "rv_gyro_linear_accel_100hz");
  return ok;
}

void serviceImu();

bool waitForFirstImuSample() {
  const uint32_t start_us = micros();
  while (!isImuFresh() && micros() - start_us < kImuFirstSampleTimeoutUs) {
    serviceImu();
    delay(1);
  }
  return isImuFresh();
}

bool initImuAtAddress(const uint8_t address) {
  if (!bno08x.begin_I2C(address, &Wire)) return false;
  Wire.setClock(imu.clock_hz);
  if (!configureImuReports()) {
    sh2_close();
    return false;
  }
  imu.initialized = true;
  imu.address = address;
  imu.last_any_event_us = 0;
  imu.last_quat_us = 0;
  if (!waitForFirstImuSample()) {
    imu.initialized = false;
    sh2_close();
    return false;
  }
  return true;
}

bool initImu() {
  Wire.setSDA(kImuSdaPin);
  Wire.setSCL(kImuSclPin);
  Wire.begin();
  const uint32_t clocks[] = {kI2cClockHz, kI2cFallbackClockHz};
  const uint8_t addresses[] = {kBno08xPrimaryAddress, kBno08xSecondaryAddress};
  for (const uint32_t clock_hz : clocks) {
    imu.clock_hz = clock_hz;
    Wire.setClock(clock_hz);
    delay(500);
    for (const uint8_t address : addresses) {
      if (initImuAtAddress(address)) {
        Serial.printf(
            "event,%lu,imu_initialized,address_0x%02X,clock_%lu\n",
            static_cast<unsigned long>(micros()), address,
            static_cast<unsigned long>(clock_hz));
        return true;
      }
    }
  }
  printEvent("imu_failed", "no_fresh_reports_at_0x4A_or_0x4B");
  return false;
}

void serviceImu() {
  if (!imu.initialized) return;
  if (bno08x.wasReset()) {
    imu.reset_count++;
    imu.last_any_event_us = 0;
    imu.last_quat_us = 0;
    if (!configureImuReports() && isRunning()) {
      latchFault("imu_report_reenable_failed");
      return;
    }
    printEvent("imu_reset", "reports_reenabled");
  }

  const bool sampling = state == SystemState::kPreflight;
  for (uint8_t i = 0; i < 8; ++i) {
    if (!bno08x.getSensorEvent(&imu_event)) break;
    const uint32_t event_us = micros();
    imu.last_any_event_us = event_us;
    if (imu_event.sensorId == SH2_ROTATION_VECTOR) {
      const auto& quat = imu_event.un.rotationVector;
      imu.quat_real = quat.real;
      imu.quat_i = quat.i;
      imu.quat_j = quat.j;
      imu.quat_k = quat.k;
      imu.last_quat_us = event_us;
      if (sampling) {
        preflight.imu_quat_samples++;
        const float norm_error = quatNormError();
        if (norm_error > preflight.max_quat_norm_error) {
          preflight.max_quat_norm_error = norm_error;
        }
        const float pitch_error = fabsf(pitchRad() - FinnLqrSeeded::kTargetPitchRad);
        if (pitch_error > preflight.max_abs_pitch_error_rad) {
          preflight.max_abs_pitch_error_rad = pitch_error;
        }
      }
    } else if (imu_event.sensorId == SH2_GYROSCOPE_CALIBRATED) {
      const auto& gyro = imu_event.un.gyroscope;
      imu.gyro_x_rad_s = gyro.x;
      imu.gyro_y_rad_s = gyro.y;
      imu.gyro_z_rad_s = gyro.z;
      if (sampling) {
        preflight.imu_gyro_samples++;
        const float magnitude = gyroMagnitudeRadS();
        if (magnitude > preflight.max_gyro_mag_rad_s) {
          preflight.max_gyro_mag_rad_s = magnitude;
        }
      }
    } else if (imu_event.sensorId == SH2_LINEAR_ACCELERATION) {
      const auto& accel = imu_event.un.linearAcceleration;
      imu.linear_accel_x_m_s2 = accel.x;
      imu.linear_accel_y_m_s2 = accel.y;
      imu.linear_accel_z_m_s2 = accel.z;
      if (sampling) preflight.imu_accel_samples++;
    }
  }
}

bool checkMoteusPreflight() {
  if (!sendAllStop()) return false;
  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  return left.fault == 0 && right.fault == 0;
}

bool checkActiveSafety() {
  if (!isImuAlive()) {
    latchFault("imu_no_reports");
    return false;
  }
  if (!isImuFresh()) {
    latchFault("imu_quaternion_stale");
    return false;
  }
  if (last_heartbeat_ms == 0 || millis() - last_heartbeat_ms > FinnLqrSafety::kHeartbeatTimeoutMs) {
    latchFault("host_heartbeat_timeout");
    return false;
  }

  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  const FinnLqrControl::ActiveControlState active_control_state = {
      imu.quat_real,
      imu.quat_i,
      imu.quat_j,
      imu.quat_k,
      imu.gyro_x_rad_s,
      imu.gyro_y_rad_s,
      pitch_zero_raw_rad,
      static_cast<float>(left.position),
      static_cast<float>(left.velocity),
      static_cast<float>(left.temperature),
      static_cast<float>(right.position),
      static_cast<float>(right.velocity),
      static_cast<float>(right.temperature),
      start_left_pos_rev,
      start_right_pos_rev,
      reference_forward_pos_m,
  };
  if (!FinnLqrControl::activeControlStateIsFinite(active_control_state)) {
    latchFault("non_finite_active_state");
    return false;
  }
  if (fabsf(pitchRad()) > FinnLqrSafety::kMaxAbsPitchRad) {
    latchFault("pitch_limit");
    return false;
  }
  if (left.fault != 0 || right.fault != 0) {
    latchFault("moteus_fault");
    return false;
  }
  if (left.temperature >= FinnLqrSafety::kMaxMoteusTempC ||
      right.temperature >= FinnLqrSafety::kMaxMoteusTempC) {
    latchFault("moteus_overtemp");
    return false;
  }
  if (fabsf(static_cast<float>(left.velocity)) >= FinnLqrSafety::kMaxWheelSpeedRevS ||
      fabsf(static_cast<float>(right.velocity)) >= FinnLqrSafety::kMaxWheelSpeedRevS) {
    latchFault("wheel_speed_limit");
    return false;
  }
  if (fabsf(leftForwardPositionRev()) >= FinnLqrSafety::kMaxWheelTravelRev ||
      fabsf(rightForwardPositionRev()) >= FinnLqrSafety::kMaxWheelTravelRev) {
    latchFault("wheel_travel_limit");
    return false;
  }
  return true;
}

void completeTrial(const char* reason) {
  sendAllStop();
  state = SystemState::kComplete;
  state_start_ms = millis();
  printEvent("lqr_complete", reason);
}

void runControllerTick() {
  if (!isRunning()) return;
  const uint32_t tick_start_us = micros();
  last_control_dt_us =
      last_control_tick_us == 0 ? FinnLqrSeeded::kControlPeriodUs
                                : tick_start_us - last_control_tick_us;
  last_control_tick_us = tick_start_us;
  if (!checkActiveSafety()) return;
  if (millis() - state_start_ms >= trial_duration_ms) {
    completeTrial("trial_timeout_motors_stopped");
    return;
  }

  const float dt_s = static_cast<float>(FinnLqrSeeded::kControlPeriodUs) * 1.0e-6f;
  if (!FinnLqrControl::stepCommandArbiter(&arbiter, millis(), dt_s, kCommandLimits)) {
    latchFault("non_finite_command_state");
    return;
  }

  const float forward_pos_m = forwardPositionM();
  if (FinnLqrSafety::kDriveEnabled) {
    reference_forward_pos_m += arbiter.forward_m_s * dt_s;
    reference_forward_pos_m = FinnLqrControl::clampFloat(
        reference_forward_pos_m,
        forward_pos_m - FinnLqrSeeded::kRefPositionBandM,
        forward_pos_m + FinnLqrSeeded::kRefPositionBandM);
  } else {
    reference_forward_pos_m = 0.0f;
  }

  const float position_error_m = forward_pos_m - reference_forward_pos_m;
  const float position_correction_m_s = FinnLqrControl::clampFloat(
      -FinnLqrSeeded::kPositionHoldKpS * position_error_m,
      -FinnLqrSeeded::kMaxPositionCorrectionMS,
      FinnLqrSeeded::kMaxPositionCorrectionMS);
  last_target_forward_vel_m_s = FinnLqrSeeded::kTargetForwardVelMS +
                                arbiter.forward_m_s + position_correction_m_s;

  const float pitch_error = pitchRad() - FinnLqrSeeded::kTargetPitchRad;
  const float pitch_rate_error = pitchRateRadS();
  const float velocity_error = forwardVelocityMS() - last_target_forward_vel_m_s;
  last_balance_tau_raw_nm = -(
      FinnLqrSeeded::kGainPitch * pitch_error +
      FinnLqrSeeded::kGainPitchRate * pitch_rate_error +
      FinnLqrSeeded::kGainForwardVel * velocity_error);

  const float raw_yaw_tau_nm =
      FinnLqrSafety::kDriveEnabled
          ? -FinnLqrSeeded::kGainYawRate * (yawRateRadS() - arbiter.yaw_rad_s)
          : 0.0f;

  const FinnLqrControl::WheelTorques torque_allocation = FinnLqrControl::allocateWheelTorques(
      last_balance_tau_raw_nm, raw_yaw_tau_nm, FinnLqrSafety::kHardTorqueLimitNm);
  if (!torque_allocation.valid) {
    latchFault("non_finite_control_output");
    return;
  }
  last_balance_tau_nm = torque_allocation.common_nm;
  last_yaw_tau_nm = torque_allocation.yaw_nm;
  last_saturated = fabsf(last_balance_tau_raw_nm - last_balance_tau_nm) > 1.0e-6f;
  sendWheelTorques(last_balance_tau_nm, last_yaw_tau_nm);
  last_tick_duration_us = micros() - tick_start_us;
}

void printTelemetryHeader() {
  Serial.println("schema,lqr_v2");
  Serial.println(
      "data,t_us,state,phase,armed,control_tick_us,control_dt_us,tick_duration_us,"
      "trial_elapsed_ms,trial_duration_ms,heartbeat_age_ms,left_cmd_nm,right_cmd_nm,"
      "left_mode,left_pos_rev,left_vel_rev_s,left_torque_nm,left_voltage_v,left_temp_c,"
      "left_fault,right_mode,right_pos_rev,right_vel_rev_s,right_torque_nm,right_voltage_v,"
      "right_temp_c,right_fault,imu_ok,imu_age_ms,imu_age_us,imu_resets,imu_qr,imu_qi,"
      "imu_qj,imu_qk,imu_gyro_x_rad_s,imu_gyro_y_rad_s,imu_gyro_z_rad_s,"
      "imu_linear_accel_x_m_s2,imu_linear_accel_y_m_s2,imu_linear_accel_z_m_s2,"
      "robot_forward_accel_m_s2,pitch_rad,pitch_rate_rad_s,yaw_rate_rad_s,forward_pos_m,"
      "forward_vel_m_s,target_pitch_rad,target_forward_vel_m_s,ref_pos_m,"
      "cmd_forward_vel_m_s,cmd_yaw_rate_rad_s,command_stale,balance_tau_raw_nm,"
      "balance_tau_nm,tau_yaw_nm,saturated,model_sha256,fault_reason");
}

void printTelemetry() {
  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  const uint32_t imu_age_us = imuQuatAgeUs();
  const long imu_age_ms = imu.last_quat_us == 0 ? -1L : static_cast<long>(imu_age_us / 1000U);
  const uint32_t trial_elapsed_ms = isRunning() ? millis() - state_start_ms : 0U;
  const float forward_accel =
      FinnLqrSeeded::kForwardAccelSign * imu.linear_accel_z_m_s2;
  if (state == SystemState::kPreflight) preflight.telemetry_rows++;

  Serial.printf(
      "data,%lu,%s,%s,%d,%lu,%lu,%lu,%lu,%lu,%ld,%.6f,%.6f,",
      static_cast<unsigned long>(micros()), stateName(), phaseName(), isArmed() ? 1 : 0,
      static_cast<unsigned long>(last_control_tick_us),
      static_cast<unsigned long>(last_control_dt_us),
      static_cast<unsigned long>(last_tick_duration_us),
      static_cast<unsigned long>(trial_elapsed_ms),
      static_cast<unsigned long>(trial_duration_ms), heartbeatAgeMs(),
      static_cast<double>(last_left_command_nm), static_cast<double>(last_right_command_nm));
  Serial.printf(
      "%d,%.7f,%.7f,%.6f,%.3f,%.3f,%d,%d,%.7f,%.7f,%.6f,%.3f,%.3f,%d,%d,%ld,%lu,%lu,",
      static_cast<int>(left.mode), left.position, left.velocity, left.torque,
      left.voltage, left.temperature, static_cast<int>(left.fault),
      static_cast<int>(right.mode), right.position, right.velocity, right.torque,
      right.voltage, right.temperature, static_cast<int>(right.fault),
      isImuAlive() ? 1 : 0, imu_age_ms,
      static_cast<unsigned long>(imu.last_quat_us == 0 ? 0U : imu_age_us),
      static_cast<unsigned long>(imu.reset_count));
  Serial.printf(
      "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,",
      static_cast<double>(imu.quat_real), static_cast<double>(imu.quat_i),
      static_cast<double>(imu.quat_j), static_cast<double>(imu.quat_k),
      static_cast<double>(imu.gyro_x_rad_s), static_cast<double>(imu.gyro_y_rad_s),
      static_cast<double>(imu.gyro_z_rad_s),
      static_cast<double>(imu.linear_accel_x_m_s2),
      static_cast<double>(imu.linear_accel_y_m_s2),
      static_cast<double>(imu.linear_accel_z_m_s2), static_cast<double>(forward_accel));
  Serial.printf(
      "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%d,%.7f,%.7f,%.7f,%d,%s,%s\n",
      static_cast<double>(pitchRad()), static_cast<double>(pitchRateRadS()),
      static_cast<double>(yawRateRadS()), static_cast<double>(forwardPositionM()),
      static_cast<double>(forwardVelocityMS()),
      static_cast<double>(FinnLqrSeeded::kTargetPitchRad),
      static_cast<double>(last_target_forward_vel_m_s),
      static_cast<double>(reference_forward_pos_m),
      static_cast<double>(arbiter.forward_m_s), static_cast<double>(arbiter.yaw_rad_s),
      arbiter.stale ? 1 : 0,
      static_cast<double>(last_balance_tau_raw_nm), static_cast<double>(last_balance_tau_nm),
      static_cast<double>(last_yaw_tau_nm),
      last_saturated ? 1 : 0, FinnLqrSeeded::kModelSha256,
      fault_reason[0] ? fault_reason : "none");
}

void printStatus() {
  Serial.printf(
      "status,state=%s,phase=%s,imu_fresh=%d,pitch_zero_valid=%d,pitch_rad=%.6f,"
      "pitch_sign_verified=%d,wheel_signs_verified=%d,yaw_sign_verified=%d,"
      "drive_enabled=%d,trial_duration_ms=%lu,preflight_failures=%d,heartbeat_age_ms=%ld,"
      "left_misses=%u,right_misses=%u,model_sha256=%s,fault=%s\n",
      stateName(), phaseName(), isImuFresh() ? 1 : 0, pitch_zero_valid ? 1 : 0,
      static_cast<double>(pitchRad()),
      FinnLqrSeeded::kPitchDirectionBenchVerified ? 1 : 0,
      FinnLqrSeeded::kWheelEncoderDirectionsBenchVerified ? 1 : 0,
      FinnLqrSeeded::kYawDirectionBenchVerified ? 1 : 0,
      FinnLqrSafety::kDriveEnabled ? 1 : 0,
      static_cast<unsigned long>(trial_duration_ms),
      last_preflight_valid ? static_cast<int>(last_preflight_failures) : -1,
      heartbeatAgeMs(), left_health.consecutive_misses, right_health.consecutive_misses,
      FinnLqrSeeded::kModelSha256, fault_reason[0] ? fault_reason : "none");
}

void printHelp() {
  Serial.println("# Finn LQR unsupported-floor bring-up");
  Serial.println("# Keep one person ready to catch the robot before every motor stop.");
  Serial.println("# Commands:");
  Serial.println("#   ZERO UPRIGHT       - capture mechanical upright with motors stopped");
  Serial.println("#   CHECK CONVENTIONS  - 15 s motor-disabled sign-check telemetry");
  Serial.println("#   PREFLIGHT          - 2 s motor-disabled system check, never arms");
  Serial.println("#   HEARTBEAT          - required continuously during active control");
  Serial.println("#   TRIAL <ms>         - select trial length inside the reviewed limit");
  Serial.println("#   ARM FINN           - run preflight, then arm only if every check passes");
  Serial.println("#   RUN LQR            - start one heartbeat-guarded balance trial");
  Serial.println("#   DRIVE <v> <w>      - bounded drive intent; refused while drive is gated");
  Serial.println("#   STOP               - immediately stop both motors");
  Serial.println("#   CLEAR              - clear a latched fault after the cause is fixed");
  Serial.println("#   STATUS / HELP      - inspect state or print this text");
}

void printCheck(
    const char* name, const char* result, const double measured, const double limit,
    const char* detail) {
  Serial.printf(
      "check,%lu,%s,%s,%.4f,%.4f,%s\n",
      static_cast<unsigned long>(micros()), name, result, measured, limit, detail);
}

void checkPass(const char* name, const bool ok, const double measured, const double limit,
               const char* detail) {
  printCheck(name, ok ? "pass" : "fail", measured, limit, detail);
  if (!ok) preflight_failures++;
}

void checkSkip(const char* name, const double measured, const char* detail) {
  printCheck(name, "skip", measured, 0.0, detail);
}

void startPreflight(const bool arm_on_pass) {
  if (state == SystemState::kFault) {
    printEvent("preflight_rejected", "fault_latched_send_clear");
    return;
  }
  if (state != SystemState::kSafeIdle && state != SystemState::kComplete) {
    printEvent("preflight_rejected", "must_be_safe_idle");
    return;
  }
  if (!pitch_zero_valid || !isImuFresh()) {
    printEvent("preflight_rejected", "zero_upright_and_fresh_imu_required");
    return;
  }
  sendAllStop();
  preflight = PreflightStats();
  preflight.start_ms = millis();
  preflight.imu_reset_start = imu.reset_count;
  preflight_arms_on_pass = arm_on_pass;
  preflight_failures = 0;
  FinnLqrControl::resetCommandArbiter(&arbiter);
  reference_forward_pos_m = 0.0f;
  start_left_pos_rev = static_cast<float>(left_moteus.last_result().values.position);
  start_right_pos_rev = static_cast<float>(right_moteus.last_result().values.position);
  state = SystemState::kPreflight;
  state_start_ms = preflight.start_ms;
  next_preflight_poll_us = micros();
  // The idle telemetry deadline can sit up to 1 s away; left stale it starves
  // telemetry_rate_hz of half its 2 s window and fails a healthy preflight.
  next_telemetry_us = micros();
  printEvent(
      "preflight_start",
      arm_on_pass ? "hold_finn_still_near_trim_motors_stopped" : "dry_run_will_not_arm");
}

// Sampled at the control rate with the motors commanded stopped, so the CAN
// path, its timing, and the pack are exercised the way the balance loop will
// exercise them rather than probed once.
void servicePreflight() {
  if (state != SystemState::kPreflight) return;
  const uint32_t now_us = micros();
  if (static_cast<int32_t>(now_us - next_preflight_poll_us) < 0) return;
  next_preflight_poll_us += FinnLqrSeeded::kControlPeriodUs;

  const uint32_t poll_start_us = micros();
  const bool left_ok = left_moteus.SetStop();
  const bool right_ok = right_moteus.SetStop();
  const uint32_t duration_us = micros() - poll_start_us;
  updateHealth(left_health, left_ok);
  updateHealth(right_health, right_ok);

  preflight.poll_ticks++;
  if (left_ok) preflight.left_replies++; else preflight.left_misses++;
  if (right_ok) preflight.right_replies++; else preflight.right_misses++;
  if (duration_us > preflight.max_tick_duration_us) {
    preflight.max_tick_duration_us = duration_us;
  }

  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  if (left.fault != 0) preflight.left_fault = static_cast<uint8_t>(left.fault);
  if (right.fault != 0) preflight.right_fault = static_cast<uint8_t>(right.fault);
  for (const double voltage : {left.voltage, right.voltage}) {
    const float value = static_cast<float>(voltage);
    if (value < preflight.min_voltage_v) preflight.min_voltage_v = value;
    if (value > preflight.max_voltage_v) preflight.max_voltage_v = value;
  }
  for (const double temperature : {left.temperature, right.temperature}) {
    const float value = static_cast<float>(temperature);
    if (value > preflight.max_temp_c) preflight.max_temp_c = value;
  }
  for (const double velocity : {left.velocity, right.velocity}) {
    const float value = fabsf(static_cast<float>(velocity));
    if (value > preflight.max_abs_wheel_vel_rev_s) preflight.max_abs_wheel_vel_rev_s = value;
  }
  if (!isfinite(static_cast<float>(left.position)) ||
      !isfinite(static_cast<float>(right.position))) {
    preflight.encoders_finite = false;
  }
}

void evaluatePreflight() {
  const float window_s = static_cast<float>(millis() - preflight.start_ms) * 1.0e-3f;
  const float safe_window_s = window_s > 0.0f ? window_s : 1.0f;
  preflight_failures = 0;

  checkPass(
      "conventions_pitch", FinnLqrSeeded::kPitchDirectionBenchVerified,
      FinnLqrSeeded::kPitchDirectionBenchVerified ? 1.0 : 0.0, 1.0,
      "bench_verify_pitch_then_regenerate_header");
  checkPass(
      "conventions_wheels", FinnLqrSeeded::kWheelEncoderDirectionsBenchVerified,
      FinnLqrSeeded::kWheelEncoderDirectionsBenchVerified ? 1.0 : 0.0, 1.0,
      "bench_verify_wheels_then_regenerate_header");
  printCheck("model_sha256", "pass", 0.0, 0.0, FinnLqrSeeded::kModelSha256);

  checkPass(
      "imu_present", imu.initialized, imu.initialized ? 1.0 : 0.0, 1.0,
      "bno085_initialized");
  checkPass("imu_fresh", isImuFresh(), imuQuatAgeUs() * 1.0e-3, kImuFreshTimeoutUs * 1.0e-3,
            "quaternion_age_ms");
  const float quat_rate_hz = preflight.imu_quat_samples / safe_window_s;
  const float gyro_rate_hz = preflight.imu_gyro_samples / safe_window_s;
  const float accel_rate_hz = preflight.imu_accel_samples / safe_window_s;
  checkPass("imu_quat_rate_hz", quat_rate_hz >= FinnLqrSafety::kMinImuReportRateHz,
            quat_rate_hz, FinnLqrSafety::kMinImuReportRateHz, "rotation_vector_reports");
  checkPass("imu_gyro_rate_hz", gyro_rate_hz >= FinnLqrSafety::kMinImuReportRateHz,
            gyro_rate_hz, FinnLqrSafety::kMinImuReportRateHz, "calibrated_gyro_reports");
  checkPass("imu_accel_rate_hz", accel_rate_hz >= FinnLqrSafety::kMinImuReportRateHz,
            accel_rate_hz, FinnLqrSafety::kMinImuReportRateHz, "linear_accel_reports");
  checkPass("imu_no_reset", imu.reset_count == preflight.imu_reset_start,
            static_cast<double>(imu.reset_count - preflight.imu_reset_start), 0.0,
            "resets_during_preflight");
  checkPass("imu_quat_norm", preflight.max_quat_norm_error <= FinnLqrSafety::kMaxQuatNormError,
            preflight.max_quat_norm_error, FinnLqrSafety::kMaxQuatNormError,
            "worst_quaternion_norm_error");
  checkPass("imu_stationary", preflight.max_gyro_mag_rad_s <= FinnLqrSafety::kMaxPreflightGyroRadS,
            preflight.max_gyro_mag_rad_s, FinnLqrSafety::kMaxPreflightGyroRadS,
            "hold_finn_still_worst_gyro_magnitude");

  checkPass(
      "pitch_zeroed", pitch_zero_valid, pitch_zero_valid ? 1.0 : 0.0, 1.0,
      "zero_upright_captured");
  checkPass(
      "pitch_near_trim",
      preflight.max_abs_pitch_error_rad <= FinnLqrSafety::kMaxStartPitchErrorRad,
      preflight.max_abs_pitch_error_rad, FinnLqrSafety::kMaxStartPitchErrorRad,
      "worst_pitch_error_from_target_rad");

  const bool moteus_data = preflight.left_replies > 0 || preflight.right_replies > 0;
  const float left_reply_hz = preflight.left_replies / safe_window_s;
  const float right_reply_hz = preflight.right_replies / safe_window_s;
  checkPass("moteus_left_link_hz", left_reply_hz >= FinnLqrSafety::kMinMoteusReplyRateHz,
            left_reply_hz, FinnLqrSafety::kMinMoteusReplyRateHz, "query_replies");
  checkPass("moteus_right_link_hz", right_reply_hz >= FinnLqrSafety::kMinMoteusReplyRateHz,
            right_reply_hz, FinnLqrSafety::kMinMoteusReplyRateHz, "query_replies");
  checkPass("moteus_left_misses", preflight.left_misses == 0,
            static_cast<double>(preflight.left_misses), 0.0, "missed_replies");
  checkPass("moteus_right_misses", preflight.right_misses == 0,
            static_cast<double>(preflight.right_misses), 0.0, "missed_replies");
  checkPass("moteus_left_fault", preflight.left_fault == 0,
            static_cast<double>(preflight.left_fault), 0.0, "controller_fault_code");
  checkPass("moteus_right_fault", preflight.right_fault == 0,
            static_cast<double>(preflight.right_fault), 0.0, "controller_fault_code");
  checkPass("moteus_temp_c",
            moteus_data && preflight.max_temp_c <= FinnLqrSafety::kMaxPreflightTempC,
            moteus_data ? preflight.max_temp_c : 0.0, FinnLqrSafety::kMaxPreflightTempC,
            "warmest_controller");

  const float voltage_sag_v = preflight.max_voltage_v - preflight.min_voltage_v;
  checkPass("bus_voltage_stable",
            moteus_data && voltage_sag_v <= FinnLqrSafety::kMaxPreflightVoltageSagV,
            moteus_data ? voltage_sag_v : 0.0, FinnLqrSafety::kMaxPreflightVoltageSagV,
            "sag_with_motors_stopped");
  if (FinnLqrSafety::kMinBusVoltageV > 0.0f) {
    checkPass("bus_voltage_floor",
              moteus_data && preflight.min_voltage_v >= FinnLqrSafety::kMinBusVoltageV,
              moteus_data ? preflight.min_voltage_v : 0.0, FinnLqrSafety::kMinBusVoltageV,
              "lowest_bus_voltage");
  } else {
    checkSkip("bus_voltage_floor", preflight.min_voltage_v, "set_kMinBusVoltageV_from_pack_spec");
  }

  checkPass(
      "encoders_finite", moteus_data && preflight.encoders_finite,
      preflight.encoders_finite ? 1.0 : 0.0, 1.0, "moteus_positions_readable");
  checkPass(
      "wheels_stationary",
      moteus_data && preflight.max_abs_wheel_vel_rev_s <= FinnLqrSafety::kMaxStartWheelSpeedRevS,
      preflight.max_abs_wheel_vel_rev_s, FinnLqrSafety::kMaxStartWheelSpeedRevS,
      "worst_wheel_speed");

  const float telemetry_rate_hz = preflight.telemetry_rows / safe_window_s;
  checkPass("telemetry_rate_hz", telemetry_rate_hz >= FinnLqrSafety::kMinTelemetryRateHz,
            telemetry_rate_hz, FinnLqrSafety::kMinTelemetryRateHz, "rows_emitted_to_host");
  checkPass("control_budget_us", preflight.max_tick_duration_us <= FinnLqrSafety::kControlBudgetUs,
            static_cast<double>(preflight.max_tick_duration_us), FinnLqrSafety::kControlBudgetUs,
            "worst_can_round_trip");
  checkPass(
      "host_heartbeat",
      last_heartbeat_ms != 0 && (millis() - last_heartbeat_ms) <= FinnLqrSafety::kHeartbeatTimeoutMs,
      static_cast<double>(heartbeatAgeMs()), FinnLqrSafety::kHeartbeatTimeoutMs, "host_is_alive");

  last_preflight_failures = preflight_failures;
  last_preflight_valid = true;

  char summary[96];
  snprintf(
      summary, sizeof(summary), "failures_%u_of_window_%.2fs_polls_%lu",
      static_cast<unsigned>(preflight_failures), static_cast<double>(window_s),
      static_cast<unsigned long>(preflight.poll_ticks));
  sendAllStop();

  if (preflight_failures != 0) {
    state = SystemState::kSafeIdle;
    printEvent("preflight_failed", summary);
    return;
  }
  if (!preflight_arms_on_pass) {
    state = SystemState::kSafeIdle;
    printEvent("preflight_passed", summary);
    return;
  }
  state = SystemState::kArmedIdle;
  state_start_ms = millis();
  arm_deadline_ms = state_start_ms + FinnLqrSafety::kArmTimeoutMs;
  printEvent("preflight_passed", summary);
  printEvent("armed", "awaiting_run_lqr");
}

void startConventionCheck() {
  if (state != SystemState::kSafeIdle && state != SystemState::kComplete) {
    printEvent("check_rejected", "must_be_safe_idle");
    return;
  }
  if (!pitch_zero_valid || !isImuFresh()) {
    printEvent("check_rejected", "zero_upright_and_fresh_imu_required");
    return;
  }
  sendAllStop();
  start_left_pos_rev = static_cast<float>(left_moteus.last_result().values.position);
  start_right_pos_rev = static_cast<float>(right_moteus.last_result().values.position);
  state = SystemState::kConventionCheck;
  state_start_ms = millis();
  next_telemetry_us = micros();
  printEvent("convention_check_start", "motors_stopped_tip_and_roll_forward_by_hand");
}

void startLqr() {
  if (state != SystemState::kArmedIdle) {
    printEvent("run_rejected", "not_armed");
    return;
  }
  if (last_heartbeat_ms == 0 || millis() - last_heartbeat_ms > FinnLqrSafety::kHeartbeatTimeoutMs) {
    printEvent("run_rejected", "host_heartbeat_required");
    return;
  }
  if (!isImuFresh() ||
      fabsf(pitchRad() - FinnLqrSeeded::kTargetPitchRad) >
          FinnLqrSafety::kMaxStartPitchErrorRad) {
    printEvent("run_rejected", "fresh_imu_and_near_target_pitch_required");
    return;
  }
  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  if (left.fault != 0 || right.fault != 0 ||
      fabsf(static_cast<float>(left.velocity)) > FinnLqrSafety::kMaxStartWheelSpeedRevS ||
      fabsf(static_cast<float>(right.velocity)) > FinnLqrSafety::kMaxStartWheelSpeedRevS) {
    printEvent("run_rejected", "moteus_healthy_and_stationary_required");
    return;
  }
  start_left_pos_rev = static_cast<float>(left_moteus.last_result().values.position);
  start_right_pos_rev = static_cast<float>(right_moteus.last_result().values.position);
  last_balance_tau_raw_nm = 0.0f;
  last_balance_tau_nm = 0.0f;
  last_yaw_tau_nm = 0.0f;
  last_target_forward_vel_m_s = FinnLqrSeeded::kTargetForwardVelMS;
  reference_forward_pos_m = 0.0f;
  last_control_tick_us = 0;
  last_control_dt_us = 0;
  last_tick_duration_us = 0;
  FinnLqrControl::resetCommandArbiter(&arbiter);
  state = SystemState::kRunningLqr;
  state_start_ms = millis();
  next_control_us = micros();
  next_telemetry_us = micros();
  printEvent("lqr_start", "release_robot_operator_ready_to_catch");
}

void setTrialDuration(const char* argument) {
  char* end = nullptr;
  const long requested_ms = strtol(argument, &end, 10);
  if (end == argument || requested_ms <= 0) {
    printEvent("trial_rejected", "expected_TRIAL_<milliseconds>");
    return;
  }
  if (state != SystemState::kSafeIdle && state != SystemState::kComplete) {
    printEvent("trial_rejected", "must_be_safe_idle");
    return;
  }
  const unsigned long clamped_ms = static_cast<unsigned long>(
      requested_ms < static_cast<long>(FinnLqrSafety::kMinTrialDurationMs)
          ? FinnLqrSafety::kMinTrialDurationMs
          : (requested_ms > static_cast<long>(FinnLqrSafety::kMaxTrialDurationMs)
                 ? FinnLqrSafety::kMaxTrialDurationMs
                 : requested_ms));
  trial_duration_ms = clamped_ms;
  char detail[64];
  snprintf(detail, sizeof(detail), "trial_duration_ms_%lu", clamped_ms);
  printEvent("trial_duration_set", detail);
}

void handleDrive(const char* argument) {
  if (!FinnLqrSafety::kDriveEnabled) {
    printEvent("drive_rejected", "drive_layer_gated_see_kDriveEnabled");
    return;
  }
  if (!FinnLqrSeeded::kYawDirectionBenchVerified) {
    printEvent("drive_rejected", "bench_verify_yaw_then_regenerate_header");
    return;
  }
  char* end = nullptr;
  const float forward_m_s = strtof(argument, &end);
  if (end == argument) {
    printEvent("drive_rejected", "expected_DRIVE_<forward_m_s>_<yaw_rad_s>");
    return;
  }
  const char* yaw_text = end;
  char* yaw_end = nullptr;
  const float yaw_rad_s = strtof(yaw_text, &yaw_end);
  if (yaw_end == yaw_text) {
    printEvent("drive_rejected", "expected_DRIVE_<forward_m_s>_<yaw_rad_s>");
    return;
  }
  if (!isfinite(forward_m_s) || !isfinite(yaw_rad_s)) {
    printEvent("drive_rejected", "non_finite_command");
    return;
  }
  if (!FinnLqrControl::submitCommand(&arbiter, forward_m_s, yaw_rad_s, millis(), kCommandLimits)) {
    printEvent("drive_rejected", "non_finite_command");
  }
}

void handleCommand(const char* line) {
  if (line[0] == '\0') return;
  if (strcmp(line, "HELP") == 0) {
    printHelp();
  } else if (strcmp(line, "STATUS") == 0) {
    printStatus();
  } else if (strcmp(line, "HEARTBEAT") == 0) {
    last_heartbeat_ms = millis();
  } else if (strcmp(line, "ZERO UPRIGHT") == 0) {
    if ((state != SystemState::kSafeIdle && state != SystemState::kComplete) || !isImuFresh()) {
      printEvent("zero_rejected", "safe_idle_and_fresh_imu_required");
    } else {
      sendAllStop();
      pitch_zero_raw_rad = sensorXRotationRad();
      pitch_zero_valid = true;
      start_left_pos_rev = static_cast<float>(left_moteus.last_result().values.position);
      start_right_pos_rev = static_cast<float>(right_moteus.last_result().values.position);
      state = SystemState::kSafeIdle;
      printEvent("upright_zeroed", "mechanical_upright_reference_captured");
    }
  } else if (strcmp(line, "CHECK CONVENTIONS") == 0) {
    startConventionCheck();
  } else if (strcmp(line, "PREFLIGHT") == 0) {
    startPreflight(false);
  } else if (strcmp(line, "ARM FINN") == 0) {
    startPreflight(true);
  } else if (strcmp(line, "RUN LQR") == 0) {
    startLqr();
  } else if (strncmp(line, "TRIAL ", 6) == 0) {
    setTrialDuration(line + 6);
  } else if (strncmp(line, "DRIVE ", 6) == 0) {
    handleDrive(line + 6);
  } else if (strcmp(line, "STOP") == 0) {
    if (isRunning()) {
      completeTrial("operator_stop_motors_stopped");
    } else {
      sendAllStop();
      FinnLqrControl::resetCommandArbiter(&arbiter);
      if (state == SystemState::kFault) {
        printEvent("stop", "motors_stopped_fault_remains_latched");
      } else {
        state = SystemState::kSafeIdle;
        printEvent("stop", "operator_stop_motors_stopped");
      }
    }
  } else if (strcmp(line, "CLEAR") == 0) {
    sendAllStop();
    FinnLqrControl::resetCommandArbiter(&arbiter);
    fault_reason[0] = '\0';
    state = SystemState::kSafeIdle;
    printEvent("fault_cleared", "safe_idle");
  } else {
    printEvent("unknown_command", line);
  }
}

void serviceSerial() {
  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n' || ch == '\r') {
      serial_line[serial_line_len] = '\0';
      handleCommand(serial_line);
      serial_line_len = 0;
    } else if (serial_line_len + 1U < sizeof(serial_line)) {
      serial_line[serial_line_len++] = ch;
    } else {
      serial_line_len = 0;
      printEvent("serial_error", "line_too_long");
    }
  }
}

void serviceStateTimeouts() {
  if (state == SystemState::kConventionCheck &&
      millis() - state_start_ms >= FinnLqrSafety::kConventionCheckDurationMs) {
    sendAllStop();
    state = SystemState::kSafeIdle;
    printEvent("convention_check_complete", "review_signs_before_enabling_lqr");
  }
  if (state == SystemState::kPreflight &&
      millis() - preflight.start_ms >= FinnLqrSafety::kPreflightDurationMs) {
    evaluatePreflight();
  }
  if (state == SystemState::kArmedIdle &&
      static_cast<int32_t>(millis() - arm_deadline_ms) >= 0) {
    sendAllStop();
    state = SystemState::kSafeIdle;
    printEvent("arm_timeout", "motors_stopped_rearm_required");
  }
}

void serviceIdleStop() {
  if (isRunning() || state == SystemState::kPreflight) return;
  const uint32_t now_us = micros();
  if (static_cast<int32_t>(now_us - next_idle_stop_us) < 0) return;
  next_idle_stop_us = now_us + kIdleStopPeriodUs;
  sendAllStop();
}

void serviceLed() {
  static uint32_t next_led_ms = 0;
  static bool led_on = false;
  uint32_t period_ms = 1000;
  if (state == SystemState::kConventionCheck) period_ms = 400;
  if (state == SystemState::kPreflight) period_ms = 300;
  if (state == SystemState::kArmedIdle) period_ms = 200;
  if (state == SystemState::kRunningLqr) period_ms = 60;
  if (state == SystemState::kFault) period_ms = 100;
  if (static_cast<int32_t>(millis() - next_led_ms) >= 0) {
    next_led_ms = millis() + period_ms;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }
}

void initCanAndMoteus() {
  const uint32_t error_code = ACAN_T4::can3.beginFD(can_settings);
  if (error_code != 0) {
    char reason[96];
    snprintf(reason, sizeof(reason), "can3_beginfd_error_0x%lx", static_cast<unsigned long>(error_code));
    latchFault(reason);
    return;
  }
  sendAllStop();
  printEvent("moteus_stop_sent", "boot_safe_state");
}

}  // namespace

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(kSerialBaud);
  const uint32_t serial_start_ms = millis();
  while (!Serial && millis() - serial_start_ms < 3000U) {
  }

  Serial.println();
  Serial.println("# Finn real-robot LQR bring-up firmware");
  printTelemetryHeader();
  printHelp();
  initImu();
  initCanAndMoteus();
  if (!FinnLqrSeeded::kPitchDirectionBenchVerified ||
      !FinnLqrSeeded::kWheelEncoderDirectionsBenchVerified) {
    printEvent("safety_gate", "active_lqr_disabled_until_pitch_and_wheel_signs_are_verified");
  }
  if (!FinnLqrSafety::kDriveEnabled) {
    printEvent("safety_gate", "drive_layer_present_but_gated_yaw_torque_held_at_zero");
  }
  printStatus();
}

void loop() {
  serviceSerial();
  serviceImu();
  servicePreflight();
  serviceStateTimeouts();

  const uint32_t now_us = micros();
  if (isRunning() && static_cast<int32_t>(now_us - next_control_us) >= 0) {
    if (now_us - next_control_us > kMaxControlLagUs) {
      latchFault("control_deadline_missed");
    }
    next_control_us += FinnLqrSeeded::kControlPeriodUs;
    runControllerTick();
  }

  const bool high_rate =
      isRunning() || state == SystemState::kConventionCheck || state == SystemState::kPreflight;
  const uint32_t telemetry_period_us = high_rate ? kTelemetryPeriodUs : kIdleTelemetryPeriodUs;
  if (static_cast<int32_t>(now_us - next_telemetry_us) >= 0) {
    next_telemetry_us = now_us + telemetry_period_us;
    printTelemetry();
  }

  serviceIdleStop();
  serviceLed();
}
