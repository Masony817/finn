#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <MoteusTeensy.h>
#include <sh2.h>

#include <math.h>
#include <string.h>

#include "lqr_safety_config.h"
#include "lqr_seeded_config.h"

namespace {

// First unsupported-floor LQR bring-up firmware. The robot must be held by an
// operator for ZERO UPRIGHT, convention checking, release, and catch. Active
// control requires a host heartbeat and stops after one short trial.

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

float clampFloat(const float value, const float lower, const float upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;
  return value;
}

float wrapPi(float value) {
  while (value > PI) value -= 2.0f * PI;
  while (value < -PI) value += 2.0f * PI;
  return value;
}

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
uint32_t last_control_tick_us = 0;

float last_left_command_nm = 0.0f;
float last_right_command_nm = 0.0f;
float last_balance_tau_raw_nm = 0.0f;
float last_balance_tau_nm = 0.0f;
float last_target_forward_vel_m_s = 0.0f;
bool last_saturated = false;

const char* stateName() {
  switch (state) {
    case SystemState::kSafeIdle: return "safe_idle";
    case SystemState::kConventionCheck: return "convention_check";
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
  command.feedforward_torque = clampFloat(
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

bool sendBalanceTorque(const float tau_nm) {
  const float left_request = FinnLqrSeeded::kRealLeftCommonTorqueSign * tau_nm;
  const float right_request = FinnLqrSeeded::kRealRightCommonTorqueSign * tau_nm;
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
    } else if (imu_event.sensorId == SH2_GYROSCOPE_CALIBRATED) {
      const auto& gyro = imu_event.un.gyroscope;
      imu.gyro_x_rad_s = gyro.x;
      imu.gyro_y_rad_s = gyro.y;
      imu.gyro_z_rad_s = gyro.z;
    } else if (imu_event.sensorId == SH2_LINEAR_ACCELERATION) {
      const auto& accel = imu_event.un.linearAcceleration;
      imu.linear_accel_x_m_s2 = accel.x;
      imu.linear_accel_y_m_s2 = accel.y;
      imu.linear_accel_z_m_s2 = accel.z;
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
  if (fabsf(pitchRad()) > FinnLqrSafety::kMaxAbsPitchRad) {
    latchFault("pitch_limit");
    return false;
  }

  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
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
  last_control_tick_us = micros();
  if (!checkActiveSafety()) return;
  if (millis() - state_start_ms >= FinnLqrSafety::kFirstTrialDurationMs) {
    completeTrial("trial_timeout_motors_stopped");
    return;
  }

  const float forward_pos_m = forwardPositionM();
  const float position_correction_m_s = clampFloat(
      -FinnLqrSeeded::kPositionHoldKpS * forward_pos_m,
      -FinnLqrSeeded::kMaxPositionCorrectionMS,
      FinnLqrSeeded::kMaxPositionCorrectionMS);
  last_target_forward_vel_m_s =
      FinnLqrSeeded::kTargetForwardVelMS + position_correction_m_s;

  const float pitch_error = pitchRad() - FinnLqrSeeded::kTargetPitchRad;
  const float pitch_rate_error = pitchRateRadS();
  const float velocity_error = forwardVelocityMS() - last_target_forward_vel_m_s;
  last_balance_tau_raw_nm = -(
      FinnLqrSeeded::kGainPitch * pitch_error +
      FinnLqrSeeded::kGainPitchRate * pitch_rate_error +
      FinnLqrSeeded::kGainForwardVel * velocity_error);
  last_balance_tau_nm = clampFloat(
      last_balance_tau_raw_nm,
      -FinnLqrSafety::kHardTorqueLimitNm,
      FinnLqrSafety::kHardTorqueLimitNm);
  last_saturated = fabsf(last_balance_tau_raw_nm - last_balance_tau_nm) > 1.0e-6f;
  sendBalanceTorque(last_balance_tau_nm);
}

void printTelemetryHeader() {
  Serial.println("schema,lqr_v1");
  Serial.println(
      "data,t_us,state,phase,armed,control_tick_us,trial_elapsed_ms,heartbeat_age_ms,"
      "left_cmd_nm,right_cmd_nm,left_mode,left_pos_rev,left_vel_rev_s,left_torque_nm,"
      "left_voltage_v,left_temp_c,left_fault,right_mode,right_pos_rev,right_vel_rev_s,"
      "right_torque_nm,right_voltage_v,right_temp_c,right_fault,imu_ok,imu_age_ms,"
      "imu_qr,imu_qi,imu_qj,imu_qk,imu_gyro_x_rad_s,imu_gyro_y_rad_s,imu_gyro_z_rad_s,"
      "imu_linear_accel_x_m_s2,imu_linear_accel_y_m_s2,imu_linear_accel_z_m_s2,"
      "robot_forward_accel_m_s2,pitch_rad,pitch_rate_rad_s,yaw_rate_rad_s,forward_pos_m,"
      "forward_vel_m_s,target_pitch_rad,target_forward_vel_m_s,balance_tau_raw_nm,"
      "balance_tau_nm,saturated,model_sha256,fault_reason");
}

void printTelemetry() {
  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  const long imu_age_ms = imu.last_quat_us == 0 ? -1L : static_cast<long>(imuQuatAgeUs() / 1000U);
  const uint32_t trial_elapsed_ms = isRunning() ? millis() - state_start_ms : 0U;
  const float forward_accel =
      FinnLqrSeeded::kForwardAccelSign * imu.linear_accel_z_m_s2;

  Serial.printf(
      "data,%lu,%s,%s,%d,%lu,%lu,%ld,%.6f,%.6f,%d,%.7f,%.7f,%.6f,%.3f,%.3f,%d,"
      "%d,%.7f,%.7f,%.6f,%.3f,%.3f,%d,%d,%ld,%.7f,%.7f,%.7f,%.7f,",
      static_cast<unsigned long>(micros()), stateName(), phaseName(), isArmed() ? 1 : 0,
      static_cast<unsigned long>(last_control_tick_us),
      static_cast<unsigned long>(trial_elapsed_ms), heartbeatAgeMs(),
      static_cast<double>(last_left_command_nm), static_cast<double>(last_right_command_nm),
      static_cast<int>(left.mode), left.position, left.velocity, left.torque,
      left.voltage, left.temperature, static_cast<int>(left.fault),
      static_cast<int>(right.mode), right.position, right.velocity, right.torque,
      right.voltage, right.temperature, static_cast<int>(right.fault),
      isImuAlive() ? 1 : 0, imu_age_ms,
      static_cast<double>(imu.quat_real), static_cast<double>(imu.quat_i),
      static_cast<double>(imu.quat_j), static_cast<double>(imu.quat_k));
  Serial.printf(
      "%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,"
      "%.7f,%.7f,%d,%s,%s\n",
      static_cast<double>(imu.gyro_x_rad_s), static_cast<double>(imu.gyro_y_rad_s),
      static_cast<double>(imu.gyro_z_rad_s),
      static_cast<double>(imu.linear_accel_x_m_s2),
      static_cast<double>(imu.linear_accel_y_m_s2),
      static_cast<double>(imu.linear_accel_z_m_s2), static_cast<double>(forward_accel),
      static_cast<double>(pitchRad()), static_cast<double>(pitchRateRadS()),
      static_cast<double>(imu.gyro_y_rad_s), static_cast<double>(forwardPositionM()),
      static_cast<double>(forwardVelocityMS()),
      static_cast<double>(FinnLqrSeeded::kTargetPitchRad),
      static_cast<double>(last_target_forward_vel_m_s),
      static_cast<double>(last_balance_tau_raw_nm), static_cast<double>(last_balance_tau_nm),
      last_saturated ? 1 : 0, FinnLqrSeeded::kModelSha256,
      fault_reason[0] ? fault_reason : "none");
}

void printStatus() {
  Serial.printf(
      "status,state=%s,phase=%s,imu_fresh=%d,pitch_zero_valid=%d,pitch_rad=%.6f,"
      "pitch_sign_verified=%d,wheel_signs_verified=%d,heartbeat_age_ms=%ld,"
      "left_misses=%u,right_misses=%u,"
      "model_sha256=%s,fault=%s\n",
      stateName(), phaseName(), isImuFresh() ? 1 : 0, pitch_zero_valid ? 1 : 0,
      static_cast<double>(pitchRad()),
      FinnLqrSeeded::kPitchDirectionBenchVerified ? 1 : 0,
      FinnLqrSeeded::kWheelEncoderDirectionsBenchVerified ? 1 : 0,
      heartbeatAgeMs(), left_health.consecutive_misses, right_health.consecutive_misses,
      FinnLqrSeeded::kModelSha256, fault_reason[0] ? fault_reason : "none");
}

void printHelp() {
  Serial.println("# Finn LQR unsupported-floor bring-up");
  Serial.println("# Keep one person ready to catch the robot before every motor stop.");
  Serial.println("# Commands:");
  Serial.println("#   ZERO UPRIGHT       - capture mechanical upright with motors stopped");
  Serial.println("#   CHECK CONVENTIONS  - 15 s motor-disabled sign-check telemetry");
  Serial.println("#   HEARTBEAT          - required continuously during active control");
  Serial.println("#   ARM FINN           - arm after zero, sign verification, and preflight");
  Serial.println("#   RUN LQR            - start one heartbeat-guarded 3 s balance trial");
  Serial.println("#   STOP               - immediately stop both motors");
  Serial.println("#   CLEAR              - clear a latched fault after the cause is fixed");
  Serial.println("#   STATUS / HELP      - inspect state or print this text");
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
  printEvent("convention_check_start", "motors_stopped_tip_and_roll_forward_by_hand");
}

void armController() {
  if (state == SystemState::kFault) {
    printEvent("arm_rejected", "fault_latched_send_clear");
    return;
  }
  if (state != SystemState::kSafeIdle && state != SystemState::kComplete) {
    printEvent("arm_rejected", "must_be_safe_idle");
    return;
  }
  if (!FinnLqrSeeded::kPitchDirectionBenchVerified) {
    printEvent("arm_rejected", "bench_verify_pitch_then_update_conventions_and_regenerate_header");
    return;
  }
  if (!FinnLqrSeeded::kWheelEncoderDirectionsBenchVerified) {
    printEvent("arm_rejected", "bench_verify_wheels_then_update_conventions_and_regenerate_header");
    return;
  }
  if (!pitch_zero_valid || !isImuFresh()) {
    printEvent("arm_rejected", "zero_upright_and_fresh_imu_required");
    return;
  }
  if (!checkMoteusPreflight()) {
    printEvent("arm_rejected", "moteus_preflight_failed");
    return;
  }
  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  if (fabsf(pitchRad() - FinnLqrSeeded::kTargetPitchRad) >
      FinnLqrSafety::kMaxStartPitchErrorRad) {
    printEvent("arm_rejected", "hold_nearer_target_pitch");
    return;
  }
  if (fabsf(static_cast<float>(left.velocity)) > FinnLqrSafety::kMaxStartWheelSpeedRevS ||
      fabsf(static_cast<float>(right.velocity)) > FinnLqrSafety::kMaxStartWheelSpeedRevS) {
    printEvent("arm_rejected", "wheels_must_be_stationary");
    return;
  }
  state = SystemState::kArmedIdle;
  state_start_ms = millis();
  arm_deadline_ms = state_start_ms + FinnLqrSafety::kArmTimeoutMs;
  printEvent("armed", "awaiting_run_lqr");
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
  last_target_forward_vel_m_s = FinnLqrSeeded::kTargetForwardVelMS;
  state = SystemState::kRunningLqr;
  state_start_ms = millis();
  next_control_us = micros();
  printEvent("lqr_start", "release_robot_operator_ready_to_catch");
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
  } else if (strcmp(line, "ARM FINN") == 0) {
    armController();
  } else if (strcmp(line, "RUN LQR") == 0) {
    startLqr();
  } else if (strcmp(line, "STOP") == 0) {
    if (isRunning()) {
      completeTrial("operator_stop_motors_stopped");
    } else {
      sendAllStop();
      if (state == SystemState::kFault) {
        printEvent("stop", "motors_stopped_fault_remains_latched");
      } else {
        state = SystemState::kSafeIdle;
        printEvent("stop", "operator_stop_motors_stopped");
      }
    }
  } else if (strcmp(line, "CLEAR") == 0) {
    sendAllStop();
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
  if (state == SystemState::kArmedIdle &&
      static_cast<int32_t>(millis() - arm_deadline_ms) >= 0) {
    sendAllStop();
    state = SystemState::kSafeIdle;
    printEvent("arm_timeout", "motors_stopped_rearm_required");
  }
}

void serviceIdleStop() {
  if (isRunning()) return;
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
  printStatus();
}

void loop() {
  serviceSerial();
  serviceImu();
  serviceStateTimeouts();

  const uint32_t now_us = micros();
  if (isRunning() && static_cast<int32_t>(now_us - next_control_us) >= 0) {
    if (now_us - next_control_us > kMaxControlLagUs) {
      latchFault("control_deadline_missed");
    }
    next_control_us += FinnLqrSeeded::kControlPeriodUs;
    runControllerTick();
  }

  const bool high_rate = isRunning() || state == SystemState::kConventionCheck;
  const uint32_t telemetry_period_us = high_rate ? kTelemetryPeriodUs : kIdleTelemetryPeriodUs;
  if (static_cast<int32_t>(now_us - next_telemetry_us) >= 0) {
    next_telemetry_us = now_us + telemetry_period_us;
    printTelemetry();
  }

  serviceIdleStop();
  serviceLed();
}
