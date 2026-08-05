#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <MoteusTeensy.h>
#include <sh2.h>

#include <math.h>
#include <string.h>

namespace {

// Finn sysid in a wheels off ground test (batch 01)
// must send "ARM FINN" then "RUN BATCH1" commands for execution.


constexpr uint32_t kSerialBaud = 115200; //usb baud
constexpr uint8_t kImuSdaPin = 18; //teensy i2c pins for the imu
constexpr uint8_t kImuSclPin = 19; // ^^
constexpr uint8_t kBno08xPrimaryAddress = 0x4A; //possible imu addresses over i2c
constexpr uint8_t kBno08xSecondaryAddress = 0x4B; //^^^
constexpr uint32_t kI2cClockHz = 100000; // standard-mode i2c; BNO08x is more reliable here on Teensy 4.1
constexpr uint32_t kI2cFallbackClockHz = 400000; // faster diagnostic fallback
constexpr uint32_t kImuReportIntervalUs = 10000;  // 100hz rotation vector
constexpr uint32_t kImuTimeoutUs = 250000; //if no IMU sample arrives for 250 ms the IMU is considered "stale."
constexpr uint32_t kImuPowerSettleMs = 500;
constexpr uint32_t kImuFirstSampleTimeoutUs = 2000000;
constexpr uint32_t kMaxIgnoredImuEventPrints = 5;

constexpr int8_t kLeftMoteusId = 1; //canid's
constexpr int8_t kRightMoteusId = 2;
constexpr uint32_t kCanArbitrationBitrate = 1000000;
constexpr uint16_t kMoteusMinReceiveWaitUs = 3000; // controller reply timeout
constexpr float kHardTorqueLimitNm = 0.25f; // absolute torque ceiling enforcement
constexpr float kMoteusWatchdogTimeoutS = 0.05f; //50ms self stop without fresh command
constexpr uint8_t kMaxConsecutiveMoteusMisses = 3; // fault limit on missed commands
constexpr float kMaxMoteusTempC = 60.0f; // conservative first-run cutoff.
constexpr uint32_t kMaxBatchElapsedMs = 420000; // top-level guard; scripted max is about 294s.

constexpr uint32_t kControlPeriodUs = 10000;    // 100 Hz command loop.
constexpr uint32_t kTelemetryPeriodUs = 10000;  // 100 Hz CSV log stream while running.
constexpr uint32_t kIdleTelemetryPeriodUs = 1000000;  // 1 Hz idle/status heartbeat.
constexpr uint32_t kIdleStopPeriodUs = 1000000;  // Reassert stop at 1 Hz when idle/faulted.
constexpr float kMotionVelocityRevS = 0.02f;
constexpr float kMotionPositionRev = 0.01f;

// The official mjbots Teensy example uses 1 Mbps arbitration and data rate
// with BRS disabled. That is intentionally conservative for early bringup.
ACAN_T4FD_Settings can_settings(kCanArbitrationBitrate, DataBitRateFactor::x1);
MoteusTeensyCanFD can_bus(ACAN_T4::can3, can_settings); //wraps pins 30-31 in moteus adapter

Moteus::Options moteusOptions(const int8_t id) {
  Moteus::Options options;
  options.id = id;
  options.source = 0;
  options.disable_brs = true;
  options.default_query = true; // ask controller to report state back
  options.min_rcv_wait_us = kMoteusMinReceiveWaitUs;

  options.query_format.mode = Moteus::kInt8; //which reply fields come back + precision
  options.query_format.position = Moteus::kFloat;
  options.query_format.velocity = Moteus::kFloat;
  options.query_format.torque = Moteus::kFloat;
  options.query_format.voltage = Moteus::kFloat;
  options.query_format.temperature = Moteus::kFloat;
  options.query_format.fault = Moteus::kInt8;
  return options;
}

// Fields actually sent in a position command, and their wire precision.
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

Moteus left_moteus(can_bus, moteusOptions(kLeftMoteusId)); //both controllers share the one can bus
Moteus right_moteus(can_bus, moteusOptions(kRightMoteusId));
Moteus::PositionMode::Format torque_format = torqueCommandFormat(); //cached once

Adafruit_BNO08x bno08x(-1); //imu driver, -1 = no reset pin wired
sh2_SensorValue_t imu_event; //reusable buffer the driver fills

struct ImuState { //latest imu reading + health
  bool initialized = false; //did begin + report setup succeed
  uint8_t address = 0; //which i2c addr answered
  uint32_t clock_hz = 0; //which bus speed worked
  uint32_t last_any_event_us = 0;
  uint32_t last_event_us = 0; //timestamp of last sample, also the freshness clock
  float quat_real = 1.0f; //orientation quaternion, 1,0,0,0 = identity
  float quat_i = 0.0f;
  float quat_j = 0.0f;
  float quat_k = 0.0f;
  float accuracy_rad = 0.0f; //reported heading accuracy
  uint32_t event_count = 0;
  uint32_t rotation_event_count = 0;
  uint32_t game_rotation_event_count = 0;
  uint32_t other_event_count = 0;
  uint32_t reset_count = 0; //times the chip spontaneously reset
};

ImuState imu;

struct MoteusHealth { //per controller link health
  uint32_t last_ok_us = 0; //last good reply
  uint8_t consecutive_misses = 0; //missed replies in a row
};

MoteusHealth left_health;
MoteusHealth right_health;

using QueryValues = Moteus::Query::Result; //shorthand for the controller reply struct

enum class SystemState : uint8_t { //master interlock
  kSafeIdle, //boot/safe, no motion
  kArmedIdle, //armed, motion allowed
  kRunningBatch1, //executing the batch
  kComplete, //finished
  kFault, //latched error, needs CLEAR
};

enum class SegmentMode : uint8_t { //what a script step does
  kStop, //hold both stopped
  kTorque, //constant torque for a fixed time
  kTorqueUntilMotion, //constant torque until breakaway/motion is detected
  kSpinupUntilVelocity, //drive until target speed
  kCoastUntilSlow, //release torque, watch it slow down
};

enum class WheelSelect : uint8_t { //which wheel's velocity gates the exit
  kNone,
  kLeft,
  kRight,
};

struct BatchSegment { //one row of the test script
  const char* name;
  SegmentMode mode;
  WheelSelect observed_wheel; //wheel watched for the exit condition
  uint32_t max_duration_ms; //hard time cap
  uint32_t min_duration_ms; //don't check the velocity exit before this
  float left_torque_nm;
  float right_torque_nm;
  float exit_abs_velocity_rev_s; //speed threshold for spinup/coast exit
};

constexpr BatchSegment makeSegment(const char* name, const SegmentMode mode,
                                   const WheelSelect observed_wheel,
                                   const uint32_t max_duration_ms,
                                   const uint32_t min_duration_ms,
                                   const float left_torque_nm,
                                   const float right_torque_nm,
                                   const float exit_abs_velocity_rev_s) {
  return BatchSegment{name, mode, observed_wheel, max_duration_ms, min_duration_ms,
                      left_torque_nm, right_torque_nm, exit_abs_velocity_rev_s};
}

enum class SegmentEndReason : uint8_t {
  kNone,
  kTimeout,
  kMotionReached,
  kSpeedReached,
  kSlowReached,
};

constexpr float kBreakawayTorquesNm[] = {0.04f, 0.06f, 0.08f, 0.10f, 0.12f, 0.14f, 0.16f, 0.18f, 0.20f, 0.22f};
constexpr float kDynamicTorquesNm[] = {0.12f, 0.16f, 0.20f, 0.24f};
constexpr float kCoastTargetsRevS[] = {0.75f, 1.50f, 2.50f, 3.50f};
constexpr size_t kDirectionCount = 4; // left pos, left neg, right pos, right neg
constexpr size_t kSignSegmentCount = kDirectionCount * 2;
constexpr size_t kBreakawayRounds = 3;
constexpr size_t kBreakawaySegmentCount = kBreakawayRounds * kDirectionCount * (sizeof(kBreakawayTorquesNm) / sizeof(kBreakawayTorquesNm[0])) * 2;
constexpr size_t kDynamicSegmentCount = kDirectionCount * (sizeof(kDynamicTorquesNm) / sizeof(kDynamicTorquesNm[0])) * 2;
constexpr size_t kCoastSegmentCount = kDirectionCount * (sizeof(kCoastTargetsRevS) / sizeof(kCoastTargetsRevS[0])) * 3;
constexpr size_t kBatch1SegmentCount = 1 + kSignSegmentCount + kBreakawaySegmentCount + kDynamicSegmentCount + kCoastSegmentCount + 1;

SystemState state = SystemState::kSafeIdle;
size_t active_segment_index = 0; //which segment is running
uint32_t batch_start_ms = 0; //top-level run timeout anchor
uint32_t active_segment_start_ms = 0; //when it started
BatchSegment active_segment = makeSegment(
    "idle", SegmentMode::kStop, WheelSelect::kNone, 0, 0, 0.0f, 0.0f, 0.0f);
char active_segment_name[72] = "idle";
float active_segment_start_left_pos_rev = 0.0f;
float active_segment_start_right_pos_rev = 0.0f;
uint32_t next_control_us = 0; //scheduler deadlines
uint32_t next_telemetry_us = 0;
uint32_t next_idle_stop_us = 0;
char fault_reason[96] = ""; //latched fault text
char serial_line[96] = ""; //incoming command buffer
size_t serial_line_len = 0;
float last_left_command_nm = 0.0f; //last torque sent, for telemetry
float last_right_command_nm = 0.0f;
bool imu_stale_warning_active = false;

float clampFloat(const float value, const float low, const float high) { //min/max clamp
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

void formatFloatTag(const float value, char* out, const size_t out_size) {
  const int scaled = static_cast<int>(roundf(value * 100.0f));
  snprintf(out, out_size, "%dp%02d", scaled / 100, abs(scaled % 100));
}

const char* sideName(const WheelSelect wheel) {
  if (wheel == WheelSelect::kLeft) return "left";
  if (wheel == WheelSelect::kRight) return "right";
  return "none";
}

void directionFromIndex(const size_t direction_index, WheelSelect* wheel, int8_t* sign) {
  switch (direction_index % kDirectionCount) {
    case 0:
      *wheel = WheelSelect::kLeft;
      *sign = 1;
      return;
    case 1:
      *wheel = WheelSelect::kLeft;
      *sign = -1;
      return;
    case 2:
      *wheel = WheelSelect::kRight;
      *sign = 1;
      return;
    default:
      *wheel = WheelSelect::kRight;
      *sign = -1;
      return;
  }
}

BatchSegment buildDirectedSegment(const char* prefix, const WheelSelect wheel, const int8_t sign,
                                  const float magnitude_nm, const SegmentMode mode,
                                  const uint32_t max_duration_ms, const uint32_t min_duration_ms,
                                  const float exit_abs_velocity_rev_s) {
  char tag[24];
  formatFloatTag(magnitude_nm, tag, sizeof(tag));
  const char* dir = sign > 0 ? "pos" : "neg";
  snprintf(active_segment_name, sizeof(active_segment_name), "%s_%s_%s_%s",
           prefix, sideName(wheel), dir, tag);
  const float left_torque = (wheel == WheelSelect::kLeft) ? sign * magnitude_nm : 0.0f;
  const float right_torque = (wheel == WheelSelect::kRight) ? sign * magnitude_nm : 0.0f;
  return makeSegment(active_segment_name, mode, wheel, max_duration_ms, min_duration_ms,
                     left_torque, right_torque, exit_abs_velocity_rev_s);
}

BatchSegment buildSettleSegment(const char* prefix, const WheelSelect wheel, const int8_t sign,
                                const float magnitude, const uint32_t duration_ms) {
  char tag[24];
  formatFloatTag(magnitude, tag, sizeof(tag));
  const char* dir = sign > 0 ? "pos" : "neg";
  snprintf(active_segment_name, sizeof(active_segment_name), "%s_%s_%s_%s_settle",
           prefix, sideName(wheel), dir, tag);
  return makeSegment(active_segment_name, SegmentMode::kStop, wheel, duration_ms, 0,
                     0.0f, 0.0f, 0.0f);
}

BatchSegment batchSegmentAt(const size_t index) {
  if (index == 0) {
    snprintf(active_segment_name, sizeof(active_segment_name), "settle_stop");
    return makeSegment(active_segment_name, SegmentMode::kStop, WheelSelect::kNone,
                       2000, 0, 0.0f, 0.0f, 0.0f);
  }

  size_t local = index - 1;
  if (local < kSignSegmentCount) {
    WheelSelect wheel;
    int8_t sign;
    directionFromIndex(local / 2, &wheel, &sign);
    if ((local % 2) == 0) {
      return buildDirectedSegment("sign", wheel, sign, 0.14f, SegmentMode::kTorqueUntilMotion, 900, 100, 0.0f);
    }
    return buildSettleSegment("sign", wheel, sign, 0.14f, 400);
  }

  local -= kSignSegmentCount;
  if (local < kBreakawaySegmentCount) {
    const size_t pair_index = local / 2;
    const size_t level_count = sizeof(kBreakawayTorquesNm) / sizeof(kBreakawayTorquesNm[0]);
    const size_t round_index = pair_index / (kDirectionCount * level_count);
    const size_t round_local = pair_index % (kDirectionCount * level_count);
    WheelSelect wheel;
    int8_t sign;
    directionFromIndex(round_local / level_count, &wheel, &sign);
    const float torque = kBreakawayTorquesNm[round_local % level_count];
    char prefix[24];
    snprintf(prefix, sizeof(prefix), "breakaway_r%u", static_cast<unsigned>(round_index + 1U));
    if ((local % 2) == 0) {
      return buildDirectedSegment(prefix, wheel, sign, torque, SegmentMode::kTorqueUntilMotion, 700, 100, 0.0f);
    }
    return buildSettleSegment(prefix, wheel, sign, torque, 300);
  }

  local -= kBreakawaySegmentCount;
  if (local < kDynamicSegmentCount) {
    const size_t pair_index = local / 2;
    const size_t level_count = sizeof(kDynamicTorquesNm) / sizeof(kDynamicTorquesNm[0]);
    WheelSelect wheel;
    int8_t sign;
    directionFromIndex(pair_index / level_count, &wheel, &sign);
    const float torque = kDynamicTorquesNm[pair_index % level_count];
    if ((local % 2) == 0) {
      return buildDirectedSegment("dynamic", wheel, sign, torque, SegmentMode::kTorque, 900, 0, 0.0f);
    }
    return buildSettleSegment("dynamic", wheel, sign, torque, 500);
  }

  local -= kDynamicSegmentCount;
  if (local < kCoastSegmentCount) {
    const size_t triple_index = local / 3;
    const size_t step_in_triple = local % 3;
    const size_t target_count = sizeof(kCoastTargetsRevS) / sizeof(kCoastTargetsRevS[0]);
    WheelSelect wheel;
    int8_t sign;
    directionFromIndex(triple_index / target_count, &wheel, &sign);
    const float target = kCoastTargetsRevS[triple_index % target_count];
    char tag[24];
    formatFloatTag(target, tag, sizeof(tag));
    const char* dir = sign > 0 ? "pos" : "neg";
    if (step_in_triple == 0) {
      snprintf(active_segment_name, sizeof(active_segment_name), "coast_spinup_%s_%s_%srps",
               sideName(wheel), dir, tag);
      const float left_torque = (wheel == WheelSelect::kLeft) ? sign * 0.24f : 0.0f;
      const float right_torque = (wheel == WheelSelect::kRight) ? sign * 0.24f : 0.0f;
      return makeSegment(active_segment_name, SegmentMode::kSpinupUntilVelocity, wheel,
                         3500, 300, left_torque, right_torque, target);
    }
    if (step_in_triple == 1) {
      snprintf(active_segment_name, sizeof(active_segment_name), "coastdown_%s_%s_%srps",
               sideName(wheel), dir, tag);
      return makeSegment(active_segment_name, SegmentMode::kCoastUntilSlow, wheel,
                         5000, 500, 0.0f, 0.0f, 0.05f);
    }
    snprintf(active_segment_name, sizeof(active_segment_name), "coastdown_%s_%s_%srps_settle",
             sideName(wheel), dir, tag);
    return makeSegment(active_segment_name, SegmentMode::kStop, wheel, 500, 0,
                       0.0f, 0.0f, 0.0f);
  }

  snprintf(active_segment_name, sizeof(active_segment_name), "final_stop");
  return makeSegment(active_segment_name, SegmentMode::kStop, WheelSelect::kNone,
                     2000, 0, 0.0f, 0.0f, 0.0f);
}

const char* stateName() { //enum -> string for logs
  switch (state) {
    case SystemState::kSafeIdle:
      return "safe_idle";
    case SystemState::kArmedIdle:
      return "armed_idle";
    case SystemState::kRunningBatch1:
      return "running_batch1";
    case SystemState::kComplete:
      return "complete";
    case SystemState::kFault:
      return "fault";
  }
  return "unknown";
}

const char* activePhaseName() { //current segment name (or state) for logs
  if (state == SystemState::kRunningBatch1 && active_segment_index < kBatch1SegmentCount) return active_segment.name;
  if (state == SystemState::kComplete) return "complete";
  if (state == SystemState::kFault) return "fault";
  return "idle";
}

bool isRunning() { //true only mid-batch
  return state == SystemState::kRunningBatch1;
}

bool isArmed() { //armed or running
  return state == SystemState::kArmedIdle || state == SystemState::kRunningBatch1;
}

uint32_t imuAgeUs() { //us since last imu sample, max if none yet
  if (imu.last_event_us == 0) return UINT32_MAX;
  return static_cast<uint32_t>(micros() - imu.last_event_us); //unsigned sub, wrap-safe
}

bool isImuFresh() { //up + at least one sample within the timeout
  return imu.initialized && imu.last_event_us != 0 && imuAgeUs() <= kImuTimeoutUs;
}

void printEvent(const char* event, const char* detail) { //one-line csv event record
  Serial.printf("event,%lu,%s,%s,%s\n", static_cast<unsigned long>(micros()), event, stateName(), detail);
}

void printHelp() { //operator command list
  Serial.println("# Finn Batch 1 wheels-off-ground sysid firmware");
  Serial.println("# Commands:");
  Serial.println("#   HELP       - print this text");
  Serial.println("#   STATUS     - print current state");
  Serial.println("#   ARM FINN   - allow a later run command");
  Serial.println("#   RUN BATCH1 - start the off-ground wheel batch");
  Serial.println("#   STOP       - immediately stop both controllers and disarm");
  Serial.println("#   CLEAR      - clear a latched fault after the cause is fixed");
  Serial.println("# Capture CSV lines beginning with 'data,' for analysis.");
}

void printTelemetryHeader() { //csv schema tag + column header so the log is self-describing
  Serial.println("schema,batch1_v2");
  Serial.println("data,t_us,state,phase_index,phase,armed,left_cmd_nm,right_cmd_nm,left_mode,left_pos_rev,left_vel_rev_s,left_torque_nm,left_voltage_v,left_temp_c,left_fault,right_mode,right_pos_rev,right_vel_rev_s,right_torque_nm,right_voltage_v,right_temp_c,right_fault,imu_ok,imu_age_ms,imu_qr,imu_qi,imu_qj,imu_qk,imu_accuracy_rad,fault_reason");
}

void printStatus() { //human-readable one-shot snapshot
  const long imu_age_ms = (imu.last_event_us == 0) ? -1L : static_cast<long>(imuAgeUs() / 1000U);
  Serial.printf("status,state=%s,phase=%s,imu_initialized=%d,imu_fresh=%d,imu_age_ms=%ld,imu_address=0x%02X,imu_i2c_clock_hz=%lu,imu_events=%lu,imu_rotation_events=%lu,imu_game_rotation_events=%lu,imu_resets=%lu,left_misses=%u,right_misses=%u,fault=%s\n",
                stateName(), activePhaseName(), imu.initialized ? 1 : 0, isImuFresh() ? 1 : 0,
                imu_age_ms, imu.address, static_cast<unsigned long>(imu.clock_hz),
                static_cast<unsigned long>(imu.event_count),
                static_cast<unsigned long>(imu.rotation_event_count),
                static_cast<unsigned long>(imu.game_rotation_event_count),
                static_cast<unsigned long>(imu.reset_count),
                left_health.consecutive_misses, right_health.consecutive_misses,
                fault_reason[0] ? fault_reason : "none");
}

void printTelemetry() { //one csv data row: state + both motors + imu, this is the dataset
  const QueryValues& left = left_moteus.last_result().values;
  const QueryValues& right = right_moteus.last_result().values;
  const long imu_age_ms = (imu.last_event_us == 0) ? -1L : static_cast<long>(imuAgeUs() / 1000U);
  const int phase_index = (state == SystemState::kRunningBatch1) ? static_cast<int>(active_segment_index) : -1;

  Serial.printf(
      "data,%lu,%s,%d,%s,%d,%.5f,%.5f,%d,%.7f,%.7f,%.5f,%.3f,%.3f,%d,%d,%.7f,%.7f,%.5f,%.3f,%.3f,%d,%d,%ld,%.7f,%.7f,%.7f,%.7f,%.5f,%s\n",
      static_cast<unsigned long>(micros()),
      stateName(),
      phase_index,
      activePhaseName(),
      isArmed() ? 1 : 0,
      static_cast<double>(last_left_command_nm), //printf %f wants double
      static_cast<double>(last_right_command_nm),
      static_cast<int>(left.mode),
      left.position,
      left.velocity,
      left.torque,
      left.voltage,
      left.temperature,
      static_cast<int>(left.fault),
      static_cast<int>(right.mode),
      right.position,
      right.velocity,
      right.torque,
      right.voltage,
      right.temperature,
      static_cast<int>(right.fault),
      isImuFresh() ? 1 : 0,
      imu_age_ms,
      static_cast<double>(imu.quat_real),
      static_cast<double>(imu.quat_i),
      static_cast<double>(imu.quat_j),
      static_cast<double>(imu.quat_k),
      static_cast<double>(imu.accuracy_rad),
      fault_reason[0] ? fault_reason : "none");
}

const char* i2cStatusName(const uint8_t status) {
  switch (status) {
    case 0: return "ack";
    case 1: return "data_too_long";
    case 2: return "nack_address";
    case 3: return "nack_data";
    case 4: return "other_error";
    case 5: return "timeout";
    default: return "unknown";
  }
}

uint8_t probeI2cAddress(const uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission();
}

void serviceImu(); //fwd decl, waitForFirstImuSample uses it during setup

bool waitForFirstImuSample(const uint32_t timeout_us) {
  const uint32_t start_us = micros();
  while (!isImuFresh() && static_cast<uint32_t>(micros() - start_us) < timeout_us) {
    serviceImu();
    delay(1);
  }
  return isImuFresh();
}

bool configureImuReport() { //(re)enable the 100hz quaternion reports
  bool ok = true;
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, kImuReportIntervalUs)) {
    printEvent("imu_report_failed", "rotation_vector_100hz");
    ok = false;
  } else {
    printEvent("imu_report_enabled", "rotation_vector_100hz");
  }
  if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, kImuReportIntervalUs)) {
    printEvent("imu_report_failed", "game_rotation_vector_100hz");
    ok = false;
  } else {
    printEvent("imu_report_enabled", "game_rotation_vector_100hz");
  }
  return ok;
}

bool initImuAtAddress(const uint8_t address, const bool probe_first) { //try to bring up the imu at one i2c addr
  char detail[96];
  if (probe_first) {
    const uint8_t probe_status = probeI2cAddress(address);
    snprintf(detail, sizeof(detail), "addr_0x%02X_status_%s_clock_%lu",
             address, i2cStatusName(probe_status), static_cast<unsigned long>(imu.clock_hz));
    printEvent("imu_probe", detail);
    if (probe_status != 0) {
      return false;
    }
  } else {
    snprintf(detail, sizeof(detail), "addr_0x%02X_clock_%lu",
             address, static_cast<unsigned long>(imu.clock_hz));
    printEvent("imu_probe_skipped", detail);
  }

  snprintf(detail, sizeof(detail), "addr_0x%02X_clock_%lu",
           address, static_cast<unsigned long>(imu.clock_hz));
  printEvent("imu_begin_address", detail);
  if (!bno08x.begin_I2C(address, &Wire)) {
    printEvent("imu_begin_failed", detail);
    if (probeI2cAddress(address) == 0) {
      sh2_close();
      printEvent("imu_sh2_close", detail);
    }
    return false;
  }
  Wire.setClock(imu.clock_hz);
  if (!configureImuReport()) {
    sh2_close();
    printEvent("imu_sh2_close", "report_config_failed");
    return false;
  }
  imu.initialized = true;
  imu.address = address;
  imu.last_event_us = 0;
  imu.last_any_event_us = 0;
  imu.event_count = 0;
  imu.rotation_event_count = 0;
  imu.game_rotation_event_count = 0;
  imu.other_event_count = 0;
  Serial.printf("event,%lu,imu_initialized,address_0x%02X,rv_game_rv_100hz\n",
                static_cast<unsigned long>(micros()), address);
  if (!waitForFirstImuSample(kImuFirstSampleTimeoutUs)) {
    printEvent("imu_warning", "no_quaternion_within_2000ms");
    imu.initialized = false;
    sh2_close();
    printEvent("imu_sh2_close", "first_sample_timeout");
    return false;
  } else {
    printEvent("imu_first_sample", "quaternion_fresh");
  }
  return true;
}

bool initImu() { //set up i2c, try both possible addresses
  Wire.setSDA(kImuSdaPin);
  Wire.setSCL(kImuSclPin);
  Wire.begin();

  const uint32_t clocks[] = {kI2cClockHz, kI2cFallbackClockHz};
  const uint8_t addresses[] = {kBno08xPrimaryAddress, kBno08xSecondaryAddress};
  for (const uint32_t clock_hz : clocks) {
    imu.clock_hz = clock_hz;
    Wire.setClock(clock_hz);
    Serial.printf("event,%lu,imu_begin,wire_sda_%u_scl_%u,clock_%lu_settle_%lums\n",
                  static_cast<unsigned long>(micros()), kImuSdaPin, kImuSclPin,
                  static_cast<unsigned long>(clock_hz),
                  static_cast<unsigned long>(kImuPowerSettleMs));
    delay(kImuPowerSettleMs);

    for (const uint8_t address : addresses) {
      const bool probe_first = !(clock_hz == kI2cClockHz && address == kBno08xPrimaryAddress);
      if (initImuAtAddress(address, probe_first)) return true;
    }
  }
  imu.clock_hz = 0;
  printEvent("imu_failed", "not_found_or_no_reports_at_0x4A_0x4B_400k_100k_check_i2c_wiring_address_power_timing");
  return false;
}

void latchFault(const char* reason); //fwd decl, serviceImu needs it

void serviceImu() { //drain imu events each loop, handle resets
  if (!imu.initialized) {
    return;
  }

  if (bno08x.wasReset()) { //chip reset itself
    imu.reset_count++;
    imu.last_event_us = 0;
    imu.last_any_event_us = 0;
    printEvent("imu_warning", isRunning() ? "imu_reset_during_run" : "imu_reset");
    if (!configureImuReport()) {
      imu.initialized = false;
      printEvent("imu_warning", "report_reenable_failed");
      return;
    }
    printEvent("imu_reset", "reports_reenabled");
  }

  for (uint8_t i = 0; i < 4; ++i) { //drain up to 4 queued events, bounds loop time
    if (!bno08x.getSensorEvent(&imu_event)) {
      break;
    }
    const uint32_t event_us = micros();
    imu.event_count++;
    imu.last_any_event_us = event_us;
    if (imu_event.sensorId == SH2_ROTATION_VECTOR) {
      const auto& quat = imu_event.un.rotationVector;
      imu.quat_real = quat.real;
      imu.quat_i = quat.i;
      imu.quat_j = quat.j;
      imu.quat_k = quat.k;
      imu.accuracy_rad = quat.accuracy;
      imu.last_event_us = event_us; //stamp freshness
      imu.rotation_event_count++;
    } else if (imu_event.sensorId == SH2_GAME_ROTATION_VECTOR) {
      const auto& quat = imu_event.un.gameRotationVector;
      imu.quat_real = quat.real;
      imu.quat_i = quat.i;
      imu.quat_j = quat.j;
      imu.quat_k = quat.k;
      imu.last_event_us = event_us; //game rotation is a valid quaternion freshness source
      imu.game_rotation_event_count++;
    } else {
      if (imu.other_event_count < kMaxIgnoredImuEventPrints) {
        char detail[64];
        snprintf(detail, sizeof(detail), "sensor_0x%02X", imu_event.sensorId);
        printEvent("imu_event_ignored", detail);
      }
      imu.other_event_count++;
    }
  }
}

// pure feedforward torque: position + velocity loops disabled, only torque applied
Moteus::PositionMode::Command torqueCommand(const float torque_nm) {
  const float bounded_torque = clampFloat(torque_nm, -kHardTorqueLimitNm, kHardTorqueLimitNm); //last-line clamp
  Moteus::PositionMode::Command command;
  command.position = NaN; //no position target
  command.velocity = 0.0;
  command.feedforward_torque = bounded_torque;
  command.kp_scale = 0.0; //disable position loop
  command.kd_scale = 0.0; //disable velocity loop
  command.maximum_torque = kHardTorqueLimitNm; //hard cap
  command.watchdog_timeout = kMoteusWatchdogTimeoutS; //controller self-stops if no fresh cmd
  return command;
}

bool checkMoteusResult(const char* name, Moteus& controller, MoteusHealth& health, const bool ok) { //track link health + latch faults during a run
  if (ok) {
    health.last_ok_us = micros();
    health.consecutive_misses = 0;
    const QueryValues& result = controller.last_result().values;
    if (isRunning() && result.fault != 0) { //controller reported its own fault
      char reason[96];
      snprintf(reason, sizeof(reason), "%s_moteus_fault_%d", name, static_cast<int>(result.fault));
      latchFault(reason);
      return false;
    }
    return true;
  }

  if (health.consecutive_misses < 255) { //count the miss (saturating)
    health.consecutive_misses++;
  }
  if (isRunning() && health.consecutive_misses >= kMaxConsecutiveMoteusMisses) { //too many in a row
    char reason[96];
    snprintf(reason, sizeof(reason), "%s_moteus_no_reply", name);
    latchFault(reason);
  }
  return false;
}

bool checkMoteusTemperatureLimit() {
  const float left_temp = static_cast<float>(left_moteus.last_result().values.temperature);
  const float right_temp = static_cast<float>(right_moteus.last_result().values.temperature);
  if (isfinite(left_temp) && left_temp >= kMaxMoteusTempC) {
    char reason[96];
    snprintf(reason, sizeof(reason), "left_moteus_overtemp_%.1fC", static_cast<double>(left_temp));
    latchFault(reason);
    return false;
  }
  if (isfinite(right_temp) && right_temp >= kMaxMoteusTempC) {
    char reason[96];
    snprintf(reason, sizeof(reason), "right_moteus_overtemp_%.1fC", static_cast<double>(right_temp));
    latchFault(reason);
    return false;
  }
  return true;
}

bool checkBatchElapsedLimit() {
  const uint32_t elapsed_ms = millis() - batch_start_ms;
  if (elapsed_ms <= kMaxBatchElapsedMs) {
    return true;
  }
  latchFault("batch_elapsed_timeout");
  return false;
}

bool sendLeftStop() { //stop = de-energize the motor
  const bool ok = left_moteus.SetStop();
  checkMoteusResult("left", left_moteus, left_health, ok);
  return ok;
}

bool sendRightStop() {
  const bool ok = right_moteus.SetStop();
  checkMoteusResult("right", right_moteus, right_health, ok);
  return ok;
}

bool sendAllStop() { //stop both
  last_left_command_nm = 0.0f;
  last_right_command_nm = 0.0f;
  const bool left_ok = sendLeftStop();
  const bool right_ok = sendRightStop();
  return left_ok && right_ok;
}

// torqueCommand() already clamps to +/-kHardTorqueLimitNm, so callers pass raw.
bool sendLeftTorque(const float torque_nm) {
  const bool ok = left_moteus.SetPosition(torqueCommand(torque_nm), &torque_format);
  checkMoteusResult("left", left_moteus, left_health, ok);
  return ok;
}

bool sendRightTorque(const float torque_nm) {
  const bool ok = right_moteus.SetPosition(torqueCommand(torque_nm), &torque_format);
  checkMoteusResult("right", right_moteus, right_health, ok);
  return ok;
}

bool sendSegmentCommand(const BatchSegment& segment) { //turn one segment row into motor commands
  last_left_command_nm = 0.0f;
  last_right_command_nm = 0.0f;

  if (segment.mode == SegmentMode::kStop || segment.mode == SegmentMode::kCoastUntilSlow) { //both mean de-energized
    return sendAllStop();
  }

  bool left_ok = true;
  bool right_ok = true;
  if (fabsf(segment.left_torque_nm) > 0.0001f) { //command torque only if nonzero, else stop
    last_left_command_nm = clampFloat(segment.left_torque_nm, -kHardTorqueLimitNm, kHardTorqueLimitNm); //record clamped value for telemetry
    left_ok = sendLeftTorque(last_left_command_nm);
  } else {
    left_ok = sendLeftStop();
  }

  if (fabsf(segment.right_torque_nm) > 0.0001f) {
    last_right_command_nm = clampFloat(segment.right_torque_nm, -kHardTorqueLimitNm, kHardTorqueLimitNm);
    right_ok = sendRightTorque(last_right_command_nm);
  } else {
    right_ok = sendRightStop();
  }

  return left_ok && right_ok;
}

float observedAbsVelocityRevS(const BatchSegment& segment) { //watched wheel's abs speed
  // values.velocity is a double; take the abs in double, then narrow once.
  if (segment.observed_wheel == WheelSelect::kLeft) {
    return static_cast<float>(fabs(left_moteus.last_result().values.velocity));
  }
  if (segment.observed_wheel == WheelSelect::kRight) {
    return static_cast<float>(fabs(right_moteus.last_result().values.velocity));
  }
  return 0.0f;
}

float observedAbsPositionDeltaRev(const BatchSegment& segment) {
  if (segment.observed_wheel == WheelSelect::kLeft) {
    return static_cast<float>(fabs(left_moteus.last_result().values.position - active_segment_start_left_pos_rev));
  }
  if (segment.observed_wheel == WheelSelect::kRight) {
    return static_cast<float>(fabs(right_moteus.last_result().values.position - active_segment_start_right_pos_rev));
  }
  return 0.0f;
}

bool observedMotionReached(const BatchSegment& segment) {
  return observedAbsVelocityRevS(segment) >= kMotionVelocityRevS ||
         observedAbsPositionDeltaRev(segment) >= kMotionPositionRev;
}

const char* segmentEndReasonName(const SegmentEndReason reason) {
  switch (reason) {
    case SegmentEndReason::kNone:
      return "none";
    case SegmentEndReason::kTimeout:
      return "timeout";
    case SegmentEndReason::kMotionReached:
      return "motion_reached";
    case SegmentEndReason::kSpeedReached:
      return "speed_reached";
    case SegmentEndReason::kSlowReached:
      return "slow_reached";
  }
  return "unknown";
}

SegmentEndReason segmentCompletionReason(const BatchSegment& segment) { //exit logic for the active segment
  const uint32_t elapsed_ms = millis() - active_segment_start_ms;
  if (elapsed_ms >= segment.max_duration_ms) { //hit the time cap, always wins
    return SegmentEndReason::kTimeout;
  }
  if (elapsed_ms < segment.min_duration_ms) { //honor the min duration first
    return SegmentEndReason::kNone;
  }

  if (segment.mode == SegmentMode::kTorqueUntilMotion) {
    return observedMotionReached(segment) ? SegmentEndReason::kMotionReached : SegmentEndReason::kNone;
  }

  const float observed_speed = observedAbsVelocityRevS(segment);
  if (segment.mode == SegmentMode::kSpinupUntilVelocity) {
    return (observed_speed >= segment.exit_abs_velocity_rev_s) ? SegmentEndReason::kSpeedReached : SegmentEndReason::kNone;
  }
  if (segment.mode == SegmentMode::kCoastUntilSlow) {
    return (observed_speed <= segment.exit_abs_velocity_rev_s) ? SegmentEndReason::kSlowReached : SegmentEndReason::kNone;
  }
  return SegmentEndReason::kNone; //plain torque/stop just run to max_duration
}

void printSegmentEnd(const BatchSegment& segment, const SegmentEndReason reason) {
  const uint32_t elapsed_ms = millis() - active_segment_start_ms;
  Serial.printf("event,%lu,segment_end,%s,%s,%s,%lu\n",
                static_cast<unsigned long>(micros()), stateName(), segment.name,
                segmentEndReasonName(reason), static_cast<unsigned long>(elapsed_ms));
}

void startSegment(const size_t index) { //begin a segment, stamp the start time
  active_segment_index = index;
  active_segment = batchSegmentAt(active_segment_index);
  active_segment_start_ms = millis();
  if (active_segment_index < kBatch1SegmentCount) {
    active_segment_start_left_pos_rev = left_moteus.last_result().values.position;
    active_segment_start_right_pos_rev = right_moteus.last_result().values.position;
    printEvent("segment_start", active_segment.name);
  }
}

void completeBatch() { //stop everything, mark done
  sendAllStop();
  if (state == SystemState::kFault) return;
  state = SystemState::kComplete;
  active_segment_index = 0;
  last_left_command_nm = 0.0f;
  last_right_command_nm = 0.0f;
  printEvent("batch1_complete", "motors_stopped_state_complete");
}

void advanceSegment() { //next segment, or finish if that was the last
  if (active_segment_index + 1U >= kBatch1SegmentCount) {
    completeBatch();
    return;
  }
  startSegment(active_segment_index + 1U);
}

void latchFault(const char* reason) { //emergency brake: latch + stop + log, sticky until CLEAR
  if (state == SystemState::kFault) {
    return; //already faulted
  }
  strncpy(fault_reason, reason, sizeof(fault_reason) - 1U);
  fault_reason[sizeof(fault_reason) - 1U] = '\0';
  state = SystemState::kFault;
  sendAllStop();
  last_left_command_nm = 0.0f;
  last_right_command_nm = 0.0f;
  printEvent("failsafe", fault_reason);
}

bool preflightForRun() { //motor go/no-go check before any motion
  serviceImu();
  const uint32_t wait_start_us = micros();
  while (!isImuFresh() && static_cast<uint32_t>(micros() - wait_start_us) < 300000U) { //give the imu a short chance to appear
    serviceImu();
    delay(1);
  }

  if (!isImuFresh()) {
    printEvent("imu_warning", "run_continues_without_fresh_imu");
  }

  const bool left_ok = sendLeftStop(); //both controllers must answer a stop
  const bool right_ok = sendRightStop();
  if (!left_ok || !right_ok) {
    printEvent("run_rejected", "moteus_no_reply_to_stop");
    return false;
  }

  if (left_moteus.last_result().values.fault != 0 || right_moteus.last_result().values.fault != 0) { //and report no fault
    printEvent("run_rejected", "moteus_fault_present");
    return false;
  }

  if (!checkMoteusTemperatureLimit()) {
    return false;
  }

  return true;
}

void startBatch1() { //start the run if armed and preflight passes
  if (state != SystemState::kArmedIdle) {
    printEvent("run_rejected", "not_armed");
    return;
  }
  if (!preflightForRun()) {
    return;
  }
  fault_reason[0] = '\0';
  state = SystemState::kRunningBatch1;
  batch_start_ms = millis();
  next_control_us = micros(); //reset the schedulers
  next_telemetry_us = micros();
  startSegment(0);
}

void stopAndDisarm(const char* reason) { //operator STOP -> safe idle
  sendAllStop();
  if (state == SystemState::kFault) return;
  state = SystemState::kSafeIdle;
  active_segment_index = 0;
  fault_reason[0] = '\0';
  printEvent("stop", reason);
}

void trimLine(char* line) { //strip leading/trailing whitespace + cr/lf in place
  while (*line == ' ' || *line == '\t') {
    memmove(line, line + 1, strlen(line));
  }
  size_t len = strlen(line);
  while (len > 0 && (line[len - 1U] == ' ' || line[len - 1U] == '\t' || line[len - 1U] == '\r' || line[len - 1U] == '\n')) {
    line[len - 1U] = '\0';
    len--;
  }
}

void handleCommand(char* line) { //dispatch a serial command
  trimLine(line);
  if (line[0] == '\0') return;

  if (strcmp(line, "HELP") == 0) {
    printHelp();
  } else if (strcmp(line, "STATUS") == 0) {
    printStatus();
  } else if (strcmp(line, "ARM FINN") == 0) {
    if (state == SystemState::kFault) {
      printEvent("arm_rejected", "fault_latched_send_clear_first"); //must CLEAR first
      return;
    }
    sendAllStop();
    state = SystemState::kArmedIdle;
    fault_reason[0] = '\0';
    printEvent("armed", "awaiting_run_batch1");
  } else if (strcmp(line, "RUN BATCH1") == 0) {
    startBatch1();
  } else if (strcmp(line, "STOP") == 0) {
    stopAndDisarm("operator_command");
  } else if (strcmp(line, "CLEAR") == 0) {
    if (state == SystemState::kFault) { //only meaningful when faulted
      sendAllStop();
      state = SystemState::kSafeIdle;
      fault_reason[0] = '\0';
      printEvent("fault_cleared", "safe_idle");
    } else {
      printEvent("clear_ignored", "no_fault_latched");
    }
  } else {
    printEvent("unknown_command", line);
  }
}

void serviceSerial() { //accumulate usb bytes into a line, dispatch on newline
  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\n' || ch == '\r') {
      serial_line[serial_line_len] = '\0';
      handleCommand(serial_line);
      serial_line_len = 0;
    } else if (serial_line_len + 1U < sizeof(serial_line)) { //room left in buffer
      serial_line[serial_line_len++] = ch;
    } else {
      serial_line_len = 0; //overflow, drop the line
      printEvent("serial_error", "line_too_long");
    }
  }
}

void runBatchTick() { //one 100hz control step while running
  if (!isRunning()) return;

  if (!isImuFresh()) { //imu went stale mid-run, but Batch 1 motor sysid can continue
    if (!imu_stale_warning_active) {
      printEvent("imu_warning", "imu_timeout_run_continues");
      imu_stale_warning_active = true;
    }
  } else if (imu_stale_warning_active) {
    printEvent("imu_recovered", "fresh_samples_resumed");
    imu_stale_warning_active = false;
  }

  if (active_segment_index >= kBatch1SegmentCount) {
    completeBatch();
    return;
  }
  if (!checkBatchElapsedLimit()) {
    return;
  }

  const BatchSegment& segment = active_segment;
  sendSegmentCommand(segment);

  if (state != SystemState::kRunningBatch1) { //a command may have tripped a fault
    return;
  }
  if (!checkMoteusTemperatureLimit()) {
    return;
  }

  const SegmentEndReason reason = segmentCompletionReason(segment);
  if (reason != SegmentEndReason::kNone) {
    printSegmentEnd(segment, reason);
    advanceSegment();
  }
}

void serviceIdleStop() { //re-assert stop at 10hz whenever not running
  if (isRunning()) return;
  const uint32_t now_us = micros();
  if (static_cast<int32_t>(now_us - next_idle_stop_us) < 0) { //wrap-safe timer
    return;
  }
  next_idle_stop_us = now_us + kIdleStopPeriodUs;
  sendAllStop();
}

void serviceLed() { //blink rate encodes the state
  static uint32_t next_led_ms = 0;
  static bool led_state = false;
  const uint32_t now_ms = millis();
  uint32_t period_ms = 1000; //idle = slow
  if (state == SystemState::kFault) period_ms = 100;
  if (state == SystemState::kArmedIdle) period_ms = 250;
  if (state == SystemState::kRunningBatch1) period_ms = 80; //running = fast
  if (static_cast<int32_t>(now_ms - next_led_ms) >= 0) {
    next_led_ms = now_ms + period_ms;
    led_state = !led_state;
    digitalWrite(LED_BUILTIN, led_state ? HIGH : LOW);
  }
}

void initCanAndMoteus() { //bring up can-fd, send a boot stop
  Serial.printf("event,%lu,can_begin,can3_pins_30_31,arbitration_and_data_%lu\n",
                static_cast<unsigned long>(micros()), static_cast<unsigned long>(kCanArbitrationBitrate));
  const uint32_t error_code = ACAN_T4::can3.beginFD(can_settings);
  if (error_code != 0) { //bus failed to come up
    char reason[96];
    snprintf(reason, sizeof(reason), "can3_beginfd_error_0x%lx", static_cast<unsigned long>(error_code));
    latchFault(reason);
    return;
  }

  // Best effort stop on boot. If controllers are not powered yet, run preflight
  // will reject motion until both answer a stop command without faults.
  sendAllStop();
  printEvent("moteus_stop_sent", "boot_safe_state");
}

}  // namespace

void setup() { //boot: led, serial, imu, can, then sit in safe idle
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(kSerialBaud);
  const uint32_t serial_start_ms = millis();
  while (!Serial && millis() - serial_start_ms < 3000U) { //wait up to 3s for the host
  }

  Serial.println();
  Serial.println("# Finn MCU Batch 1: wheels-off-ground sysid");
  printTelemetryHeader();
  printHelp();

  initImu();
  initCanAndMoteus();
  if (!imu.initialized) { //no imu = degraded logs, not a motor-sysid blocker
    printEvent("imu_warning", "init_failed_run_allowed");
  }
  printStatus();
}

void loop() { //cooperative scheduler, runs flat out
  serviceSerial();
  serviceImu();

  const uint32_t now_us = micros();
  if (isRunning() && static_cast<int32_t>(now_us - next_control_us) >= 0) { //100hz control tick
    next_control_us += kControlPeriodUs;
    runBatchTick();
  }

  const uint32_t telemetry_period_us = isRunning() ? kTelemetryPeriodUs : kIdleTelemetryPeriodUs;
  if (static_cast<int32_t>(now_us - next_telemetry_us) >= 0) { //fast while running, quiet while idle
    next_telemetry_us = now_us + telemetry_period_us;
    printTelemetry();
  }

  serviceIdleStop();
  serviceLed();
}
