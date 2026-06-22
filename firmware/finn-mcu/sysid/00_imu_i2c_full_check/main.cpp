#include <Arduino.h>
#include <Adafruit_BNO08x.h>
#include <Wire.h>
#include <sh2.h>

namespace {

constexpr uint32_t kSerialBaud = 115200;
constexpr uint8_t kImuSdaPin = 18;
constexpr uint8_t kImuSclPin = 19;
constexpr int8_t kImuResetPin = -1;
constexpr uint8_t kBno08xPrimaryAddress = 0x4A;
constexpr uint8_t kBno08xSecondaryAddress = 0x4B;
constexpr uint32_t kI2cClocksHz[] = {100000, 50000, 400000};
constexpr uint32_t kPowerSettleMs = 3000;
constexpr uint32_t kReportIntervalUs = 10000;
constexpr uint32_t kInitialStreamWindowMs = 5000;
constexpr uint32_t kSummaryPeriodMs = 1000;
constexpr uint32_t kSamplePeriodMs = 100;
constexpr uint8_t kRawPacketDumpMaxBytes = 64;
constexpr uint8_t kRawPacketDumpSamples = 3;
constexpr uint32_t kRawPacketDumpSpacingMs = 20;
constexpr uint8_t kSingleReadDumpBytes = 32;

Adafruit_BNO08x bno08x(kImuResetPin);
sh2_SensorValue_t sensor_value;

bool imu_active = false;
uint8_t active_address = 0;
uint32_t active_clock_hz = 0;
uint32_t last_event_ms = 0;
uint32_t last_summary_ms = 0;
uint32_t last_sample_ms = 0;

struct LatestImu {
  bool rotation_valid = false;
  bool game_rotation_valid = false;
  bool gyro_valid = false;
  bool linear_accel_valid = false;
  bool accel_valid = false;
  bool gravity_valid = false;
  float rv_r = 1.0f;
  float rv_i = 0.0f;
  float rv_j = 0.0f;
  float rv_k = 0.0f;
  float rv_accuracy = 0.0f;
  float game_r = 1.0f;
  float game_i = 0.0f;
  float game_j = 0.0f;
  float game_k = 0.0f;
  float gyro_x = 0.0f;
  float gyro_y = 0.0f;
  float gyro_z = 0.0f;
  float linear_x = 0.0f;
  float linear_y = 0.0f;
  float linear_z = 0.0f;
  float accel_x = 0.0f;
  float accel_y = 0.0f;
  float accel_z = 0.0f;
  float gravity_x = 0.0f;
  float gravity_y = 0.0f;
  float gravity_z = 0.0f;
};

LatestImu latest;

struct ReportSpec {
  sh2_SensorId_t id;
  const char* name;
  bool required;
  bool enabled;
  uint32_t events;
};

ReportSpec reports[] = {
    {SH2_ROTATION_VECTOR, "rotation_vector", true, false, 0},
    {SH2_GAME_ROTATION_VECTOR, "game_rotation_vector", false, false, 0},
    {SH2_GYROSCOPE_CALIBRATED, "gyroscope_calibrated", true, false, 0},
    {SH2_LINEAR_ACCELERATION, "linear_acceleration", true, false, 0},
    {SH2_ACCELEROMETER, "accelerometer", true, false, 0},
    {SH2_GRAVITY, "gravity", false, false, 0},
};

constexpr size_t kReportCount = sizeof(reports) / sizeof(reports[0]);

struct Counters {
  uint32_t total_events = 0;
  uint32_t unknown_events = 0;
  uint32_t resets = 0;
};

Counters counters;

void printEvent(const char* event, const char* detail) {
  Serial.printf("event,%lu,%s,%s\n", static_cast<unsigned long>(millis()), event, detail);
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

uint8_t probeAddress(const uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission();
}

void scanBus(const uint32_t clock_hz) {
  Wire.setClock(clock_hz);
  Serial.printf("event,%lu,i2c_scan_start,clock_%lu\n",
                static_cast<unsigned long>(millis()),
                static_cast<unsigned long>(clock_hz));
  uint8_t found = 0;
  for (uint8_t address = 1; address < 127; ++address) {
    const uint8_t status = probeAddress(address);
    if (status == 0) {
      Serial.printf("event,%lu,i2c_scan_device,addr_0x%02X\n",
                    static_cast<unsigned long>(millis()), address);
      found++;
    }
  }
  Serial.printf("event,%lu,i2c_scan_done,clock_%lu_devices_%u\n",
                static_cast<unsigned long>(millis()),
                static_cast<unsigned long>(clock_hz), found);
}

void rawHeaderProbe(const uint8_t address, const uint32_t clock_hz) {
  Wire.setClock(clock_hz);
  const uint8_t got = Wire.requestFrom(static_cast<int>(address), 4);
  uint8_t bytes[4] = {0, 0, 0, 0};
  for (uint8_t i = 0; i < got && i < 4; ++i) {
    bytes[i] = Wire.read();
  }
  Serial.printf("event,%lu,i2c_raw_read,addr_0x%02X_clock_%lu_got_%u_bytes_%02X_%02X_%02X_%02X\n",
                static_cast<unsigned long>(millis()), address,
                static_cast<unsigned long>(clock_hz), got,
                bytes[0], bytes[1], bytes[2], bytes[3]);
}

void rawPacketProbe(const uint8_t address, const uint32_t clock_hz, const uint8_t sample) {
  Wire.setClock(clock_hz);

  uint8_t header[4] = {0, 0, 0, 0};
  const uint8_t header_got = Wire.requestFrom(static_cast<int>(address), 4);
  for (uint8_t i = 0; i < header_got && i < 4; ++i) {
    header[i] = Wire.read();
  }
  while (Wire.available() > 0) {
    Wire.read();
  }

  const uint16_t raw_length = static_cast<uint16_t>(header[0]) |
                              (static_cast<uint16_t>(header[1]) << 8);
  const bool continuation = (raw_length & 0x8000U) != 0;
  const uint16_t packet_length = raw_length & 0x7FFFU;
  const uint8_t channel = header[2];
  const uint8_t sequence = header[3];

  if (header_got != 4 || packet_length < 4) {
    Serial.printf(
        "event,%lu,i2c_raw_packet_header_invalid,addr_0x%02X_clock_%lu_sample_%u_header_got_%u_len_%u_cont_%u_chan_%u_seq_%u_header_%02X_%02X_%02X_%02X\n",
        static_cast<unsigned long>(millis()), address,
        static_cast<unsigned long>(clock_hz), sample, header_got, packet_length,
        continuation ? 1 : 0, channel, sequence, header[0], header[1],
        header[2], header[3]);
    return;
  }

  const uint8_t read_len = static_cast<uint8_t>(
      packet_length < kRawPacketDumpMaxBytes ? packet_length : kRawPacketDumpMaxBytes);
  uint8_t bytes[kRawPacketDumpMaxBytes] = {};
  const uint8_t got = Wire.requestFrom(static_cast<int>(address),
                                       static_cast<int>(read_len));
  for (uint8_t i = 0; i < got && i < kRawPacketDumpMaxBytes; ++i) {
    bytes[i] = Wire.read();
  }
  while (Wire.available() > 0) {
    Wire.read();
  }

  Serial.printf(
      "event,%lu,i2c_raw_packet,addr_0x%02X_clock_%lu_sample_%u_header_got_%u_len_%u_cont_%u_chan_%u_seq_%u_read_%u_bytes",
      static_cast<unsigned long>(millis()), address,
      static_cast<unsigned long>(clock_hz), sample, header_got, packet_length,
      continuation ? 1 : 0, channel, sequence, got);
  for (uint8_t i = 0; i < got && i < kRawPacketDumpMaxBytes; ++i) {
    Serial.printf("_%02X", bytes[i]);
  }
  Serial.println();
}

void rawSingleReadProbe(const uint8_t address, const uint32_t clock_hz, const uint8_t sample) {
  Wire.setClock(clock_hz);

  uint8_t bytes[kSingleReadDumpBytes] = {};
  const uint8_t got = Wire.requestFrom(static_cast<int>(address),
                                       static_cast<int>(kSingleReadDumpBytes));
  for (uint8_t i = 0; i < got && i < kSingleReadDumpBytes; ++i) {
    bytes[i] = Wire.read();
  }
  while (Wire.available() > 0) {
    Wire.read();
  }

  uint16_t packet_length = 0;
  bool continuation = false;
  uint8_t channel = 0;
  uint8_t sequence = 0;
  if (got >= 4) {
    const uint16_t raw_length = static_cast<uint16_t>(bytes[0]) |
                                (static_cast<uint16_t>(bytes[1]) << 8);
    continuation = (raw_length & 0x8000U) != 0;
    packet_length = raw_length & 0x7FFFU;
    channel = bytes[2];
    sequence = bytes[3];
  }

  Serial.printf(
      "event,%lu,i2c_single_read,addr_0x%02X_clock_%lu_sample_%u_got_%u_len_%u_cont_%u_chan_%u_seq_%u_bytes",
      static_cast<unsigned long>(millis()), address,
      static_cast<unsigned long>(clock_hz), sample, got, packet_length,
      continuation ? 1 : 0, channel, sequence);
  for (uint8_t i = 0; i < got && i < kSingleReadDumpBytes; ++i) {
    Serial.printf("_%02X", bytes[i]);
  }
  Serial.println();
}

void closeSh2AfterFailedBegin(const char* mode, const uint8_t address, const uint32_t clock_hz) {
  sh2_close();
  Serial.printf("event,%lu,sh2_close_after_failed_begin,mode_%s_addr_0x%02X_clock_%lu\n",
                static_cast<unsigned long>(millis()), mode, address,
                static_cast<unsigned long>(clock_hz));
}

ReportSpec* findReport(const sh2_SensorId_t id) {
  for (size_t i = 0; i < kReportCount; ++i) {
    if (reports[i].id == id) return &reports[i];
  }
  return nullptr;
}

void resetReportCounters() {
  for (size_t i = 0; i < kReportCount; ++i) {
    reports[i].events = 0;
  }
  counters.total_events = 0;
  counters.unknown_events = 0;
}

bool configureReports() {
  bool all_required_enabled = true;
  for (size_t i = 0; i < kReportCount; ++i) {
    reports[i].enabled = bno08x.enableReport(reports[i].id, kReportIntervalUs);
    Serial.printf("event,%lu,report_%s,%s_interval_%luus_required_%d\n",
                  static_cast<unsigned long>(millis()),
                  reports[i].enabled ? "enabled" : "failed",
                  reports[i].name, static_cast<unsigned long>(kReportIntervalUs),
                  reports[i].required ? 1 : 0);
    if (reports[i].required && !reports[i].enabled) {
      all_required_enabled = false;
    }
  }
  return all_required_enabled;
}

void printProductIds() {
  Serial.printf("event,%lu,product_ids,num_entries_%u\n",
                static_cast<unsigned long>(millis()), bno08x.prodIds.numEntries);
  for (uint8_t i = 0; i < bno08x.prodIds.numEntries; ++i) {
    const sh2_ProductId_t& entry = bno08x.prodIds.entry[i];
    Serial.printf("event,%lu,product_id,entry_%u_reset_%u_part_%lu_version_%u_%u_%u_build_%lu\n",
                  static_cast<unsigned long>(millis()), i, entry.resetCause,
                  static_cast<unsigned long>(entry.swPartNumber),
                  entry.swVersionMajor, entry.swVersionMinor, entry.swVersionPatch,
                  static_cast<unsigned long>(entry.swBuildNumber));
  }
}

void updateLatestFromEvent(const sh2_SensorValue_t& value) {
  ReportSpec* report = findReport(value.sensorId);
  if (report != nullptr) {
    report->events++;
  } else {
    counters.unknown_events++;
  }

  switch (value.sensorId) {
    case SH2_ROTATION_VECTOR: {
      const auto& q = value.un.rotationVector;
      latest.rotation_valid = true;
      latest.rv_r = q.real;
      latest.rv_i = q.i;
      latest.rv_j = q.j;
      latest.rv_k = q.k;
      latest.rv_accuracy = q.accuracy;
      break;
    }
    case SH2_GAME_ROTATION_VECTOR: {
      const auto& q = value.un.gameRotationVector;
      latest.game_rotation_valid = true;
      latest.game_r = q.real;
      latest.game_i = q.i;
      latest.game_j = q.j;
      latest.game_k = q.k;
      break;
    }
    case SH2_GYROSCOPE_CALIBRATED: {
      const auto& gyro = value.un.gyroscope;
      latest.gyro_valid = true;
      latest.gyro_x = gyro.x;
      latest.gyro_y = gyro.y;
      latest.gyro_z = gyro.z;
      break;
    }
    case SH2_LINEAR_ACCELERATION: {
      const auto& accel = value.un.linearAcceleration;
      latest.linear_accel_valid = true;
      latest.linear_x = accel.x;
      latest.linear_y = accel.y;
      latest.linear_z = accel.z;
      break;
    }
    case SH2_ACCELEROMETER: {
      const auto& accel = value.un.accelerometer;
      latest.accel_valid = true;
      latest.accel_x = accel.x;
      latest.accel_y = accel.y;
      latest.accel_z = accel.z;
      break;
    }
    case SH2_GRAVITY: {
      const auto& gravity = value.un.gravity;
      latest.gravity_valid = true;
      latest.gravity_x = gravity.x;
      latest.gravity_y = gravity.y;
      latest.gravity_z = gravity.z;
      break;
    }
    default:
      Serial.printf("event,%lu,unknown_sensor_event,sensor_0x%02X\n",
                    static_cast<unsigned long>(millis()), value.sensorId);
      break;
  }
}

uint8_t serviceEvents(const uint8_t max_events) {
  uint8_t serviced = 0;
  for (uint8_t i = 0; i < max_events; ++i) {
    if (!bno08x.getSensorEvent(&sensor_value)) {
      break;
    }
    counters.total_events++;
    last_event_ms = millis();
    updateLatestFromEvent(sensor_value);
    serviced++;
  }
  return serviced;
}

bool allRequiredReportsSeen() {
  for (size_t i = 0; i < kReportCount; ++i) {
    if (reports[i].required && reports[i].events == 0) return false;
  }
  return true;
}

bool streamWindowPass(const bool all_required_enabled) {
  if (!all_required_enabled) return false;
  return allRequiredReportsSeen();
}

void printReportCounts(const char* event_name) {
  Serial.printf("event,%lu,%s,total_%lu_unknown_%lu_resets_%lu",
                static_cast<unsigned long>(millis()), event_name,
                static_cast<unsigned long>(counters.total_events),
                static_cast<unsigned long>(counters.unknown_events),
                static_cast<unsigned long>(counters.resets));
  for (size_t i = 0; i < kReportCount; ++i) {
    Serial.printf("_%s_%lu", reports[i].name, static_cast<unsigned long>(reports[i].events));
  }
  Serial.println();
}

void printLatestSample() {
  const long age_ms = (last_event_ms == 0) ? -1L : static_cast<long>(millis() - last_event_ms);
  Serial.printf(
      "sample,%lu,addr_0x%02X,clock_%lu,event_age_ms_%ld,rv_ok_%d,rv_%+.5f_%+.5f_%+.5f_%+.5f,rv_acc_%.5f,game_ok_%d,game_%+.5f_%+.5f_%+.5f_%+.5f,gyro_ok_%d,gyro_%+.5f_%+.5f_%+.5f,linear_ok_%d,linear_%+.5f_%+.5f_%+.5f,accel_ok_%d,accel_%+.5f_%+.5f_%+.5f,gravity_ok_%d,gravity_%+.5f_%+.5f_%+.5f\n",
      static_cast<unsigned long>(millis()), active_address,
      static_cast<unsigned long>(active_clock_hz), age_ms,
      latest.rotation_valid ? 1 : 0,
      static_cast<double>(latest.rv_r), static_cast<double>(latest.rv_i),
      static_cast<double>(latest.rv_j), static_cast<double>(latest.rv_k),
      static_cast<double>(latest.rv_accuracy),
      latest.game_rotation_valid ? 1 : 0,
      static_cast<double>(latest.game_r), static_cast<double>(latest.game_i),
      static_cast<double>(latest.game_j), static_cast<double>(latest.game_k),
      latest.gyro_valid ? 1 : 0,
      static_cast<double>(latest.gyro_x), static_cast<double>(latest.gyro_y),
      static_cast<double>(latest.gyro_z),
      latest.linear_accel_valid ? 1 : 0,
      static_cast<double>(latest.linear_x), static_cast<double>(latest.linear_y),
      static_cast<double>(latest.linear_z),
      latest.accel_valid ? 1 : 0,
      static_cast<double>(latest.accel_x), static_cast<double>(latest.accel_y),
      static_cast<double>(latest.accel_z),
      latest.gravity_valid ? 1 : 0,
      static_cast<double>(latest.gravity_x), static_cast<double>(latest.gravity_y),
      static_cast<double>(latest.gravity_z));
}

bool finishSuccessfulBegin(const uint8_t address, const uint32_t clock_hz) {
  imu_active = true;
  active_address = address;
  active_clock_hz = clock_hz;
  last_event_ms = 0;
  resetReportCounters();

  Serial.printf("event,%lu,begin_i2c_ok,addr_0x%02X_clock_%lu\n",
                static_cast<unsigned long>(millis()), address,
                static_cast<unsigned long>(clock_hz));
  printProductIds();

  const bool all_required_enabled = configureReports();
  const uint32_t window_start_ms = millis();
  while (millis() - window_start_ms < kInitialStreamWindowMs) {
    if (bno08x.wasReset()) {
      counters.resets++;
      printEvent("imu_reset", "during_initial_stream_window");
      configureReports();
    }
    serviceEvents(16);
    delay(1);
  }

  printReportCounts("initial_stream_counts");
  if (streamWindowPass(all_required_enabled)) {
    printEvent("imu_i2c_test_pass", "required_reports_enabled_and_streaming");
  } else {
    printEvent("imu_i2c_test_fail", "begin_ok_but_required_reports_missing");
  }
  return true;
}

bool attemptBeginDirect(const uint8_t address, const uint32_t clock_hz, const char* mode) {
  Wire.setClock(clock_hz);
  Serial.printf("event,%lu,begin_i2c_direct_start,mode_%s_addr_0x%02X_clock_%lu\n",
                static_cast<unsigned long>(millis()), mode, address,
                static_cast<unsigned long>(clock_hz));
  if (!bno08x.begin_I2C(address, &Wire)) {
    Serial.printf("event,%lu,begin_i2c_direct_failed,mode_%s_addr_0x%02X_clock_%lu\n",
                  static_cast<unsigned long>(millis()), mode, address,
                  static_cast<unsigned long>(clock_hz));
    if (probeAddress(address) == 0) {
      closeSh2AfterFailedBegin(mode, address, clock_hz);
    }
    return false;
  }
  Wire.setClock(clock_hz);
  return finishSuccessfulBegin(address, clock_hz);
}

bool attemptBeginAfterProbe(const uint8_t address, const uint32_t clock_hz) {
  Wire.setClock(clock_hz);
  const uint8_t probe = probeAddress(address);
  Serial.printf("event,%lu,i2c_probe,addr_0x%02X_clock_%lu_status_%s\n",
                static_cast<unsigned long>(millis()), address,
                static_cast<unsigned long>(clock_hz), i2cStatusName(probe));
  if (probe != 0) return false;

  Serial.printf("event,%lu,begin_i2c_after_probe_start,addr_0x%02X_clock_%lu\n",
                static_cast<unsigned long>(millis()), address,
                static_cast<unsigned long>(clock_hz));
  if (!bno08x.begin_I2C(address, &Wire)) {
    Serial.printf("event,%lu,begin_i2c_after_probe_failed,addr_0x%02X_clock_%lu\n",
                  static_cast<unsigned long>(millis()), address,
                  static_cast<unsigned long>(clock_hz));
    closeSh2AfterFailedBegin("after_probe", address, clock_hz);
    return false;
  }
  Wire.setClock(clock_hz);
  return finishSuccessfulBegin(address, clock_hz);
}

bool runBringup() {
  Wire.setSDA(kImuSdaPin);
  Wire.setSCL(kImuSclPin);
  Wire.begin();

  Serial.printf("event,%lu,wire_begin,sda_%u_scl_%u_reset_pin_%d_settle_%lums\n",
                static_cast<unsigned long>(millis()), kImuSdaPin, kImuSclPin,
                kImuResetPin, static_cast<unsigned long>(kPowerSettleMs));
  delay(kPowerSettleMs);

  if (attemptBeginDirect(kBno08xPrimaryAddress, 100000, "clean_no_probe_no_scan")) {
    return true;
  }
  delay(500);
  if (attemptBeginDirect(kBno08xPrimaryAddress, 50000, "clean_no_probe_no_scan")) {
    return true;
  }
  delay(500);

  for (const uint32_t clock_hz : kI2cClocksHz) {
    scanBus(clock_hz);
  }

  const uint8_t addresses[] = {kBno08xPrimaryAddress, kBno08xSecondaryAddress};
  for (const uint32_t clock_hz : kI2cClocksHz) {
    for (const uint8_t address : addresses) {
      if (attemptBeginAfterProbe(address, clock_hz)) {
        return true;
      }
    }
  }

  for (const uint32_t clock_hz : kI2cClocksHz) {
    for (const uint8_t address : addresses) {
      Wire.setClock(clock_hz);
      if (probeAddress(address) == 0) {
        for (uint8_t sample = 0; sample < kRawPacketDumpSamples; ++sample) {
          rawSingleReadProbe(address, clock_hz, sample);
          delay(kRawPacketDumpSpacingMs);
        }
        rawHeaderProbe(address, clock_hz);
        for (uint8_t sample = 0; sample < kRawPacketDumpSamples; ++sample) {
          rawPacketProbe(address, clock_hz, sample);
          delay(kRawPacketDumpSpacingMs);
        }
      }
    }
  }

  if (kImuResetPin < 0) {
    printEvent("imu_next_step", "wire_bno08x_rst_to_teensy_gpio_and_set_kImuResetPin");
  }
  printEvent("imu_known_issue", "adafruit_notes_bno08x_i2c_can_be_unreliable_on_some_nxp_imxrt_try_rst_then_uart_or_spi");
  printEvent("imu_i2c_test_fail", "no_address_completed_begin_i2c");
  return false;
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
  Serial.println("# Finn BNO08x I2C full check");
  Serial.println("# Isolated IMU test: no CAN, no moteus, no batch motion.");
  Serial.println("# Outputs: event lines for bringup, sample lines for live stream.");
  Serial.println("schema,imu_i2c_full_check_v1");

  runBringup();
}

void loop() {
  if (!imu_active) {
    digitalWrite(LED_BUILTIN, (millis() / 100U) & 1U);
    delay(20);
    return;
  }

  if (bno08x.wasReset()) {
    counters.resets++;
    printEvent("imu_reset", "loop_reenable_reports");
    configureReports();
  }

  serviceEvents(16);

  const uint32_t now_ms = millis();
  if (now_ms - last_sample_ms >= kSamplePeriodMs) {
    last_sample_ms = now_ms;
    printLatestSample();
  }
  if (now_ms - last_summary_ms >= kSummaryPeriodMs) {
    last_summary_ms = now_ms;
    printReportCounts("stream_counts");
  }

  digitalWrite(LED_BUILTIN, (millis() / 500U) & 1U);
}
