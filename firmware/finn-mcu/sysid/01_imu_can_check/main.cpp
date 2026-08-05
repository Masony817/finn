#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <Adafruit_BNO08x.h>
#include <Wire.h>
#include <sh2.h>

constexpr uint8_t IMU_SDA_PIN = 18;
constexpr uint8_t IMU_SCL_PIN = 19;
constexpr uint8_t IMU_PRIMARY_I2C_ADDR = 0x4A;
constexpr uint8_t IMU_SECONDARY_I2C_ADDR = 0x4B;
constexpr uint32_t I2C_CLOCK_HZ = 100000; // BNO08x is more reliable at standard-mode I2C on Teensy 4.1
constexpr uint32_t I2C_FALLBACK_CLOCK_HZ = 400000; // faster diagnostic fallback
constexpr uint32_t STREAM_HZ = 100; // 100 Hz
constexpr uint32_t IMU_INTERVAL_US = 1000000 / STREAM_HZ; // microseconds between IMU reads
constexpr uint32_t IMU_POWER_SETTLE_MS = 500;
constexpr uint32_t IMU_FIRST_EVENT_TIMEOUT_MS = 2000;
constexpr uint32_t CAN_BAUD_RATE = 1000000; // 1 Mbps for smoke test
constexpr uint32_t CAN_TX_TEST_ID = 0x7FF; // probe id

Adafruit_BNO08x bno08x(-1); 
sh2_SensorValue_t sensor_value;
FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> can3;
uint8_t active_imu_addr = 0;
uint32_t active_i2c_clock_hz = 0;

void halt_blink(const char* reason) {
    Serial.println("\nHalting execution: " + String(reason));
    while (true) {
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        delay(100);
    }
}

const char* i2c_status_name(uint8_t status) {
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

uint8_t probe_i2c_address(uint8_t addr) {
    Wire.beginTransmission(addr);
    return Wire.endTransmission();
}

bool is_bno08x_address(uint8_t addr) {
    return addr == IMU_PRIMARY_I2C_ADDR || addr == IMU_SECONDARY_I2C_ADDR;
}

bool scan_i2c(uint32_t clock_hz) {
    Wire.setClock(clock_hz);
    Serial.printf("Scanning I2C on SDA %u / SCL %u at %lu Hz...\n",
                  IMU_SDA_PIN, IMU_SCL_PIN, static_cast<unsigned long>(clock_hz));
    uint8_t found = 0;
    bool saw_imu = false;
    for (uint8_t addr = 1; addr < 127; addr++) {
        const uint8_t status = probe_i2c_address(addr);
        if (status == 0) {
            Serial.printf("  device 0x%02X status=%s", addr, i2c_status_name(status));
            if (is_bno08x_address(addr)) {
                Serial.print(" <- BNO08x candidate");
                saw_imu = true;
            }
            Serial.println();
            found++;
        }
    }
    Serial.printf("I2C scan complete, found %d device(s), bno_candidate=%s\n",
                  found, saw_imu ? "yes" : "no");
    return saw_imu;
}

bool configure_imu_report() {
    if (!bno08x.enableReport(SH2_ROTATION_VECTOR, IMU_INTERVAL_US)) {
        Serial.println(" -- enableReport(ROTATION_VECTOR) failed");
        return false;
    }
    Serial.printf(" -- rotation vector report requested at %u Hz\n", STREAM_HZ);
    return true;
}

bool wait_for_first_rotation_event(uint32_t timeout_ms) {
    const uint32_t start_ms = millis();
    uint32_t total_events = 0;
    uint32_t rotation_events = 0;
    while (millis() - start_ms < timeout_ms) {
        if (bno08x.wasReset()) {
            Serial.println(" -- reset event reported while waiting for first IMU sample; re-enabling report");
            if (!configure_imu_report()) {
                return false;
            }
        }
        if (bno08x.getSensorEvent(&sensor_value)) {
            total_events++;
            if (sensor_value.sensorId == SH2_ROTATION_VECTOR) {
                rotation_events++;
                const auto& q = sensor_value.un.rotationVector;
                Serial.printf(" -- first rotation vector after %lums: r=%+.4f i=%+.4f j=%+.4f k=%+.4f acc=%.3f\n",
                              static_cast<unsigned long>(millis() - start_ms),
                              q.real, q.i, q.j, q.k, q.accuracy);
                return true;
            }
            Serial.printf(" -- non-rotation IMU event while waiting: sensorId=0x%02X\n",
                          sensor_value.sensorId);
        }
        delay(1);
    }
    Serial.printf(" -- no rotation vector in %lums (events=%lu, rotation=%lu)\n",
                  static_cast<unsigned long>(timeout_ms),
                  static_cast<unsigned long>(total_events),
                  static_cast<unsigned long>(rotation_events));
    return false;
}

bool init_imu_at_address(uint8_t address, uint32_t clock_hz) {
    Wire.setClock(clock_hz);
    const uint8_t probe_status = probe_i2c_address(address);
    Serial.printf("Probing BNO08x address 0x%02X at %lu Hz: %s\n",
                  address, static_cast<unsigned long>(clock_hz),
                  i2c_status_name(probe_status));
    if (probe_status != 0) {
        return false;
    }

    Serial.printf("Initializing BNO08x at address 0x%02X...\n", address);
    if (!bno08x.begin_I2C(address, &Wire)) {
        Serial.printf(" -- begin_I2C(0x%02X) failed\n", address);
        sh2_close();
        Serial.println(" -- closed stale SH2 session after failed begin_I2C");
        return false;
    }
    Wire.setClock(clock_hz);

    if (!configure_imu_report()) {
        sh2_close();
        Serial.println(" -- closed SH2 session after report configuration failure");
        return false;
    }

    active_imu_addr = address;
    active_i2c_clock_hz = clock_hz;
    if (!wait_for_first_rotation_event(IMU_FIRST_EVENT_TIMEOUT_MS)) {
        Serial.println(" -- IMU initialized but did not stream usable rotation-vector data");
        sh2_close();
        Serial.println(" -- closed SH2 session after first-sample timeout");
        return false;
    }
    Serial.printf(" OK. BNO08x active at 0x%02X using %lu Hz I2C\n",
                  active_imu_addr, static_cast<unsigned long>(active_i2c_clock_hz));
    return true;
}

bool init_imu() {
    const uint32_t clocks[] = {I2C_CLOCK_HZ, I2C_FALLBACK_CLOCK_HZ};
    const uint8_t addresses[] = {IMU_PRIMARY_I2C_ADDR, IMU_SECONDARY_I2C_ADDR};
    for (const uint32_t clock_hz : clocks) {
        for (const uint8_t address : addresses) {
            if (init_imu_at_address(address, clock_hz)) return true;
        }
    }
    Serial.println(" -- failed at both BNO08x addresses (0x4A, 0x4B) and both I2C clocks (400k, 100k)");
    return false;
}

// Nothing acks this probe. A successful queue is the only signal available
// with no second node on the bus.
bool init_can_and_probe() {
    Serial.println("Initializing CAN bus...");
    can3.begin();
    can3.setBaudRate(CAN_BAUD_RATE);
    can3.setMaxMB(16); // use all mailboxes for transmission (no reception in this test)
    can3.enableFIFO();
    can3.enableFIFOInterrupt();
    Serial.printf(" Controller init OK. ");

    CAN_message_t msg;
    msg.id = CAN_TX_TEST_ID;
    msg.len = 1;
    msg.buf[0] = 0xA5; // test payload
    int result = can3.write(msg);
    Serial.printf("  TX probe write() returned %d (1 = queued, 0 = failed)\n", result);

    Serial.println("  NOTE: Frames will not be ACKed until board wiring is complete. That is");
    Serial.println("        normal for this stage of bringup.");
    return result == 1;
}

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0 < 3000)) { /* wait for serial connection, 3 second timeout */}

    Serial.println("Finn mcu bringup: IMU and CAN bus check");

    Wire.setSDA(IMU_SDA_PIN);
    Wire.setSCL(IMU_SCL_PIN);
    Wire.begin();
    Wire.setClock(I2C_CLOCK_HZ);
    Serial.printf("Configured Wire on SDA %u / SCL %u; waiting %lums for IMU power-up\n",
                  IMU_SDA_PIN, IMU_SCL_PIN, static_cast<unsigned long>(IMU_POWER_SETTLE_MS));
    delay(IMU_POWER_SETTLE_MS);

    // Run diagnostics before halting so the serial log shows the exact failure stage.
    const bool saw_primary = scan_i2c(I2C_CLOCK_HZ);
    if (!saw_primary) {
        scan_i2c(I2C_FALLBACK_CLOCK_HZ);
    }
    if (!init_imu())               halt_blink("Failed to initialize IMU");
    if (!init_can_and_probe())     halt_blink("Failed to initialize CAN bus and send probe message");

    Serial.println("\n all checks passed. starting stream...");
}

void loop() {
    static elapsedMillis print_timer;
    static elapsedMillis watchdog_timer;
    static uint32_t events_received = 0;
    static uint32_t rotation_events = 0;
    const uint32_t PRINT_INTERVAL_MS = 100;
    const uint32_t WATCHDOG_MS = 2000;

    if (bno08x.wasReset()) {
        Serial.println("IMU reset detected, re-enabling reports");
        delay(50);  // let the chip settle after reset
        bool ok = configure_imu_report();
        Serial.printf("enableReport returned %s\n", ok ? "true" : "false");
        watchdog_timer = 0;
        events_received = 0;
        rotation_events = 0;
    }

    if (bno08x.getSensorEvent(&sensor_value)) {
        events_received++;
        if (sensor_value.sensorId == SH2_ROTATION_VECTOR) {
            rotation_events++;
            if (print_timer >= PRINT_INTERVAL_MS) {
                print_timer = 0;
                const auto& q = sensor_value.un.rotationVector;
                Serial.printf("[%lu] quat r=%+.4f i=%+.4f j=%+.4f k=%+.4f acc=%.3f\n",
                              millis(), q.real, q.i, q.j, q.k, q.accuracy);
            }
        } else {
            Serial.printf("debug: non-rotation event, sensorId=0x%02X\n", sensor_value.sensorId);
        }
        watchdog_timer = 0;
    }

    if (watchdog_timer >= WATCHDOG_MS) {
        Serial.printf("warning: No IMU events in %lums (addr=0x%02X clock=%lu total=%lu rotation=%lu)\n",
                      WATCHDOG_MS, active_imu_addr,
                      static_cast<unsigned long>(active_i2c_clock_hz),
                      events_received, rotation_events);
        watchdog_timer = 0;
    }

    digitalWrite(LED_BUILTIN, (millis() / 500) & 1);
}
