#include <Arduino.h>
#include <FlexCAN_T4.h>
#include <Adafruit_BNO08x.h>
#include <Wire.h>

constexpr uint8_t IMU_I2C_ADDR = 0x4A ; // I2C address for BNO08x
constexpr uint32_t I2C_CLOCK_HZ = 400000; // 400 kHz
constexpr uint32_t STREAM_HZ = 100; // 100 Hz
constexpr uint32_t IMU_INTERVAL_US = 1000000 / STREAM_HZ; // microseconds between IMU reads
constexpr uint32_t CAN_BAUD_RATE = 1000000; // 1 Mbps for smoke test
constexpr uint32_t CAN_TX_TEST_ID = 0x7FF; // probe id

Adafruit_BNO08x bno08x(-1); 
sh2_SensorValue_t sensor_value;
FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16> can3;

void halt_blink(const char* reason) {
    // failture indicator
    Serial.println("\nHalting execution: " + String(reason));
    while (true) {
        digitalWrite(LED_BUILTIN, HIGH);
        delay(100);
        digitalWrite(LED_BUILTIN, LOW);
        delay(100);
    }
}

bool scan_i2c() {
    // scan the I2C bus for devices, looking for the IMU
    Serial.println("Scanning I2C for device on wire pins 18/19...");
    uint8_t found = 0;
    bool saw_imu = false;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.printf(" 0x%02X", addr);
            if (addr == 0x4A || addr == 0x4B) { // can be at 0x4A or 0x4B depending on the state of the ADR pin
                Serial.println("  <- found BNO08x IMU>");
                saw_imu = true;
            }
            Serial.println();
            found++;
        }
    }
    Serial.printf("I2C scan complete, found %d device(s)\n", found);
    return saw_imu;
}

bool init_imu() {
    Serial.println("Initializing BNO08x IMU...");
    // note: begin_I2C() does not actually verify communication with the sensor, it just sets up the I2C HAL. 
    if (!bno08x.begin_I2C(IMU_I2C_ADDR)) {
        Serial.println(" -- beingI2C() failed");
        return false;
    }
    Wire.setClock(I2C_CLOCK_HZ);

    // enable main data report - rotation vector with 100ms interval (10Hz)
    if (!bno08x.enableReport(SH2_ROTATION_VECTOR, IMU_INTERVAL_US)){
        Serial.println(" -- enableReport(ROTATION_VECTOR) failed");
        return false;
    }
    Serial.printf(" OK. Rotation vector requested at %u Hz\n", STREAM_HZ);
    return true;
}

// testing can with a probe transmit - wont be acked by anything but the controller should at least send it out without error 
bool init_can_and_probe() {
    Serial.println("Initializing CAN bus...");
    can3.begin();
    can3.setBaudRate(CAN_BAUD_RATE);
    can3.setMaxMB(16); // use all mailboxes for transmission (no reception in this test)
    can3.enableFIFO();
    can3.enableFIFOInterrupt();
    Serial.printf(" Controller init OK. ");

    // probe message
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

    Wire.begin(); 
    Wire.setClock(I2C_CLOCK_HZ);

    // run tests and halt for failtures
    if (!scan_i2c())               halt_blink("IMU not found on I2C bus");
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
        bool ok = bno08x.enableReport(SH2_ROTATION_VECTOR, IMU_INTERVAL_US);
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
            // catch any other sensor IDs that might be arriving
            Serial.printf("debug: non-rotation event, sensorId=0x%02X\n", sensor_value.sensorId);
        }
        watchdog_timer = 0;
    }

    // if no events at all for 2 seconds, something is wrong
    if (watchdog_timer >= WATCHDOG_MS) {
        Serial.printf("warning: No IMU events in %lums (total events: %lu, rotation: %lu)\n",
                      WATCHDOG_MS, events_received, rotation_events);
        watchdog_timer = 0;
    }

    digitalWrite(LED_BUILTIN, (millis() / 500) & 1);
}
