#include <Arduino.h>
#include "driver/i2s.h"

// =====================================================
// VoxFlow ESP32 + INMP441
// Phase 2: Hardware Control with Physical Buttons & Status LED
// Supports physical START (GPIO27), STOP (GPIO33), Status LED (GPIO4),
// and serial START/STOP/PING commands up to MAX_RECORD_SECONDS (60s).
// =====================================================

#define I2S_PORT I2S_NUM_0

// INMP441 I2S wiring
#define I2S_SCK 26
#define I2S_WS  25
#define I2S_SD  32

// Physical Buttons & Status LED
#define START_BUTTON_PIN 27
#define STOP_BUTTON_PIN  33
#define STATUS_LED_PIN   4

// Audio settings
#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024
#define MAX_RECORD_SECONDS 60
#define TOTAL_SAMPLES (SAMPLE_RATE * MAX_RECORD_SECONDS)
#define BYTES_PER_SAMPLE 2
#define EXPECTED_AUDIO_BYTES (TOTAL_SAMPLES * BYTES_PER_SAMPLE)

// Serial protocol
#define SERIAL_BAUD 460800
#define SERIAL_COMMAND_TIMEOUT_MS 100
#define I2S_READ_TIMEOUT_MS 100
#define I2S_STALL_TIMEOUT_MS 2000

// Debounce timing
#define DEBOUNCE_DELAY_MS 50

static bool i2sReady = false;
static bool recording = false;

// =====================================================
// FATAL ERROR
// =====================================================

void fatalError(const char* code) {
  digitalWrite(STATUS_LED_PIN, LOW);
  Serial.println(code);
  Serial.flush();

  while (true) {
    delay(1000);
  }
}

// =====================================================
// I2S SETUP
// =====================================================

bool setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,

    // INMP441 L/R pin connected to GND = LEFT channel.
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,

    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  esp_err_t result = i2s_driver_install(
    I2S_PORT,
    &i2s_config,
    0,
    NULL
  );

  if (result != ESP_OK) {
    return false;
  }

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  result = i2s_set_pin(I2S_PORT, &pin_config);
  if (result != ESP_OK) {
    return false;
  }

  result = i2s_start(I2S_PORT);
  if (result != ESP_OK) {
    return false;
  }

  i2s_zero_dma_buffer(I2S_PORT);
  delay(250);

  return true;
}

// =====================================================
// SERIAL HELPERS
// =====================================================

void drainSerialInput() {
  while (Serial.available() > 0) {
    Serial.read();
  }
}

// =====================================================
// RECORD AUDIO
// =====================================================
// Serial protocol while recording:
//   RECORDING\n
//   AUDIO_BYTES=1920000\n
//   SAMPLE_RATE=16000\n
//   FORMAT=PCM_S16LE_MONO\n
//   [binary PCM payload]
//   DONE\n
// IMPORTANT: no text is printed during binary PCM transmission.
// =====================================================

void recordAudio() {
  if (!i2sReady || recording) {
    return;
  }

  recording = true;

  // Turn Status LED ON during recording
  digitalWrite(STATUS_LED_PIN, HIGH);

  // Clear stale microphone samples before capture.
  i2s_zero_dma_buffer(I2S_PORT);
  delay(50);

  // Text protocol header. Everything after FORMAT is binary PCM
  // until recording completes or STOP button is pressed.
  Serial.println("RECORDING");
  Serial.printf("AUDIO_BYTES=%lu\n", (unsigned long)EXPECTED_AUDIO_BYTES);
  Serial.printf("SAMPLE_RATE=%u\n", SAMPLE_RATE);
  Serial.println("FORMAT=PCM_S16LE_MONO");
  Serial.flush();

  int32_t samples[BUFFER_SIZE];
  int16_t pcmBuffer[BUFFER_SIZE];

  uint32_t totalSamplesSent = 0;
  uint32_t lastSuccessfulReadMs = millis();
  uint32_t recordStartTimeMs = millis();

  while (totalSamplesSent < TOTAL_SAMPLES) {
    // 1. Check physical STOP button during recording (active LOW with INPUT_PULLUP)
    if (digitalRead(STOP_BUTTON_PIN) == LOW) {
      delay(20); // Quick debounce
      if (digitalRead(STOP_BUTTON_PIN) == LOW) {
        break; // Stop collecting new samples immediately
      }
    }

    // 2. Safety limit: check 60 seconds elapsed
    if (millis() - recordStartTimeMs >= ((uint32_t)MAX_RECORD_SECONDS * 1000UL)) {
      break;
    }

    size_t bytesRead = 0;

    esp_err_t result = i2s_read(
      I2S_PORT,
      samples,
      sizeof(samples),
      &bytesRead,
      pdMS_TO_TICKS(I2S_READ_TIMEOUT_MS)
    );

    if (result != ESP_OK || bytesRead == 0) {
      if (millis() - lastSuccessfulReadMs >= I2S_STALL_TIMEOUT_MS) {
        digitalWrite(STATUS_LED_PIN, LOW);
        recording = false;
        Serial.println("ERROR:I2S_STALL");
        Serial.flush();
        return;
      }
      continue;
    }

    lastSuccessfulReadMs = millis();

    int samplesRead = bytesRead / sizeof(int32_t);
    uint32_t remaining = TOTAL_SAMPLES - totalSamplesSent;

    if ((uint32_t)samplesRead > remaining) {
      samplesRead = (int)remaining;
    }

    // Preserve the existing tested INMP441 conversion.
    // The microphone's useful 24-bit signal is carried in the upper
    // portion of the 32-bit I2S word for this setup.
    for (int i = 0; i < samplesRead; ++i) {
      int32_t sample = samples[i] >> 16;

      if (sample > 32767) {
        sample = 32767;
      } else if (sample < -32768) {
        sample = -32768;
      }

      pcmBuffer[i] = (int16_t)sample;
    }

    size_t bytesToSend = (size_t)samplesRead * sizeof(int16_t);

    size_t written = Serial.write(
      reinterpret_cast<const uint8_t*>(pcmBuffer),
      bytesToSend
    );

    if (written != bytesToSend) {
      digitalWrite(STATUS_LED_PIN, LOW);
      recording = false;
      Serial.flush();
      Serial.println("ERROR:SERIAL_WRITE");
      Serial.flush();
      return;
    }

    totalSamplesSent += (uint32_t)samplesRead;
  }

  // Make sure the entire PCM payload has physically left the UART
  // before sending the DONE marker.
  Serial.flush();
  delay(20);
  Serial.println("DONE");
  Serial.flush();

  // Turn Status LED OFF when recording ends
  digitalWrite(STATUS_LED_PIN, LOW);
  recording = false;
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(SERIAL_COMMAND_TIMEOUT_MS);

  // Configure physical buttons and LED
  pinMode(START_BUTTON_PIN, INPUT_PULLUP);
  pinMode(STOP_BUTTON_PIN, INPUT_PULLUP);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  delay(1500);

  if (!setupI2S()) {
    fatalError("ERROR:I2S_INIT");
  }

  i2sReady = true;

  // Inform a human / simple serial monitor that the board is alive.
  Serial.println("READY");
  Serial.flush();
}

// =====================================================
// LOOP
// =====================================================

void loop() {
  if (recording || !i2sReady) {
    delay(2);
    return;
  }

  // 1. Check physical START button (active LOW with INPUT_PULLUP)
  if (digitalRead(START_BUTTON_PIN) == LOW) {
    delay(DEBOUNCE_DELAY_MS);
    if (digitalRead(START_BUTTON_PIN) == LOW) {
      // Wait for release to avoid immediate re-triggering
      while (digitalRead(START_BUTTON_PIN) == LOW) {
        delay(10);
      }
      drainSerialInput();
      recordAudio();
      return;
    }
  }

  // 2. Check serial commands while idle
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();

    if (command == "PING") {
      // Handshake used by the PC bridge so stale boot messages cannot
      // cause synchronization errors.
      Serial.println("PONG");
      Serial.flush();

    } else if (command == "START") {
      drainSerialInput();
      recordAudio();
    }
  }

  delay(2);
}
