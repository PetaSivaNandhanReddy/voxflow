#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_heap_caps.h"

// =====================================================
// VoxFlow ESP32 + INMP441
// Phase 2: Hardware Control with Physical Buttons & Status LED
// Supports physical START (GPIO27), STOP (GPIO33), Status LED (GPIO4),
// serial START/STOP/PING commands, dynamic AUDIO_BYTES, and safe protocol framing.
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
#define MAX_TOTAL_SAMPLES (SAMPLE_RATE * MAX_RECORD_SECONDS)
#define BYTES_PER_SAMPLE 2

// Serial protocol
#define SERIAL_BAUD 460800
#define SERIAL_COMMAND_TIMEOUT_MS 100
#define I2S_READ_TIMEOUT_MS 100
#define I2S_STALL_TIMEOUT_MS 2000

// Debounce timing
#define DEBOUNCE_DELAY_MS 50

static bool i2sReady = false;
static bool recording = false;

// Audio capture buffer
static int16_t* audioBuffer = nullptr;
static uint32_t maxBufferSamples = 0;

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
// MEMORY BUFFER INITIALIZATION
// =====================================================

bool initAudioBuffer() {
  // 1. Try allocating full 60-second buffer from PSRAM if available
  if (psramInit()) {
    audioBuffer = (int16_t*)ps_malloc(MAX_TOTAL_SAMPLES * sizeof(int16_t));
    if (audioBuffer != nullptr) {
      maxBufferSamples = MAX_TOTAL_SAMPLES;
      return true;
    }
  }

  // 2. Fallback to largest available internal heap block
  size_t freeBlock = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  if (freeBlock > 24576) {
    size_t allocBytes = freeBlock - 24576; // Preserve 24KB margin for system/UART buffers
    audioBuffer = (int16_t*)malloc(allocBytes);
    if (audioBuffer != nullptr) {
      maxBufferSamples = (uint32_t)(allocBytes / sizeof(int16_t));
      return true;
    }
  }

  return false;
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
// Protocol flow:
// 1. Capture PCM samples into buffer until STOP button, serial STOP, or 60s max.
// 2. Compute exact captured audio bytes.
// 3. Emit metadata header with dynamic AUDIO_BYTES:
//      RECORDING\n
//      AUDIO_BYTES=<exact_bytes>\n
//      SAMPLE_RATE=16000\n
//      FORMAT=PCM_S16LE_MONO\n
// 4. Stream exactly that many binary PCM bytes.
// 5. Send DONE\n.
// =====================================================

void recordAudio() {
  if (!i2sReady || recording || audioBuffer == nullptr) {
    return;
  }

  recording = true;

  // Status LED ON while recording is active
  digitalWrite(STATUS_LED_PIN, HIGH);

  // Clear microphone DMA buffer before capture
  i2s_zero_dma_buffer(I2S_PORT);
  delay(50);

  int32_t i2sSamples[BUFFER_SIZE];
  uint32_t totalSamplesCaptured = 0;
  uint32_t lastSuccessfulReadMs = millis();
  uint32_t recordStartTimeMs = millis();
  bool stopRequested = false;

  // Audio capture loop
  while (totalSamplesCaptured < maxBufferSamples && !stopRequested) {
    // 1. Physical STOP button check (active LOW with INPUT_PULLUP)
    if (digitalRead(STOP_BUTTON_PIN) == LOW) {
      delay(20);
      if (digitalRead(STOP_BUTTON_PIN) == LOW) {
        stopRequested = true;
        break;
      }
    }

    // 2. Non-blocking serial STOP command check (avoids blocking I2S capture)
    while (Serial.available() > 0) {
      char c = (char)Serial.read();
      if (c == 'S' || c == 's') {
        stopRequested = true;
        break;
      }
    }
    if (stopRequested) {
      break;
    }

    // 3. Safety limit: maximum 60 seconds reached
    if (millis() - recordStartTimeMs >= ((uint32_t)MAX_RECORD_SECONDS * 1000UL)) {
      break;
    }

    size_t bytesRead = 0;
    esp_err_t result = i2s_read(
      I2S_PORT,
      i2sSamples,
      sizeof(i2sSamples),
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
    uint32_t remaining = maxBufferSamples - totalSamplesCaptured;
    if ((uint32_t)samplesRead > remaining) {
      samplesRead = (int)remaining;
    }

    // Convert 24-bit in 32-bit I2S word to 16-bit PCM with clamping
    for (int i = 0; i < samplesRead; ++i) {
      int32_t sample = i2sSamples[i] >> 16;
      if (sample > 32767) {
        sample = 32767;
      } else if (sample < -32768) {
        sample = -32768;
      }
      audioBuffer[totalSamplesCaptured++] = (int16_t)sample;
    }
  }

  // Calculate exact dynamic byte count of captured PCM audio
  uint32_t actualAudioBytes = totalSamplesCaptured * sizeof(int16_t);

  // Send exact protocol header
  Serial.println("RECORDING");
  Serial.printf("AUDIO_BYTES=%lu\n", (unsigned long)actualAudioBytes);
  Serial.printf("SAMPLE_RATE=%u\n", SAMPLE_RATE);
  Serial.println("FORMAT=PCM_S16LE_MONO");
  Serial.flush();

  // Transmit exact binary PCM payload
  const uint8_t* pcmBytes = reinterpret_cast<const uint8_t*>(audioBuffer);
  size_t totalBytesWritten = 0;

  while (totalBytesWritten < actualAudioBytes) {
    size_t chunk = actualAudioBytes - totalBytesWritten;
    if (chunk > 1024) {
      chunk = 1024;
    }

    size_t written = Serial.write(pcmBytes + totalBytesWritten, chunk);
    if (written == 0) {
      digitalWrite(STATUS_LED_PIN, LOW);
      recording = false;
      Serial.println("ERROR:SERIAL_WRITE");
      Serial.flush();
      return;
    }
    totalBytesWritten += written;
  }

  // Flush UART and send DONE marker
  Serial.flush();
  delay(20);
  Serial.println("DONE");
  Serial.flush();

  // Turn Status LED OFF when session capture and transmission complete
  digitalWrite(STATUS_LED_PIN, LOW);
  recording = false;
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(SERIAL_COMMAND_TIMEOUT_MS);

  // Configure physical buttons and status LED
  pinMode(START_BUTTON_PIN, INPUT_PULLUP);
  pinMode(STOP_BUTTON_PIN, INPUT_PULLUP);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  delay(1000);

  if (!initAudioBuffer()) {
    fatalError("ERROR:MEM_ALLOC");
  }

  if (!setupI2S()) {
    fatalError("ERROR:I2S_INIT");
  }

  i2sReady = true;

  // Inform PC bridge that firmware is ready
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

  // 1. Physical START button monitoring (active LOW with INPUT_PULLUP)
  if (digitalRead(START_BUTTON_PIN) == LOW) {
    delay(DEBOUNCE_DELAY_MS);
    if (digitalRead(START_BUTTON_PIN) == LOW) {
      // Wait for release before starting capture
      while (digitalRead(START_BUTTON_PIN) == LOW) {
        delay(10);
      }
      drainSerialInput();
      recordAudio();
      return;
    }
  }

  // 2. Serial command monitoring while idle
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();

    if (command == "PING") {
      Serial.println("PONG");
      Serial.flush();
    } else if (command == "START") {
      drainSerialInput();
      recordAudio();
    } else if (command == "STOP") {
      // STOP while idle does nothing
      drainSerialInput();
    }
  }

  delay(2);
}
