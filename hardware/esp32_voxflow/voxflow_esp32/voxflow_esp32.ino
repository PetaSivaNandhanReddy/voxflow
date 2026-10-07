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
#define BYTES_PER_SAMPLE 2

// Chunked Audio Buffer Settings
// 4096 samples = 8192 bytes = 0.256s per chunk
#define CHUNK_SAMPLES 4096
#define MAX_CHUNKS 240 // Up to 61.44 seconds (240 * 4096 = 983,040 samples)

// Serial protocol
#define SERIAL_BAUD 460800
#define SERIAL_COMMAND_TIMEOUT_MS 100
#define I2S_READ_TIMEOUT_MS 100
#define I2S_STALL_TIMEOUT_MS 2000

// Debounce timing
#define DEBOUNCE_DELAY_MS 50

static bool i2sReady = false;
static bool recording = false;

// Segmented audio capture buffer
static int16_t* chunkPointers[MAX_CHUNKS];
static uint32_t numAllocatedChunks = 0;
static uint32_t maxBufferSamples = 0;

inline void writeAudioSample(uint32_t index, int16_t sample) {
  uint32_t c = index / CHUNK_SAMPLES;
  uint32_t offset = index % CHUNK_SAMPLES;
  chunkPointers[c][offset] = sample;
}

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
  numAllocatedChunks = 0;
  bool usePSRAM = psramInit();

  for (uint32_t i = 0; i < MAX_CHUNKS; ++i) {
    int16_t* ptr = nullptr;
    if (usePSRAM) {
      ptr = (int16_t*)ps_malloc(CHUNK_SAMPLES * sizeof(int16_t));
    }
    if (ptr == nullptr) {
      // Allocate from internal DRAM if safe system margin (16KB) remains
      if (heap_caps_get_free_size(MALLOC_CAP_8BIT) > (CHUNK_SAMPLES * sizeof(int16_t) + 16384)) {
        ptr = (int16_t*)malloc(CHUNK_SAMPLES * sizeof(int16_t));
      }
    }
    if (ptr == nullptr) {
      break;
    }
    chunkPointers[numAllocatedChunks++] = ptr;
  }

  maxBufferSamples = numAllocatedChunks * CHUNK_SAMPLES;
  // Require at least 12 chunks = ~3.07 seconds (49,152 samples)
  return numAllocatedChunks >= 12;
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
    .dma_buf_count = 4,
    .dma_buf_len = 512,
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
  if (!i2sReady || recording || numAllocatedChunks == 0) {
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
      writeAudioSample(totalSamplesCaptured++, (int16_t)sample);
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

  // Transmit exact binary PCM payload chunk by chunk
  uint32_t samplesSent = 0;
  for (uint32_t c = 0; c < numAllocatedChunks && samplesSent < totalSamplesCaptured; ++c) {
    uint32_t samplesInThisChunk = totalSamplesCaptured - samplesSent;
    if (samplesInThisChunk > CHUNK_SAMPLES) {
      samplesInThisChunk = CHUNK_SAMPLES;
    }
    size_t bytesInThisChunk = samplesInThisChunk * sizeof(int16_t);
    const uint8_t* pcmBytes = reinterpret_cast<const uint8_t*>(chunkPointers[c]);
    size_t chunkBytesWritten = 0;

    while (chunkBytesWritten < bytesInThisChunk) {
      size_t toWrite = bytesInThisChunk - chunkBytesWritten;
      if (toWrite > 1024) {
        toWrite = 1024;
      }
      size_t written = Serial.write(pcmBytes + chunkBytesWritten, toWrite);
      if (written == 0) {
        digitalWrite(STATUS_LED_PIN, LOW);
        recording = false;
        Serial.println("ERROR:SERIAL_WRITE");
        Serial.flush();
        return;
      }
      chunkBytesWritten += written;
    }
    samplesSent += samplesInThisChunk;
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

  // Initialize I2S first so DMA buffers are secured
  if (!setupI2S()) {
    fatalError("ERROR:I2S_INIT");
  }

  // Initialize segmented audio buffer from PSRAM or internal DRAM
  if (!initAudioBuffer()) {
    fatalError("ERROR:MEM_ALLOC");
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
