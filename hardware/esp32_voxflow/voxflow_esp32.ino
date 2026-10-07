#include <Arduino.h>
#include "driver/i2s.h"
#include "esp_heap_caps.h"

// =====================================================
// VoxFlow ESP32 + INMP441
// True Real-Time Continuous Streaming Firmware (VXF1)
// Supports continuous sessions from 3s up to 60s
// Physical START (GPIO27), STOP (GPIO33), Status LED (GPIO4),
// Bounded producer-consumer ring buffer, 460800 baud binary framing.
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
#define SAMPLES_PER_FRAME 1024
#define BYTES_PER_SAMPLE 2
#define FRAME_PAYLOAD_BYTES (SAMPLES_PER_FRAME * BYTES_PER_SAMPLE) // 2048 bytes (64ms of audio)
#define MAX_RECORD_SECONDS 60

// Ring Buffer: 16 slots * 2048 bytes = 32,768 bytes (~1.024s buffer cushion)
#define RING_BUFFER_SLOTS 16

// Serial protocol
#define SERIAL_BAUD 460800
#define SERIAL_COMMAND_TIMEOUT_MS 100
#define I2S_READ_TIMEOUT_MS 100
#define I2S_STALL_TIMEOUT_MS 2000

// Debounce timing
#define DEBOUNCE_DELAY_MS 50

// Protocol Magic: "VXF1"
static const uint8_t VXF_MAGIC[4] = {'V', 'X', 'F', '1'};

#pragma pack(push, 1)
struct FrameHeader {
  uint8_t magic[4];
  uint32_t sequence;
  uint32_t payloadLength;
  uint32_t crc32;
};
#pragma pack(pop)

struct AudioFrame {
  int16_t samples[SAMPLES_PER_FRAME];
  size_t sampleCount;
};

static AudioFrame audioFramePool[RING_BUFFER_SLOTS];
static QueueHandle_t freeQueue = NULL;
static QueueHandle_t readyQueue = NULL;
static TaskHandle_t captureTaskHandle = NULL;

static bool i2sReady = false;
static volatile bool isRecording = false;
static volatile bool stopRequested = false;
static volatile bool streamError = false;
static const char* volatile streamErrorCode = NULL;

// CRC32 table for fast IEEE 802.3 polynomial calculation
static uint32_t crcTable[256];

void initCrcTable() {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int j = 0; j < 8; ++j) {
      if (c & 1) {
        c = 0xEDB88320UL ^ (c >> 1);
      } else {
        c >>= 1;
      }
    }
    crcTable[i] = c;
  }
}

uint32_t calculateCRC32(const uint8_t* data, size_t length) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t i = 0; i < length; ++i) {
    uint8_t idx = (uint8_t)((crc ^ data[i]) & 0xFF);
    crc = crcTable[idx] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFUL;
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
// I2S CAPTURE TASK (PRODUCER)
// Runs on Core 0 to continuously capture audio from I2S DMA
// =====================================================

void i2sCaptureTask(void* parameter) {
  int32_t rawI2SSamples[SAMPLES_PER_FRAME];

  while (true) {
    // Wait until recording is triggered
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    uint32_t lastSuccessfulReadMs = millis();
    uint32_t recordStartTimeMs = millis();

    // Clear microphone DMA buffer at session start
    i2s_zero_dma_buffer(I2S_PORT);
    delay(50);

    while (isRecording && !stopRequested) {
      // 1. Check physical STOP button (active LOW)
      if (digitalRead(STOP_BUTTON_PIN) == LOW) {
        vTaskDelay(pdMS_TO_TICKS(20));
        if (digitalRead(STOP_BUTTON_PIN) == LOW) {
          stopRequested = true;
          break;
        }
      }

      // 2. Check 60-second maximum duration
      if (millis() - recordStartTimeMs >= ((uint32_t)MAX_RECORD_SECONDS * 1000UL)) {
        stopRequested = true;
        break;
      }

      // 3. Acquire a free buffer slot from the ring buffer queue
      AudioFrame* frame = NULL;
      if (xQueueReceive(freeQueue, &frame, 0) != pdTRUE || frame == NULL) {
        // OVERRUN: Consumer (Serial) could not drain ring buffer in time!
        streamError = true;
        streamErrorCode = "ERROR:OVERRUN";
        break;
      }

      // 4. Read samples from I2S DMA
      size_t bytesRead = 0;
      esp_err_t result = i2s_read(
        I2S_PORT,
        rawI2SSamples,
        sizeof(rawI2SSamples),
        &bytesRead,
        pdMS_TO_TICKS(I2S_READ_TIMEOUT_MS)
      );

      if (result != ESP_OK || bytesRead == 0) {
        // Return frame to free pool
        xQueueSend(freeQueue, &frame, 0);

        if (millis() - lastSuccessfulReadMs >= I2S_STALL_TIMEOUT_MS) {
          streamError = true;
          streamErrorCode = "ERROR:I2S_STALL";
          break;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
        continue;
      }

      lastSuccessfulReadMs = millis();
      size_t samplesRead = bytesRead / sizeof(int32_t);

      // 5. Convert 24-bit in 32-bit I2S word to 16-bit PCM with clamping
      for (size_t i = 0; i < samplesRead; ++i) {
        int32_t sample = rawI2SSamples[i] >> 16;
        if (sample > 32767) {
          sample = 32767;
        } else if (sample < -32768) {
          sample = -32768;
        }
        frame->samples[i] = (int16_t)sample;
      }
      frame->sampleCount = samplesRead;

      // 6. Push captured frame to ready queue for consumer transmission
      if (xQueueSend(readyQueue, &frame, pdMS_TO_TICKS(50)) != pdTRUE) {
        streamError = true;
        streamErrorCode = "ERROR:QUEUE_FULL";
        break;
      }
    }

    // Capture loop ended
    isRecording = false;
  }
}

// =====================================================
// TRANSMIT & RECORDING CONTROLLER (CONSUMER)
// =====================================================

void recordAudio() {
  if (!i2sReady || isRecording) {
    return;
  }

  // Reset state flags
  isRecording = true;
  stopRequested = false;
  streamError = false;
  streamErrorCode = NULL;

  // Turn Status LED ON during recording and streaming
  digitalWrite(STATUS_LED_PIN, HIGH);

  // Reset ring buffer queues
  xQueueReset(freeQueue);
  xQueueReset(readyQueue);
  for (int i = 0; i < RING_BUFFER_SLOTS; ++i) {
    AudioFrame* ptr = &audioFramePool[i];
    xQueueSend(freeQueue, &ptr, 0);
  }

  // Send VXF1 protocol control header
  Serial.println("RECORDING");
  Serial.printf("SAMPLE_RATE=%u\n", SAMPLE_RATE);
  Serial.println("FORMAT=PCM_S16LE_MONO");
  Serial.println("STREAM=VXF1");
  Serial.println("DATA_START");
  Serial.flush();

  // Trigger capture task on Core 0
  xTaskNotifyGive(captureTaskHandle);

  uint32_t sequenceNumber = 0;

  // Consumer loop: Stream frames as they arrive in readyQueue
  while (true) {
    AudioFrame* frame = NULL;

    // Check if a frame is available to transmit
    if (xQueueReceive(readyQueue, &frame, pdMS_TO_TICKS(10)) == pdTRUE && frame != NULL) {
      size_t payloadBytes = frame->sampleCount * sizeof(int16_t);
      const uint8_t* pcmBytes = reinterpret_cast<const uint8_t*>(frame->samples);
      uint32_t crc = calculateCRC32(pcmBytes, payloadBytes);

      // Build 16-byte binary frame header
      FrameHeader header;
      memcpy(header.magic, VXF_MAGIC, 4);
      header.sequence = sequenceNumber++;
      header.payloadLength = (uint32_t)payloadBytes;
      header.crc32 = crc;

      // Transmit header + payload
      Serial.write(reinterpret_cast<const uint8_t*>(&header), sizeof(FrameHeader));
      Serial.write(pcmBytes, payloadBytes);

      // Return buffer slot to freeQueue
      xQueueSend(freeQueue, &frame, 0);
    }

    // Check if a serial STOP command was received during recording
    while (Serial.available() > 0) {
      char c = (char)Serial.read();
      if (c == 'S' || c == 's') {
        stopRequested = true;
      }
    }

    // Handle fatal stream errors (e.g. Overrun or I2S stall)
    if (streamError) {
      digitalWrite(STATUS_LED_PIN, LOW);
      isRecording = false;
      Serial.flush();
      if (streamErrorCode) {
        Serial.println(streamErrorCode);
      } else {
        Serial.println("ERROR:STREAM_ERROR");
      }
      Serial.flush();
      return;
    }

    // Check if producer has stopped and all remaining queued frames have been transmitted
    if (!isRecording && uxQueueMessagesWaiting(readyQueue) == 0) {
      break;
    }
  }

  // Drain any remaining frames left in readyQueue before terminating
  AudioFrame* finalFrame = NULL;
  while (xQueueReceive(readyQueue, &finalFrame, 0) == pdTRUE && finalFrame != NULL) {
    size_t payloadBytes = finalFrame->sampleCount * sizeof(int16_t);
    const uint8_t* pcmBytes = reinterpret_cast<const uint8_t*>(finalFrame->samples);
    uint32_t crc = calculateCRC32(pcmBytes, payloadBytes);

    FrameHeader header;
    memcpy(header.magic, VXF_MAGIC, 4);
    header.sequence = sequenceNumber++;
    header.payloadLength = (uint32_t)payloadBytes;
    header.crc32 = crc;

    Serial.write(reinterpret_cast<const uint8_t*>(&header), sizeof(FrameHeader));
    Serial.write(pcmBytes, payloadBytes);

    xQueueSend(freeQueue, &finalFrame, 0);
  }

  // Send zero-length end-of-stream final frame
  FrameHeader endHeader;
  memcpy(endHeader.magic, VXF_MAGIC, 4);
  endHeader.sequence = sequenceNumber++;
  endHeader.payloadLength = 0;
  endHeader.crc32 = 0;

  Serial.write(reinterpret_cast<const uint8_t*>(&endHeader), sizeof(FrameHeader));
  Serial.flush();

  // Send DONE marker
  delay(10);
  Serial.println("DONE");
  Serial.flush();

  // Turn Status LED OFF when session capture and transmission complete
  digitalWrite(STATUS_LED_PIN, LOW);
  isRecording = false;
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

  delay(500);

  // Initialize CRC32 lookup table
  initCrcTable();

  // Create producer-consumer FreeRTOS queues
  freeQueue = xQueueCreate(RING_BUFFER_SLOTS, sizeof(AudioFrame*));
  readyQueue = xQueueCreate(RING_BUFFER_SLOTS, sizeof(AudioFrame*));

  if (freeQueue == NULL || readyQueue == NULL) {
    fatalError("ERROR:QUEUE_INIT");
  }

  for (int i = 0; i < RING_BUFFER_SLOTS; ++i) {
    AudioFrame* ptr = &audioFramePool[i];
    xQueueSend(freeQueue, &ptr, 0);
  }

  // Initialize I2S
  if (!setupI2S()) {
    fatalError("ERROR:I2S_INIT");
  }

  // Spawn I2S Capture Task pinned to Core 0 with high priority
  xTaskCreatePinnedToCore(
    i2sCaptureTask,
    "I2SCaptureTask",
    4096,
    NULL,
    configMAX_PRIORITIES - 1,
    &captureTaskHandle,
    0
  );

  if (captureTaskHandle == NULL) {
    fatalError("ERROR:TASK_CREATE");
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
  if (isRecording || !i2sReady) {
    delay(2);
  }

  // 1. Physical START button monitoring (active LOW with INPUT_PULLUP)
  if (digitalRead(START_BUTTON_PIN) == LOW) {
    delay(DEBOUNCE_DELAY_MS);
    if (digitalRead(START_BUTTON_PIN) == LOW) {
      // Wait for button release before starting capture
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

