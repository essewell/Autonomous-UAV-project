#include "driver/spi_slave.h"
#include "esp_log.h"
#include <esp32cam.h>

// SPI Pin Definitions for ESP32-S3
#define GPIO_MOSI 39
#define GPIO_MISO 38
#define GPIO_SCLK 40
#define GPIO_CS   14
#define GPIO_HANDSHAKE 21

// SPI Configuration
#define SPI_MODE 0
#define RCV_HOST SPI2_HOST
#define BUFFER_SIZE 4096

DMA_ATTR uint8_t recvbuf[BUFFER_SIZE];
DMA_ATTR uint8_t sendbuf[BUFFER_SIZE];

// Camera frame buffer
std::unique_ptr<esp32cam::Frame> currentFrame;
size_t frameOffset = 0;
bool frameReady = false;
uint32_t frameSize = 0;

// Protocol definitions
#define CMD_GET_FRAME_SIZE 0x01
#define CMD_GET_FRAME_DATA 0x02
#define CMD_FRAME_COMPLETE 0x03

// Response types to track what we're sending
enum ResponseType {
  RESP_FRAME_SIZE,
  RESP_CHUNK_DATA,
  RESP_ACK
};
ResponseType currentResponse = RESP_FRAME_SIZE;

void setupCamera() {
  using namespace esp32cam;
  
  Serial.println("Initializing camera...");
  
  Config cfg;
  esp32cam::Pins pins;
  pins.D0 = 11;
  pins.D1 = 9;
  pins.D2 = 8;
  pins.D3 = 10;
  pins.D4 = 12;
  pins.D5 = 18;
  pins.D6 = 17;
  pins.D7 = 16;
  pins.XCLK = 15;
  pins.PCLK = 13;
  pins.VSYNC = 6;
  pins.HREF = 7;
  pins.SDA = 4;
  pins.SCL = 5;
  pins.PWDN = -1;
  pins.RESET = -1;
  cfg.setPins(pins);
  cfg.setResolution(Resolution::find(640, 480));
  cfg.setJpeg(80);
  
  bool ok = Camera.begin(cfg);
  if (!ok) {
    Serial.println("Camera initialization failed!");
    delay(5000);
    ESP.restart();
  }
  
  Serial.println("Camera initialized successfully");
}

void captureNewFrame() {
  Serial.println("\n=== Capturing new frame ===");
  currentFrame = esp32cam::capture();
  
  if (currentFrame == nullptr) {
    Serial.println("Frame capture failed");
    frameReady = false;
    frameSize = 0;
    return;
  }
  
  frameOffset = 0;
  frameReady = true;
  frameSize = currentFrame->size();
  
  Serial.printf("Frame captured: %d bytes\n", frameSize);
}

void prepareResponse(ResponseType type) {
  memset(sendbuf, 0xFF, BUFFER_SIZE);  // Fill with 0xFF to detect uninitialized reads
  
  switch (type) {
    case RESP_FRAME_SIZE:
      if (frameReady && currentFrame) {
        memcpy(sendbuf, &frameSize, sizeof(uint32_t));
        Serial.printf("[PREP] Frame size: %d bytes\n", frameSize);
      } else {
        uint32_t zero = 0;
        memcpy(sendbuf, &zero, sizeof(uint32_t));
        Serial.println("[PREP] No frame ready (size=0)");
      }
      currentResponse = RESP_FRAME_SIZE;
      break;
      
    case RESP_CHUNK_DATA:
      if (frameReady && currentFrame && frameOffset < frameSize) {
        size_t remaining = frameSize - frameOffset;
        size_t chunkSize = (remaining > BUFFER_SIZE - 4) ? BUFFER_SIZE - 4 : remaining;
        
        memcpy(sendbuf, &chunkSize, sizeof(uint32_t));
        memcpy(sendbuf + 4, currentFrame->data() + frameOffset, chunkSize);
        
        Serial.printf("[PREP] Chunk: %d bytes at offset %d/%d\n", chunkSize, frameOffset, frameSize);
        frameOffset += chunkSize;
      } else {
        uint32_t zero = 0;
        memcpy(sendbuf, &zero, sizeof(uint32_t));
        Serial.println("[PREP] No data available");
      }
      currentResponse = RESP_CHUNK_DATA;
      break;
      
    case RESP_ACK:
      sendbuf[0] = 0x01;  // ACK
      Serial.println("[PREP] ACK");
      currentResponse = RESP_ACK;
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32-S3 Camera SPI Slave (Improved) ===");
  
  pinMode(GPIO_HANDSHAKE, OUTPUT);
  digitalWrite(GPIO_HANDSHAKE, LOW);
  
  setupCamera();
  
  // Initialize SPI slave
  spi_bus_config_t buscfg = {
    .mosi_io_num = GPIO_MOSI,
    .miso_io_num = GPIO_MISO,
    .sclk_io_num = GPIO_SCLK,
    .quadwp_io_num = -1,
    .quadhd_io_num = -1,
    .max_transfer_sz = BUFFER_SIZE,
  };
  
  spi_slave_interface_config_t slvcfg = {
    .spics_io_num = GPIO_CS,
    .flags = 0,
    .queue_size = 3,
    .mode = SPI_MODE,
  };
  
  esp_err_t ret = spi_slave_initialize(RCV_HOST, &buscfg, &slvcfg, SPI_DMA_CH_AUTO);
  if (ret != ESP_OK) {
    Serial.printf("SPI init failed: %d\n", ret);
    return;
  }
  
  Serial.println("SPI Slave initialized");
  
  // Capture initial frame
  captureNewFrame();
  
  // Pre-load with frame size response
  prepareResponse(RESP_FRAME_SIZE);
  
  digitalWrite(GPIO_HANDSHAKE, HIGH);
  Serial.println("Ready for communication\n");
}

void loop() {
  memset(recvbuf, 0, BUFFER_SIZE);
  
  // Setup transaction with PRE-LOADED response
  spi_slave_transaction_t t;
  memset(&t, 0, sizeof(t));
  t.length = BUFFER_SIZE * 8;
  t.tx_buffer = sendbuf;
  t.rx_buffer = recvbuf;
  
  // This sends sendbuf and receives into recvbuf
  esp_err_t ret = spi_slave_transmit(RCV_HOST, &t, portMAX_DELAY);
  
  if (ret == ESP_OK) {
    uint8_t cmd = recvbuf[0];
    
    Serial.printf("[RECV] CMD=0x%02X | Sent response type=%d\n", cmd, currentResponse);
    
    // Now prepare response for NEXT transaction
    switch (cmd) {
      case CMD_GET_FRAME_SIZE:
        // We just sent the frame size, next they'll ask for data
        frameOffset = 0;  // Reset for new transfer
        prepareResponse(RESP_CHUNK_DATA);
        break;
      
      case CMD_GET_FRAME_DATA:
        // We just sent a chunk, prepare next chunk or wait for complete
        if (frameOffset < frameSize) {
          prepareResponse(RESP_CHUNK_DATA);
        } else {
          prepareResponse(RESP_ACK);
        }
        break;
      
      case CMD_FRAME_COMPLETE:
        // Capture new frame and prepare size response
        captureNewFrame();
        prepareResponse(RESP_FRAME_SIZE);
        break;
      
      case 0x00:
        // Null command - maintain current state
        Serial.println("[WARN] Null command");
        break;
      
      default:
        Serial.printf("[WARN] Unknown CMD=0x%02X\n", cmd);
        // Try to recover by sending frame size
        prepareResponse(RESP_FRAME_SIZE);
        break;
    }
  } else {
    Serial.printf("SPI error: %d\n", ret);
    delay(100);
  }
  
  delayMicroseconds(10);  // Tiny delay to allow processing (originally 100)
}