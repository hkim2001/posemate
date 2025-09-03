//esp32.cpp

// - 기능: MPU6050 → Notify(ESP32->Pi), 피드백 Write(Pi->ESP32) 수신 시 GPIO5 진동 모터 제어
// - NimBLE-Arduino 기반


#include <Wire.h>
#include <MPU6050.h>
#include <NimBLEDevice.h>

// ====== 기존 UUID 유지 ======
#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define CHARACTERISTIC_UUID "abcdefab-1234-5678-1234-56789abcdef1"   // IMU Notify (ESP32 -> Pi)
// 피드백(Write)용 추가 UUID
#define FEEDBACK_UUID       "abcdefab-1234-5678-1234-56789abcdef2"   // Pi -> ESP32

// ====== 보드 식별 ======
#define DEVICE_NAME       "ESP32-IMU-WAIST"
#define ENABLE_FEEDBACK   1   // WAIST 보드: 1

// ====== 하드웨어 핀 ======
#define VIBE_PIN          5   // GPIO5에 진동 모터 드라이브 신호 출력

// ====== 전역 ======
MPU6050 mpu;
static NimBLECharacteristic* imuChar = nullptr;
static NimBLECharacteristic* fbChar  = nullptr;

// 피드백 콜백: '0' OFF, '1' ON, 'B,200' 버스트, 'P,3,150' 패턴
class FeedbackCallback : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo& info) override {
#if ENABLE_FEEDBACK
    std::string v = c->getValue();
    if (v.empty()) return;

    if (v[0] == '0') {
      digitalWrite(VIBE_PIN, LOW);
    } else if (v[0] == '1') {
      digitalWrite(VIBE_PIN, HIGH);
    } else if (v[0] == 'B') { // "B,<ms>"
      uint32_t ms = 150;
      int i = v.find(',');
      if (i != (int)std::string::npos) ms = (uint32_t)atoi(v.c_str() + i + 1);
      digitalWrite(VIBE_PIN, HIGH);
      delay(ms);
      digitalWrite(VIBE_PIN, LOW);
    } else if (v[0] == 'P') { // "P,<count>,<ms>"
      int c1 = v.find(','), c2 = (c1==-1) ? -1 : v.find(',', c1+1);
      int count = 3, ms = 120;
      if (c1!=-1 && c2!=-1) {
        count = atoi(v.c_str()+c1+1);
        ms    = atoi(v.c_str()+c2+1);
      }
      for (int k=0; k<count; ++k) {
        digitalWrite(VIBE_PIN, HIGH);
        delay(ms);
        digitalWrite(VIBE_PIN, LOW);
        delay(ms);
      }
    }
#endif
  }
};

void setup() {
  Serial.begin(115200);
  delay(100);

#if ENABLE_FEEDBACK
  pinMode(VIBE_PIN, OUTPUT);
  digitalWrite(VIBE_PIN, LOW);
#endif

  // I2C (SDA=8, SCL=9) 
  Wire.begin(8, 9);
  Wire.setClock(400000);

  // MPU6050 초기화
  mpu.initialize();
  mpu.setSleepEnabled(false);
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 연결 실패(0x68 확인)");
    while (1) { delay(1000); }
  }

  // BLE 초기화
  NimBLEDevice::init(DEVICE_NAME);
  NimBLEDevice::setPower(ESP_PWR_LVL_P6);
  NimBLEDevice::setMTU(185);

  NimBLEServer* server = NimBLEDevice::createServer();
  NimBLEService* svc   = server->createService(SERVICE_UUID);

  // IMU Notify
  imuChar = svc->createCharacteristic(
    CHARACTERISTIC_UUID,
    NIMBLE_PROPERTY::NOTIFY | NIMBLE_PROPERTY::READ
  );
  imuChar->createDescriptor("2902"); // CCCD

#if ENABLE_FEEDBACK
  // Feedback Write (FEEDBACK_UUID)
  fbChar = svc->createCharacteristic(
    FEEDBACK_UUID,
    NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR
  );
  static FeedbackCallback fbCb;
  fbChar->setCallbacks(&fbCb);
#endif

  svc->start();

  // 광고
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  NimBLEAdvertisementData ad;
  ad.setName(DEVICE_NAME);
  ad.setCompleteServices(NimBLEUUID(SERVICE_UUID));
  adv->setAdvertisementData(ad);
  adv->start();

  Serial.println("BLE Advertising started");
}

void loop() {
  // IMU 읽기 → 16B 패킷 구성: ax ay az gx gy gz (int16 x6) + t_ms(uint32)
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);
  uint32_t t_ms = millis();

  uint8_t pkt[16];
  memcpy(&pkt[0],  &ax, 2);
  memcpy(&pkt[2],  &ay, 2);
  memcpy(&pkt[4],  &az, 2);
  memcpy(&pkt[6],  &gx, 2);
  memcpy(&pkt[8],  &gy, 2);
  memcpy(&pkt[10], &gz, 2);
  memcpy(&pkt[12], &t_ms, 4);

  imuChar->setValue(pkt, sizeof(pkt));
  imuChar->notify();

  // 전송 주기: 10ms(100Hz)
  delay(10);
}
