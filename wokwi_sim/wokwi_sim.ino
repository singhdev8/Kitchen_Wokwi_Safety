#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ── PIN DEFINITIONS ──────────────────────────────────────────
#define TEMP_PIN   34
#define GAS_PIN    35
#define PIR_PIN    26
#define RELAY_PIN  27
#define BUZZER_PIN 25

// ── THRESHOLDS ───────────────────────────────────────────────
#define TEMP_BOILOVER   88.0
#define TEMP_RISE_RATE  0.25
#define GAS_LEAK_THRESH 400.0
#define TEMP_DROP_RATE  -0.5
#define TIMEOUT_LIMIT   2700

// ── GLOBALS ──────────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);

float temp_history[30];
float gas_history[30];
int   history_idx   = 0;
bool  history_full  = false;
bool  shutoff       = false;
int   no_pres_timer = 0;
int   loop_count    = 0;

// ── FUNCTION: READ TEMPERATURE ───────────────────────────────
float readTemperature() {
  int raw = analogRead(TEMP_PIN);
  return map(raw, 0, 4095, 25, 120);
}

// ── FUNCTION: READ GAS ───────────────────────────────────────
float readGas() {
  int raw = analogRead(GAS_PIN);
  return map(raw, 0, 4095, 0, 1000);
}

// ── FUNCTION: ROLLING SLOPE ──────────────────────────────────
float rollingSlope(float* arr, int len) {
  float sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
  for (int i = 0; i < len; i++) {
    sum_x  += i;
    sum_y  += arr[i];
    sum_xy += i * arr[i];
    sum_x2 += i * i;
  }
  float denom = len * sum_x2 - sum_x * sum_x;
  if (denom == 0) return 0;
  return (len * sum_xy - sum_x * sum_y) / denom;
}

// ── FUNCTION: TRIGGER SHUTOFF ────────────────────────────────
void triggerShutoff(String reason) {
  shutoff = true;
  digitalWrite(RELAY_PIN,  LOW);
  digitalWrite(BUZZER_PIN, HIGH);

  Serial.println("================================");
  Serial.println("!!  SHUTOFF TRIGGERED  !!");
  Serial.print  ("    Reason : "); Serial.println(reason);
  Serial.print  ("    Time   : "); Serial.print(loop_count); Serial.println("s");
  Serial.println("================================");

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("!! SHUTOFF !!");
  lcd.setCursor(0, 1); lcd.print(reason.substring(0, 16));
}

// ── SETUP ────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(RELAY_PIN,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(PIR_PIN,    INPUT);

  digitalWrite(RELAY_PIN,  HIGH);
  digitalWrite(BUZZER_PIN, LOW);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("Kitchen Safety");
  lcd.setCursor(0, 1); lcd.print("System Ready");

  Serial.println("================================");
  Serial.println("  Smart Kitchen Safety System");
  Serial.println("  Layer 1 Rule Engine ACTIVE");
  Serial.println("================================");
  Serial.println("Time | Temp  | Gas   | PIR | Status");
  Serial.println("-----+-------+-------+-----+-------");

  delay(2000);
}

// ── MAIN LOOP ────────────────────────────────────────────────
void loop() {
  if (shutoff) {
    delay(1000);
    return;
  }

  loop_count++;

  // Read sensors
  static float filt_temp = 0;
  static float filt_gas  = 0;

  float raw_temp = readTemperature();
  float raw_gas  = readGas();

  filt_temp = 0.8*filt_temp + 0.2*raw_temp;
  filt_gas  = 0.8*filt_gas  + 0.2*raw_gas;

  float temperature = filt_temp;
  float gas_ppm     = filt_gas;
  bool  presence    = digitalRead(PIR_PIN);

  // Store history
  temp_history[history_idx] = temperature;
  gas_history[history_idx]  = gas_ppm;
  history_idx = (history_idx + 1) % 30;
  if (history_idx == 0) history_full = true;

  int   window_len = history_full ? 30 : max(history_idx, 1);
  float dT_avg     = (window_len > 1) ? rollingSlope(temp_history, window_len) : 0.0;
  float dG_avg     = (window_len > 1) ? rollingSlope(gas_history,  window_len) : 0.0;

  // Serial log
  Serial.print(loop_count);
  Serial.print("s\t| ");
  Serial.print(temperature, 1);
  Serial.print("C\t| ");
  Serial.print(gas_ppm, 0);
  Serial.print(" ppm\t| ");
  Serial.print(presence ? "YES" : "NO");
  Serial.print("\t| ");

  // LCD update
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(temperature, 1);
  lcd.print(" G:");
  lcd.print((int)gas_ppm);

  // Rule 1: Boilover
  if (temperature >= TEMP_BOILOVER && dT_avg >= TEMP_RISE_RATE && gas_ppm > 200) {
    Serial.println("RULE 1 FIRED — OVERFLOW RISK");
    triggerShutoff("OVERFLOW RISK");
    return;
  }

  // Rule 2: Gas leak
  if (gas_ppm >= GAS_LEAK_THRESH && temperature < 80.0) {
    Serial.println("RULE 2 FIRED — GAS LEAK");
    triggerShutoff("GAS LEAK");
    return;
  }

  // Rule 3: Timeout
  if (!presence && gas_ppm > 200) {
    no_pres_timer += 1;
  } else {
    no_pres_timer = 0;
  }
  if (no_pres_timer >= TIMEOUT_LIMIT) {
    Serial.println("RULE 3 FIRED — TIMEOUT");
    triggerShutoff("TIMEOUT 45min");
    return;
  }

  // Rule 4: Flame-out
  if (dT_avg < TEMP_DROP_RATE && dG_avg > 0.5 && gas_ppm > 300) {
    Serial.println("RULE 4 FIRED — FLAME-OUT");
    triggerShutoff("FLAME-OUT");
    return;
  }

  // Safe
  Serial.println("SAFE");
  lcd.setCursor(0, 1);
  lcd.print("Status: SAFE    ");

  delay(1000);
}


/*simualtion runner +recompilation
arduino-cli compile \
  --fqbn esp32:esp32:esp32 \
  --output-dir /Users/devkaransinghsarkaria/kitchen_safety/wokwi_sim/build \
  /Users/devkaransinghsarkaria/kitchen_safety/wokwi_sim
  
wokwi-cli --serial-log-file serial_output.txt --scenario scenario.yaml --timeout 120000 .


*/