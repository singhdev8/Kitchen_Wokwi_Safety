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
#define TIMEOUT_LIMIT   90     // FIX 2: Reduced to 90s for simulation

// ── SCENARIO ENUM ────────────────────────────────────────────
enum Scenario {
  NORMAL_COOKING,
  MILK_BOILOVER,
  GAS_LEAK,
  TIMEOUT_RISK,
  FLAME_OUT
};

// ── CONFIGURATION ────────────────────────────────────────────
Scenario ACTIVE_SCENARIO = GAS_LEAK;  // Default scenario for training
bool DEMO_MODE = false;              // Set to true to cycle through all scenarios
int last_scenario_index = -1;        // Track last scenario for clean transitions
int scenario_start_time = 0;         // FIX 1: Track when current scenario started

// ── STATE VARIABLES ──────────────────────────────────────────
String current_scenario;
String current_phase;
String last_status = "SAFE";

float temperature = 30;
float gas_ppm = 150;
bool presence = true;

// ── GLOBALS ──────────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);

float temp_history[30];
float gas_history[30];
int   history_idx   = 0;
bool  history_full  = false;
bool  shutoff       = false;
int   no_pres_timer = 0;
int   loop_count    = 0;

// ── NOISE FUNCTION ──────────────────────────────────────────
float noise(float amp) {
  return random(-100, 100) / 100.0 * amp;
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

// ── FUNCTION: GENERATE SCENARIO DATA ────────────────────────
void generateScenarioData() {
  // FIX 1: Use scenario-relative time instead of global loop_count
  int t = loop_count - scenario_start_time;
  
  // ── DEMO MODE: Auto-switch scenarios with clean reset ──
  if (DEMO_MODE) {
    // Use scenario-relative time for switching
    int scenario_index = (t / 120) % 5;  // Switch every 120 seconds within scenario
    
    // Detect scenario change
    if (scenario_index != last_scenario_index) {
      
      // Update to new scenario
      ACTIVE_SCENARIO = (Scenario)scenario_index;
      
      // ── CRITICAL: Reset ALL state for clean transition ──
      history_idx = 0;
      history_full = false;
      no_pres_timer = 0;
      
      // Reset shutoff state
      shutoff = false;
      last_status = "SAFE";
      
      // Clear history arrays
      for (int i = 0; i < 30; i++) {
        temp_history[i] = 0;
        gas_history[i] = 0;
      }
      
      // Reset relay and buzzer for new scenario
      digitalWrite(RELAY_PIN, HIGH);
      digitalWrite(BUZZER_PIN, LOW);
      
      // FIX 1: Reset scenario start time when switching
      scenario_start_time = loop_count;
      
      // Update tracking
      last_scenario_index = scenario_index;
      
      // Log scenario change
      Serial.println("================================");
      Serial.print("NEW SCENARIO: ");
      switch(scenario_index) {
        case 0: Serial.println("NORMAL_COOKING"); break;
        case 1: Serial.println("MILK_BOILOVER"); break;
        case 2: Serial.println("GAS_LEAK"); break;
        case 3: Serial.println("TIMEOUT_RISK"); break;
        case 4: Serial.println("FLAME_OUT"); break;
      }
      Serial.println("================================");
      
      // Update t to new scenario-relative time
      t = 0;
    }
  }
  
  // ── SCENARIO GENERATION ──────────────────────────────────
  switch(ACTIVE_SCENARIO) {
    
    case GAS_LEAK:
      current_scenario = "GAS_LEAK";
      
      if(t < 20) {
        current_phase = "SAFE";
        gas_ppm = 150 + noise(3);
      }
      else if(t < 40) {
        current_phase = "FORMING";
        gas_ppm = 150 + (t-20)*5 + noise(5);
      }
      else if(t < 55) {
        current_phase = "ML_ZONE";
        gas_ppm = 250 + (t-40)*6 + noise(6);
      }
      else if(t < 65) {
        current_phase = "RULE_ZONE";
        gas_ppm = 340 + (t-55)*5 + noise(8);
      }
      else {
        current_phase = "TRIGGERED";
        gas_ppm = 410 + noise(5);
      }
      
      temperature = 45 + noise(0.5);
      presence = true;
      break;
      
    case MILK_BOILOVER:
      current_scenario = "BOILOVER";
      
      if(t < 20) {
        current_phase = "SAFE";
        temperature = 50 + t + noise(0.3);
      }
      else if(t < 40) {
        current_phase = "FORMING";
        temperature = 70 + (t-20)*0.5 + noise(0.3);
      }
      else if(t < 55) {
        current_phase = "ML_ZONE";
        temperature = 80 + (t-40)*0.4 + noise(0.3);
      }
      else if(t < 65) {
        current_phase = "RULE_ZONE";
        temperature = 86 + (t-55)*0.3 + noise(0.3);
      }
      else {
        current_phase = "TRIGGERED";
        // FIX 3: Increased slope from 0.1 to 0.35 to guarantee Rule 1 fires
        temperature = 89 + (t-65)*0.35 + noise(0.3);
      }
      
      gas_ppm = 250 + noise(5);
      presence = true;
      break;
      
    case TIMEOUT_RISK:
      current_scenario = "TIMEOUT";
      
      if(t < 30) {
        current_phase = "COOKING";
        temperature = 55 + t*0.3 + noise(0.5);
        presence = true;
      }
      else if(t < 60) {
        current_phase = "LEAVING";
        temperature = 65 + (t-30)*0.1 + noise(0.5);
        presence = (t % 10 < 8);  // Intermittent presence
      }
      else {
        current_phase = "UNATTENDED";
        temperature = 68 + (t-60)*0.05 + noise(0.5);
        presence = false;
      }
      
      gas_ppm = 300 + 5*sin(t/10) + noise(5);
      break;
      
    case FLAME_OUT:
      current_scenario = "FLAME_OUT";
      
      if(t < 40) {
        current_phase = "HEATING";
        temperature = 35 + 45*(1-exp(-t/20)) + noise(0.5);
        gas_ppm = 300 + noise(5);
      }
      else if(t < 70) {
        current_phase = "FLAME_DROP";
        temperature = 78 - 0.6*(t-40) + noise(0.5);
        gas_ppm = 300 + 2.2*(t-40) + noise(5);
      }
      else {
        current_phase = "GAS_LEAK";
        temperature = 45 + 10*exp(-(t-70)/30) + noise(0.5);
        gas_ppm = 366 + 1.2*(t-70) + noise(6);
      }
      
      presence = true;
      break;
      
    case NORMAL_COOKING:
    default:
      current_scenario = "NORMAL";
      
      if(t < 30) {
        current_phase = "HEATING";
        temperature = 30 + 40*(1-exp(-t/15)) + noise(0.5);
      }
      else {
        current_phase = "STEADY";
        temperature = 65 + 5*sin(t/20) + noise(0.5);
      }
      
      gas_ppm = 300 + 10*sin(t/15) + noise(5);
      presence = (t % 5 < 4);  // Present most of the time
      break;
  }
}

// ── FUNCTION: TRIGGER SHUTOFF ────────────────────────────────
void triggerShutoff(String reason) {
  shutoff = true;
  digitalWrite(RELAY_PIN,  LOW);
  digitalWrite(BUZZER_PIN, HIGH);
  
  // Update status for serial output
  last_status = reason;
  
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
  randomSeed(analogRead(0));  // Seed random for noise
  
  pinMode(RELAY_PIN,  OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(PIR_PIN,    INPUT);
  
  digitalWrite(RELAY_PIN,  HIGH);
  digitalWrite(BUZZER_PIN, LOW);
  
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("Kitchen Safety");
  lcd.setCursor(0, 1); lcd.print("System Ready");
  
  // Initialize scenario_start_time
  scenario_start_time = 0;
  
  // Print header in new format
  Serial.println("================================");
  Serial.println("  Smart Kitchen Safety System");
  Serial.println("  Layer 1 Rule Engine ACTIVE");
  if (DEMO_MODE) {
    Serial.println("  DEMO MODE: Cycling scenarios");
  } else {
    Serial.println("  TRAINING MODE: Single scenario");
  }
  Serial.println("================================");
  Serial.println("Time|Temp|Gas|Presence|Scenario|Phase|Status");
  
  delay(2000);
}

// ── MAIN LOOP ────────────────────────────────────────────────
void loop() {
  if (shutoff) {
    // Keep logging the triggered state
    Serial.print(loop_count);
    Serial.print("|");
    Serial.print(temperature, 1);
    Serial.print("|");
    Serial.print(gas_ppm, 0);
    Serial.print("|");
    Serial.print(presence ? "YES" : "NO");
    Serial.print("|");
    Serial.print(current_scenario);
    Serial.print("|");
    Serial.print(current_phase);
    Serial.print("|");
    Serial.println(last_status);
    
    delay(1000);
    loop_count++;
    return;
  }
  
  loop_count++;
  
  // Generate scenario data instead of reading sensors
  generateScenarioData();
  
  // Store history
  temp_history[history_idx] = temperature;
  gas_history[history_idx]  = gas_ppm;
  history_idx = (history_idx + 1) % 30;
  if (history_idx == 0) history_full = true;
  
  int   window_len = history_full ? 30 : max(history_idx, 1);
  float dT_avg     = (window_len > 1) ? rollingSlope(temp_history, window_len) : 0.0;
  float dG_avg     = (window_len > 1) ? rollingSlope(gas_history,  window_len) : 0.0;
  
  // Serial log in NEW format: time|temp|gas|presence|scenario|phase|status
  Serial.print(loop_count);
  Serial.print("|");
  Serial.print(temperature, 1);
  Serial.print("|");
  Serial.print(gas_ppm, 0);
  Serial.print("|");
  Serial.print(presence ? "YES" : "NO");
  Serial.print("|");
  Serial.print(current_scenario);
  Serial.print("|");
  Serial.print(current_phase);
  Serial.print("|");
  
  // LCD update - Top row
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(temperature, 1);
  lcd.print(" G:");
  lcd.print((int)gas_ppm);
  
  // ── LCD SECOND ROW ── Enhanced for paper demonstration
  lcd.setCursor(0, 1);
  
  // Show ML warning in ML_ZONE
  if (current_phase == "ML_ZONE") {
    lcd.print("ML Warning     ");
  }
  // Show Rule approaching in RULE_ZONE
  else if (current_phase == "RULE_ZONE") {
    lcd.print("Rule Near      ");
  }
  // Show triggered state
  else if (current_phase == "TRIGGERED") {
    lcd.print("!! TRIGGERED !!");
  }
  else {
    lcd.print("Status: SAFE   ");
  }
  
  // ── RULE ENGINE ──────────────────────────────────────────────
  
  // Rule 1: Boilover (Milk boilover detection)
  if (temperature >= TEMP_BOILOVER && dT_avg >= TEMP_RISE_RATE && gas_ppm > 200) {
    last_status = "RULE1_FIRED";
    Serial.println(last_status);
    triggerShutoff("OVERFLOW RISK");
    return;
  }
  
  // Rule 2: Gas leak (Gas leak with no flame)
  if (gas_ppm >= GAS_LEAK_THRESH && temperature < 80.0) {
    last_status = "RULE2_FIRED";
    Serial.println(last_status);
    triggerShutoff("GAS LEAK");
    return;
  }
  
  // Rule 3: Timeout (Person leaves kitchen)
  if (!presence && gas_ppm > 200) {
    no_pres_timer += 1;
  } else {
    no_pres_timer = 0;
  }
  if (no_pres_timer >= TIMEOUT_LIMIT) {  // FIX 2: Now 90s instead of 2700s
    last_status = "RULE3_FIRED";
    Serial.println(last_status);
    triggerShutoff("TIMEOUT 45min");
    return;
  }
  
  // Rule 4: Flame-out (Temperature dropping while gas continues)
  if (dT_avg < TEMP_DROP_RATE && dG_avg > 0.5 && gas_ppm > 300) {
    last_status = "RULE4_FIRED";
    Serial.println(last_status);
    triggerShutoff("FLAME-OUT");
    return;
  }
  
  // Safe
  last_status = "SAFE";
  Serial.println(last_status);
  
  delay(1000);
}

/* Simulation runner + recompilation
arduino-cli compile \
  --fqbn esp32:esp32:esp32 \
  --output-dir /Users/devkaransinghsarkaria/kitchen_safety/wokwi_sim/build \
  /Users/devkaransinghsarkaria/kitchen_safety/wokwi_sim
  
wokwi-cli --serial-log-file serial_output.txt --scenario scenario.yaml --timeout 120000 .
*/