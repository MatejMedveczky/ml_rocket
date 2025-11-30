#include <Arduino.h>

bool idle_state();
bool launch_state();
bool coast_ascent_state();
bool coast_descent_state();
bool para_descent_state();
bool landing_state();
bool recovery_state();

enum State{
  IDLE,
  LAUNCH,
  COAST_ASCENT,
  COAST_DESCENT,
  PARA_DESCENT,
  LANDING,
  RECOVERY  
};

struct {
  State current_state;
  float altitude;
  float orientation[3];
  float acceleration[3];
  float gyro[3];
  float gimbal[2];
  double flight_time;
  unsigned long start_time;
  unsigned long current_time;
  unsigned long last_accel_check;  // Add: timestamp of last acceleration check
  float previous_acceleration;      // Add: stores previous acceleration reading
} rocket_data;

void setup() {
  rocket_data.current_state = IDLE;
  rocket_data.altitude = 0.0;
  rocket_data.flight_time = 0.0;
  rocket_data.start_time = 0;
  rocket_data.current_time = 0;
  rocket_data.last_accel_check = 0;  
  rocket_data.previous_acceleration = 0.0;  // Add: initialize

  constexpr int PIN_LED         = 8;   // on-board LED (example)
  constexpr int PIN_PYRO_1      = 2;   // pyro/actuator channel 1
  constexpr int PIN_PYRO_2      = 3;   // pyro/actuator channel 2
  constexpr int PIN_GIMBAL_PWM1 = 10;  // gimbal PWM output 1
  constexpr int PIN_GIMBAL_PWM2 = 11;  // gimbal PWM output 2
  constexpr int PIN_IMU_INT     = 5;   // IMU interrupt input
  constexpr int PIN_SD_CS       = 7;   // SD card CS
  constexpr int PIN_BATT_MON    = 4;   // battery voltage analog input

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  pinMode(PIN_PYRO_1, OUTPUT);
  digitalWrite(PIN_PYRO_1, LOW);
  pinMode(PIN_PYRO_2, OUTPUT);
  digitalWrite(PIN_PYRO_2, LOW);

  // Inputs
  pinMode(PIN_IMU_INT, INPUT_PULLUP);

  // SD / SPI CS
  pinMode(PIN_SD_CS, OUTPUT);
  digitalWrite(PIN_SD_CS, HIGH);

  // ADC: analogRead(PIN_BATT_MON) can be used without extra pinMode

  // PWM (LEDC) setup for gimbal servos on ESP32-C3
  const int GIMBAL_CH1 = 0;
  const int GIMBAL_CH2 = 1;
  const int PWM_FREQ   = 50; // typical servo frequency
  const int PWM_RES    = 16; // resolution bits

  ledcSetup(GIMBAL_CH1, PWM_FREQ, PWM_RES);
  ledcSetup(GIMBAL_CH2, PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_GIMBAL_PWM1, GIMBAL_CH1);
  ledcAttachPin(PIN_GIMBAL_PWM2, GIMBAL_CH2);

  // Center servos (midpoint for given resolution)
  uint32_t pwm_center = (1u << (PWM_RES - 1));
  ledcWrite(GIMBAL_CH1, pwm_center);
  ledcWrite(GIMBAL_CH2, pwm_center);

  int result = myFunction(2, 3);
}

void loop() {
  rocket_data.current_time = millis();
  if (rocket_data.start_time > 0) {
    rocket_data.flight_time = (rocket_data.current_time - rocket_data.start_time) / 100.0; 
  }

  switch (rocket_data.current_state) {
    case IDLE:
      idle_state();
      break;
    case LAUNCH:
      launch_state();
      break;
    case COAST_ASCENT:
      coast_ascent_state();
      break;
    case COAST_DESCENT:
      coast_descent_state();
      break;
    case PARA_DESCENT:
      para_descent_state();
      break;
    case LANDING:
      landing_state();
      break;
    case RECOVERY:
      recovery_state();
      break;
    default:
      idle_state();
      break;
  }
}

bool idle_state() {
  if (rocket_data.current_time - rocket_data.last_accel_check >= 500) {
    float current_acceleration = rocket_data.acceleration[0]; 
    
    if (current_acceleration > rocket_data.previous_acceleration && current_acceleration > 1.0) {
      rocket_data.start_time = millis();
      rocket_data.current_state = LAUNCH;
      return true;
    }
    
    rocket_data.previous_acceleration = current_acceleration;
    rocket_data.last_accel_check = rocket_data.current_time;
  }
  
  return true;
}

bool launch_state() {
  rocket_data.current_state = COAST_ASCENT;
  return true;
}

bool coast_ascent_state() {
  rocket_data.current_state = COAST_DESCENT;
  return true;
}

bool coast_descent_state() {
  rocket_data.current_state = PARA_DESCENT;
  return true;
}

bool para_descent_state() {
  rocket_data.current_state = LANDING;
  return true;
}

bool landing_state() {
  rocket_data.current_state = RECOVERY;
  return true;
}

bool recovery_state() {
  return true;
}