#include <AccelStepper.h>

const float CMS_FOR_10_REVOLUTIONS = 60;
const float CM_PER_REVOLUTION = CMS_FOR_10_REVOLUTIONS / 10;
const float STEPS_PER_CM_X = 800 / CM_PER_REVOLUTION;
const float STEPS_PER_CM_Y = 800 / CM_PER_REVOLUTION;
const float STEPS_PER_CM_Z = 800 / CM_PER_REVOLUTION;
const float CONVEYOR_SPEED = 8.727;  // cm/s
// const int LOWEST_Z = 10;
// const int HIGHEST_Z = 50;

// const int SUCTION_CUP_WRITE_PIN = 12;
// const int SUCTION_CUP_READ_PIN = 13;
// const int LIMIT_SWITCH_READ_PIN = 11;

const long MaxSpeed = 4000; // Can go to 8000 at least
const long Acceleration = 4000; // Can go to 8000 at least

const long XMaxSpeed = MaxSpeed;
const long XAcceleration = Acceleration;
const long YMaxSpeed = MaxSpeed;
const long YAcceleration = Acceleration;
const long ZMaxSpeed = MaxSpeed;
const long ZAcceleration = Acceleration;

struct Coordinates {
  long x;
  long y;
};

// Define stepper motors. Adjust pin numbers as needed.
AccelStepper stepperX(AccelStepper::DRIVER, 2, 3);
AccelStepper stepperY(AccelStepper::DRIVER, 6, 7);
// AccelStepper stepperZ(AccelStepper::DRIVER, 9, 8);
// ezButton limitSwitch(LIMIT_SWITCH_READ_PIN); // when z is touching, limitSwitch is LOW

void setup() {
  // Set max speed and acceleration for each stepper
  Serial.begin(9600);
  Serial.println("Starting...");
  delay(4000);  // Avoid starting failure of running first moveCoordinated before arduino setup
  stepperX.setMaxSpeed(XMaxSpeed);
  stepperX.setAcceleration(XAcceleration);
  stepperY.setMaxSpeed(YMaxSpeed);
  stepperY.setAcceleration(YAcceleration);
  // stepperZ.setMaxSpeed(ZMaxSpeed);
  // stepperZ.setAcceleration(ZAcceleration);

  // Starting point is (0, 0)
  Serial.println("0/3...");
  moveCoordinated(5, 0);
  moveCoordinated(0, 0);
  Serial.println("1/3...");
  moveCoordinated(0, 5);
  moveCoordinated(0, 0);
  Serial.println("2/3...");
  moveCoordinated(5, 5);
  moveCoordinated(5, -5);
  moveCoordinated(-5, -5);
  moveCoordinated(-5, 5);
  moveCoordinated(0, 0);
  Serial.println("3/3...");
  Serial.println("Done!");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); // input is x,y\n
    int comma = data.indexOf(',');
    if (comma != -1) {
      int num1 = data.substring(0, comma).toInt();
      int num2 = data.substring(comma + 1).toInt();

      moveCoordinated(num1, num2);
      // add error handling
      Serial.println("success"); // or "fail" if error
      // error could be:
      // - Cannot lift item
      // - Dropped item on the way
    }
  }
}

Coordinates reverseKinematics(int input_x, int input_y) {  // Position -> robot movements
  // Converts normal x, y coordinates into what the motors have to do.
  return {
    input_x + input_y, // x
    input_x - input_y  // y
  };
  // z = x + z_is_up * HIGHEST_Z;
}

void moveCoordinated(float future_x, float future_y) {
  unsigned long start_time = millis();
  
  long target_x = future_x * STEPS_PER_CM_X;
  long target_y = future_y * STEPS_PER_CM_Y;
  
  Coordinates coords = reverseKinematics(target_x, target_y);
  
  long x = coords.x - stepperX.currentPosition();
  long y = coords.y - stepperY.currentPosition();
  
  float abs_x = abs(x);
  float abs_y = abs(y);
  float max_distance = max(abs_x, abs_y);
  
  if (max_distance > 0) {
    float total_distance = sqrt(abs_x * abs_x + abs_y * abs_y);
    float x_ratio = abs_x / total_distance;
    float y_ratio = abs_y / total_distance;
    
    float x_speed = x_ratio * MaxSpeed * (x >= 0 ? 1 : -1);
    float y_speed = y_ratio * MaxSpeed * (y >= 0 ? 1 : -1);
    
    Serial.print("Setting motor speeds - X: ");
    Serial.print(x_speed);
    Serial.print(" Y: ");
    Serial.println(y_speed);
    
    stepperX.moveTo(coords.x);
    stepperY.moveTo(coords.y);
    
    stepperX.setSpeed(x_speed);
    stepperY.setSpeed(y_speed);
    
    // Calculate step intervals in microseconds
    unsigned long x_interval = abs(1000000.0 / x_speed);  // microseconds between steps
    unsigned long y_interval = abs(1000000.0 / y_speed);  // microseconds between steps
    
    unsigned long last_x_step = micros();
    unsigned long last_y_step = micros();
    
    int x_dir = x >= 0 ? 1 : -1;
    int y_dir = y >= 0 ? 1 : -1;
    long x_target = coords.x;
    long y_target = coords.y;
    
    while (stepperX.distanceToGo() != 0 || stepperY.distanceToGo() != 0) {
      unsigned long now = micros();
      unsigned long now_millis = millis();
      
      // Only step if we haven't reached the target yet
      if (stepperX.distanceToGo() != 0 && (now - last_x_step) >= x_interval) {
        if ((x_dir > 0 && stepperX.currentPosition() < x_target) ||
            (x_dir < 0 && stepperX.currentPosition() > x_target)) {
          stepperX.runSpeed();
        }
        last_x_step = now;
      }
      
      if (stepperY.distanceToGo() != 0 && (now - last_y_step) >= y_interval) {
        if ((y_dir > 0 && stepperY.currentPosition() < y_target) ||
            (y_dir < 0 && stepperY.currentPosition() > y_target)) {
          stepperY.runSpeed();
        }
        last_y_step = now;
      }
      
    }
  }

  unsigned long time_taken = millis() - start_time;
  Serial.print("Movement took ");
  Serial.print(time_taken);
  Serial.println(" ms");
}