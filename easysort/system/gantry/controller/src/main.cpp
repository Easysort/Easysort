#include <AccelStepper.h>

const float CMS_FOR_10_REVOLUTIONS = 60;
const float CM_PER_REVOLUTION = CMS_FOR_10_REVOLUTIONS / 10;
const float STEPS_PER_CM_X = 800 / CM_PER_REVOLUTION;
const float STEPS_PER_CM_Y = 800 / CM_PER_REVOLUTION;
const float STEPS_PER_CM_Z = 800 / CM_PER_REVOLUTION;
const float CONVEYOR_SPEED = 8.727;  // cm/s

const long MaxSpeed = 4000; // Can go to 8000 at least
const long Acceleration = 4000; // Can go to 8000 at least
const int SUCTION_CUP_WRITE_PIN = 5;

const long XMaxSpeed = MaxSpeed;
const long XAcceleration = Acceleration;
const long YMaxSpeed = MaxSpeed;
const long YAcceleration = Acceleration;
const long ZMaxSpeed = MaxSpeed;
const long ZAcceleration = Acceleration;

struct Coordinates {
  long x;
  long y;
  long z;
};

// Define stepper motors. Adjust pin numbers as needed.
AccelStepper stepperX(AccelStepper::DRIVER, 2, 3);
AccelStepper stepperY(AccelStepper::DRIVER, 6, 7);
AccelStepper stepperZ(AccelStepper::DRIVER, 8, 9);
// ezButton limitSwitch(LIMIT_SWITCH_READ_PIN); // when z is touching, limitSwitch is LOW

Coordinates reverseKinematics(int input_x, int input_y, int input_z) {  // Position -> robot movements
  // Converts normal x, y coordinates into what the motors have to do.
  return {
    input_x + input_y, // x
    input_x - input_y,  // y
    input_x + input_z, // z

  };
}

void moveCoordinated(float future_x, float future_y, float future_z) {
  unsigned long start_time = millis();
  
  long target_x = future_x * STEPS_PER_CM_X;
  long target_y = future_y * STEPS_PER_CM_Y;
  long target_z = future_z * STEPS_PER_CM_Z;
  Coordinates coords = reverseKinematics(target_x, target_y, target_z);
  
  long x = coords.x - stepperX.currentPosition();
  long y = coords.y - stepperY.currentPosition();
  long z = target_z - stepperZ.currentPosition();

  Serial.print("x: ");
  Serial.println(x);
  Serial.print("y: ");
  Serial.println(y);
  Serial.print("z: ");
  Serial.println(z);
  
  float abs_x = abs(x);
  float abs_y = abs(y);
  float abs_z = abs(z);
  float max_distance = max(abs_x, max(abs_y, abs_z));
  
  if (max_distance > 0) {
    float total_distance = sqrt(abs_x * abs_x + abs_y * abs_y + abs_z * abs_z);
    float x_ratio = abs_x / total_distance;
    float y_ratio = abs_y / total_distance;
    float z_ratio = abs_z / total_distance;
    
    float x_speed = x_ratio * MaxSpeed * (x >= 0 ? 1 : -1);
    float y_speed = y_ratio * MaxSpeed * (y >= 0 ? 1 : -1);
    float z_speed = z_ratio * MaxSpeed * (z >= 0 ? 1 : -1);
    
    Serial.print("Setting motor speeds - X: ");
    Serial.print(x_speed);
    Serial.print(" Y: ");
    Serial.print(y_speed);
    Serial.print(" Z: ");
    Serial.println(z_speed);
    
    stepperX.moveTo(coords.x);
    stepperY.moveTo(coords.y);
    stepperZ.moveTo(coords.z);
    stepperX.setSpeed(x_speed);
    stepperY.setSpeed(y_speed);
    stepperZ.setSpeed(z_speed);
    
    // Calculate step intervals in microseconds
    unsigned long x_interval = abs(1000000.0 / x_speed);  // microseconds between steps
    unsigned long y_interval = abs(1000000.0 / y_speed);  // microseconds between steps
    unsigned long z_interval = abs(1000000.0 / z_speed);  // microseconds between steps
    
    unsigned long last_x_step = micros();
    unsigned long last_y_step = micros();
    unsigned long last_z_step = micros();
    
    int x_dir = x >= 0 ? 1 : -1;
    int y_dir = y >= 0 ? 1 : -1;
    int z_dir = z >= 0 ? 1 : -1;
    long x_target = coords.x;
    long y_target = coords.y;
    long z_target = coords.z;
    
    while (stepperX.distanceToGo() != 0 || stepperY.distanceToGo() != 0 || stepperZ.distanceToGo() != 0) {
      unsigned long now = micros();
      
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

      if (stepperZ.distanceToGo() != 0 && (now - last_z_step) >= z_interval) {
        if ((z_dir > 0 && stepperZ.currentPosition() < z_target) ||
            (z_dir < 0 && stepperZ.currentPosition() > z_target)) {
          stepperZ.runSpeed();
        }
        last_z_step = now;
      }
      
    }
  }

  unsigned long time_taken = millis() - start_time;
  Serial.print("Movement took ");
  Serial.print(time_taken);
  Serial.println(" ms");
}

void setup() {
  // Set max speed and acceleration for each stepper
  pinMode(SUCTION_CUP_WRITE_PIN, OUTPUT);
  Serial.begin(9600);
  Serial.println("Starting...");
  delay(4000);  // Avoid starting failure of running first moveCoordinated before arduino setup
  stepperX.setMaxSpeed(XMaxSpeed);
  stepperX.setAcceleration(XAcceleration);
  stepperY.setMaxSpeed(YMaxSpeed);
  stepperY.setAcceleration(YAcceleration);

  Serial.println("Testing suction cup");
  Serial.println("Suction cup should be on for 3 seconds");
  digitalWrite(SUCTION_CUP_WRITE_PIN, HIGH);
  delay(3000);
  Serial.println("Suction cup should be off now");
  digitalWrite(SUCTION_CUP_WRITE_PIN, LOW);
  // delay(3000);

  // stepperZ.setMaxSpeed(ZMaxSpeed);
  // stepperZ.setAcceleration(ZAcceleration);

  // Starting point is (0, 0)
  // Serial.println("0/3...");
  // moveCoordinated(5, 0);
  // moveCoordinated(0, 0);
  // Serial.println("1/3...");
  // moveCoordinated(0, 5);
  // moveCoordinated(0, 0);
  // Serial.println("2/3...");
  // moveCoordinated(5, 5);
  // moveCoordinated(5, -5);
  // moveCoordinated(-5, -5);
  // moveCoordinated(-5, 5);
  // moveCoordinated(0, 0);
  // Serial.println("3/3...");
  Serial.println("Done!");
}

void loop() {
  if (Serial.available() > 0) {
    Serial.println("Received data:");
    String data = Serial.readStringUntil('\n'); // input is x,y,z,suction\n
    int firstComma = data.indexOf(',');
    int secondComma = data.indexOf(',', firstComma + 1);
    int thirdComma = data.indexOf(',', secondComma + 1);
    
    if (firstComma != -1 && secondComma != -1) {
      int num1 = data.substring(0, firstComma).toInt();
      int num2 = data.substring(firstComma + 1, secondComma).toInt();
      int num3 = data.substring(secondComma + 1, thirdComma).toInt();
      int suctionState = data.substring(thirdComma + 1).toInt();
      Serial.println("Received data:");
      Serial.println(num1);
      Serial.println(num2);
      Serial.println(num3);
      Serial.println(suctionState);
      if (suctionState == 1) {
        digitalWrite(SUCTION_CUP_WRITE_PIN, HIGH); // on
      } else {
        digitalWrite(SUCTION_CUP_WRITE_PIN, LOW); // off
      }
      moveCoordinated(num1, num2, num3);

      // add error handling
      Serial.println("success"); // or "fail" if error
      // error could be:
      // - Cannot lift item
      // - Dropped item on the way
    }
  }
}