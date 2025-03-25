#include <AccelStepper.h>

const float CMS_FOR_10_REVOLUTIONS = 60;
const float CM_PER_REVOLUTION = CMS_FOR_10_REVOLUTIONS / 10;
const float STEPS_PER_CM_X = 800 / CM_PER_REVOLUTION;
const float STEPS_PER_CM_Y = 800 / CM_PER_REVOLUTION;
const float STEPS_PER_CM_Z = 800 / CM_PER_REVOLUTION;
const float CONVEYOR_SPEED = 8.727;  // cm/s

const long MaxSpeed = 1000; // Can go to 8000 at least
const long Acceleration = 1000; // Can go to 8000 at least
const int SUCTION_CUP_WRITE_PIN = 5;

const long XMaxSpeed = MaxSpeed;
const long XAcceleration = Acceleration;
const long YMaxSpeed = MaxSpeed;
const long YAcceleration = Acceleration;
const long ZMaxSpeed = MaxSpeed;
const long ZAcceleration = Acceleration;
constexpr double SpeedCoefficientHorizontal = 1.2; // Allowed larger than 1.0 when the horizontal space is small.
constexpr double SpeedCoefficientVertical = 1.2; // Allowed larger than 1.0 when the height is small.

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
    input_x - input_y, // y
    input_x + input_z, // z

  };
}

void moveCoordinated(float future_x, float future_y, float future_z) {
  // unsigned long start_time = millis();
  
  long target_x = future_x * STEPS_PER_CM_X;
  long target_y = future_y * STEPS_PER_CM_Y;
  long target_z = future_z * STEPS_PER_CM_Z;
  Coordinates coords = reverseKinematics(target_x, target_y, target_z);

  long x = coords.x - stepperX.currentPosition();
  long y = coords.y - stepperY.currentPosition();
  long z = coords.z - stepperZ.currentPosition();
  
  stepperX.moveTo(coords.x);
  stepperY.moveTo(coords.y);
  stepperZ.moveTo(coords.z);
  
  float abs_x = abs(coords.x - stepperX.currentPosition());
  float abs_y = abs(coords.y - stepperY.currentPosition());
  float abs_z = abs(coords.z - stepperZ.currentPosition());
  float max_distance = max(abs_x, max(abs_y, abs_z));

  unsigned long last_x_step = micros();
  unsigned long last_y_step = micros();
  unsigned long last_z_step = micros();
  
  if (max_distance > 0) {
    float length = sqrt(abs_x * abs_x + abs_y * abs_y + abs_z * abs_z);
    float x_ratio = abs_x / length;
    float y_ratio = abs_y / length;
    float z_ratio = abs_z / length;
    
    float x_speed = SpeedCoefficientHorizontal * x_ratio * MaxSpeed * (x >= 0 ? 1 : -1);
    float y_speed = SpeedCoefficientHorizontal * y_ratio * MaxSpeed * (y >= 0 ? 1 : -1);
    float z_speed = SpeedCoefficientVertical * z_ratio * MaxSpeed * (z >= 0 ? 1 : -1);
    
    // Serial.print("Setting motor speeds - X: ");
    // Serial.print(x_speed);
    // Serial.print(" Y: ");
    // Serial.print(y_speed);
    // Serial.print(" Z: ");
    // Serial.println(z_speed);
    
    stepperX.setSpeed(x_speed);
    stepperY.setSpeed(y_speed);
    stepperZ.setSpeed(z_speed);
    
    unsigned long x_interval = abs(1000000.0 / x_speed);  // microseconds between steps
    unsigned long y_interval = abs(1000000.0 / y_speed);  // microseconds between steps
    unsigned long z_interval = abs(1000000.0 / z_speed);  // microseconds between steps

    last_x_step = micros();
    last_y_step = micros();
    last_z_step = micros();
    
    while (stepperX.distanceToGo() != 0 || stepperY.distanceToGo() != 0 || stepperZ.distanceToGo() != 0) {
      unsigned long now = micros();
      
      if (stepperX.distanceToGo() != 0 && (now - last_x_step) >= x_interval) {
        stepperX.runSpeed();
        last_x_step = now;
      }
      
      if (stepperY.distanceToGo() != 0 && (now - last_y_step) >= y_interval) {
        stepperY.runSpeed();
        last_y_step = now;
      }

      if (stepperZ.distanceToGo() != 0 && (now - last_z_step) >= z_interval) {
        stepperZ.runSpeed();
        last_z_step = now;
      }
    }
  }

  // unsigned long time_taken = millis() - start_time;
  // Serial.print("Movement took ");
  // Serial.print(time_taken);
  // Serial.println(" ms");
  // Serial.print("X ended at time: ");
  // Serial.println(last_x_step);
  // Serial.print("Y ended at time: ");
  // Serial.println(last_y_step);
  // Serial.print("Z ended at time: ");
  // Serial.println(last_z_step);
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
  stepperZ.setMaxSpeed(ZMaxSpeed);
  stepperZ.setAcceleration(ZAcceleration);

  Serial.println("Testing suction cup");
  Serial.println("Suction cup should be on for 3 seconds");
  digitalWrite(SUCTION_CUP_WRITE_PIN, HIGH);
  delay(3000);
  Serial.println("Suction cup should be off now");
  digitalWrite(SUCTION_CUP_WRITE_PIN, LOW);
  Serial.println("-READY-");
}

void loop() {
  if (Serial.available() > 0) {
    Serial.println("-NOT-READY-");
    // Serial.println("Received data:");
    String data = Serial.readStringUntil('\n'); // input is x,y,z,suction\n
    int firstComma = data.indexOf(',');
    int secondComma = data.indexOf(',', firstComma + 1);
    int thirdComma = data.indexOf(',', secondComma + 1);
    
    if (firstComma != -1 && secondComma != -1) {
      int num1 = data.substring(0, firstComma).toInt();
      int num2 = data.substring(firstComma + 1, secondComma).toInt();
      int num3 = data.substring(secondComma + 1, thirdComma).toInt();
      int suctionState = data.substring(thirdComma + 1).toInt();
      // Serial.println("Data is correctly formatted:");
      // Serial.println(num1);
      // Serial.println(num2);
      // Serial.println(num3);
      // Serial.println(suctionState);
      if (suctionState == 1) {
        digitalWrite(SUCTION_CUP_WRITE_PIN, HIGH); // on
      } else {
        digitalWrite(SUCTION_CUP_WRITE_PIN, LOW); // off
      }
      moveCoordinated(num1, num2, num3);
    }
    Serial.println("-READY-");
  }
}