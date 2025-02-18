#include <AccelStepper.h>

// Steps per cm for each axis (adjust based on your setup)
const float CMS_FOR_10_REVOLUTIONS = 60;
const float CM_PER_REVOLUTION = CMS_FOR_10_REVOLUTIONS / 10;
const float STEPS_PER_CM_X = 800/CM_PER_REVOLUTION;
const float STEPS_PER_CM_Y = 800/CM_PER_REVOLUTION;
const float STEPS_PER_CM_Z = 800/CM_PER_REVOLUTION;
const float CONVEYOR_SPEED = 8.727; // cm/s
// const int LOWEST_Z = 10;
// const int HIGHEST_Z = 50;

// const int SUCTION_CUP_WRITE_PIN = 12;
// const int SUCTION_CUP_READ_PIN = 13;
// const int LIMIT_SWITCH_READ_PIN = 11;

const int MaxSpeed = 20000;
const int Acceleration = 20000;

const int XMaxSpeed = MaxSpeed;
const int XAcceleration = Acceleration;
const int YMaxSpeed = MaxSpeed;
const int YAcceleration = Acceleration;
const int ZMaxSpeed = MaxSpeed;
const int ZAcceleration = Acceleration;

long x, y;

// Define stepper motors. Adjust pin numbers as needed.
AccelStepper stepperX(AccelStepper::DRIVER, 2, 3);
AccelStepper stepperY(AccelStepper::DRIVER, 6, 7);
// AccelStepper stepperZ(AccelStepper::DRIVER, 9, 8);
// ezButton limitSwitch(LIMIT_SWITCH_READ_PIN); // when z is touching, limitSwitch is LOW

void setup() {
  // Set max speed and acceleration for each stepper
  Serial.begin(9600);
  Serial.println("Starting...");
  delay(4000); // Avoid starting failure of running first moveCoordinated before arduino setup
  stepperX.setMaxSpeed(XMaxSpeed);
  stepperX.setAcceleration(XAcceleration);
  stepperY.setMaxSpeed(YMaxSpeed);
  stepperY.setAcceleration(YAcceleration);
  // stepperZ.setMaxSpeed(ZMaxSpeed);
  // stepperZ.setAcceleration(ZAcceleration);

  // Starting point is (0, 0)
  Serial.println("0/3...");
  moveCoordinated(15, 0);
  moveCoordinated(0, 0);
  Serial.println("1/3...");
  moveCoordinated(0, 15);
  moveCoordinated(0, 0);
  Serial.println("2/3...");
  moveCoordinated(15, 10);
  moveCoordinated(10, 15);
  moveCoordinated(0, 0);
  Serial.println("3/3...");
  Serial.println("Done!");
}

void loop() {
  // if (Serial.available() > 0) {
  //   String data = Serial.readStringUntil('\n'); // input is x,y\n
  //   int comma = data.indexOf(',');
  //   if (comma != -1) {
  //     int num1 = data.substring(0, comma).toInt();
  //     int num2 = data.substring(comma + 1).toInt();

  //     moveCoordinated(num1, num2)
  //     // add error handling
  //     Serial.println("success") // or "fail" if error
  //     // error could be:
  //     // - Cannot lift item
  //     // - Dropped item on the way
  // }
}

void reverseKinematics(int input_x, int input_y) { // Position -> robot movements 
  // Converts normal x, y coordinates into what the motors have to do.
  x = input_x + input_y;
  y = input_x - input_y;
  // z = x + z_is_up * HIGHEST_Z;
}

void moveCoordinated(float future_x, float future_y) {
  long target_x = future_x * STEPS_PER_CM_X;
  long target_y = future_y * STEPS_PER_CM_Y;
  reverseKinematics(target_x, target_y);
  if (x >= y) {
    float speedRatio = static_cast<float>(x) / y;
    stepperX.setSpeed(XMaxSpeed * speedRatio);
    stepperY.setSpeed(XMaxSpeed);
  } else {
    float speedRatio = static_cast<float>(x) / y;
    stepperX.setSpeed(XMaxSpeed * speedRatio);
    stepperY.setSpeed(XMaxSpeed);
  }
  stepperX.moveTo(x);
  stepperY.moveTo(y);
  while (stepperX.distanceToGo() != 0 || stepperY.distanceToGo() != 0) {
    stepperX.run();
    stepperY.run();
  }
}