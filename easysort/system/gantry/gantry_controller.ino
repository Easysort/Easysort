#include <AccelStepper.h>

// Steps per cm for each axis (adjust based on your setup)
const float CMS_FOR_10_REVOLUTIONS = 60;
const float CM_PER_REVOLUTION = CMS_FOR_10_REVOLUTIONS / 10;
const float STEPS_PER_CM_X = 800/CM_PER_REVOLUTION;
const float STEPS_PER_CM_Y = 800/CM_PER_REVOLUTION;
const float STEPS_PER_CM_Z = 800/CM_PER_REVOLUTION;
const float CONVEYOR_SPEED = 8.727; // cm/s
const int LOWEST_Z = 10;
const int HIGHEST_Z = 50;

const int SUCTION_CUP_WRITE_PIN = 12;
const int SUCTION_CUP_READ_PIN = 13;
const int LIMIT_SWITCH_READ_PIN = 11;

const int XMaxSpeed = 200;
const int XAcceleration = 100;
const int YMaxSpeed = 200;
const int YAcceleration = 100;
const int ZMaxSpeed = 200;
const int ZAcceleration = 100;

// Define stepper motors. Adjust pin numbers as needed.
AccelStepper stepperX(AccelStepper::DRIVER, 3, 2); //stepPin = 2, dirPin = 3
AccelStepper stepperY(AccelStepper::DRIVER, 7, 6);
// AccelStepper stepperZ(AccelStepper::DRIVER, 9, 8);
// ezButton limitSwitch(LIMIT_SWITCH_READ_PIN); // when z is touching, limitSwitch is LOW

void setup() {
  // Set max speed and acceleration for each stepper
  stepperX.setMaxSpeed(XMaxSpeed);
  stepperX.setAcceleration(XAcceleration);
  stepperY.setMaxSpeed(YMaxSpeed);
  stepperY.setAcceleration(YAcceleration);
  stepperZ.setMaxSpeed(ZMaxSpeed);
  stepperZ.setAcceleration(ZAcceleration);

  moveCoordinated(30, 15);  // Move to (10cm, 15cm, -5cm)
  moveCoordinated(10, 30);
  moveCoordinated(30, 10);
  moveCoordinated(30, 15);
}

// void loop() {
//   if (Serial.available() > 0) {
//     String data = Serial.readStringUntil('\n'); // input is x,y\n
//     int comma = data.indexOf(',');
//     if (comma != -1) {
//       int num1 = data.substring(0, comma).toInt();
//       int num2 = data.substring(comma + 1).toInt();

//       moveCoordinated(num1, num2)
//       // add error handling
//       Serial.println("success") // or "fail" if error
//       // error could be:
//       // - Cannot lift item
//       // - Dropped item on the way
//   }
// }

void moveCoordinated(float future_x, float future_y) {
  long target_x = future_x * STEPS_PER_CM_X;
  long target_y = future_y * STEPS_PER_CM_Y;
  if (target_x >= target_y) {
    float speedRatio = static_cast<float>(target_y) / target_x;
    stepperX.setSpeed(maxSpeed * speedRatio);
    stepperY.setSpeed(maxSpeed);
  } else {
    float speedRatio = static_cast<float>(target_x) / target_y;
    stepperX.setSpeed(maxSpeed * speedRatio);
    stepperY.setSpeed(maxSpeed);
  }
  stepperX.moveTo(target_x);
  stepperY.moveTo(target_y);
  while (stepperX.distanceToGo() != 0 || stepperY.distanceToGo() != 0) {
    stepperX.run();
    stepperY.run();
  }
}