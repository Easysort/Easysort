#include <AccelStepper.h>
#include <TimeLib.h>

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
    (input_x - input_y) + input_z, // z

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
    float x_ratio = abs_x / max_distance;
    float y_ratio = abs_y / max_distance;
    float z_ratio = abs_z / max_distance;
    
    float x_speed = x_ratio * MaxSpeed * (x >= 0 ? 1 : -1);
    float y_speed = y_ratio * MaxSpeed * (y >= 0 ? 1 : -1);
    float z_speed = z_ratio * MaxSpeed * (z >= 0 ? 1 : -1);
    
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

// Replace the time sync variables with:
unsigned long startupMillis = 0;  // Arduino's startup time in millis

// Replace getCurrentTime() with:
unsigned long getCurrentTime() {
    return millis() - startupMillis;
}

void setup() {
  // Set max speed and acceleration for each stepper
  pinMode(SUCTION_CUP_WRITE_PIN, OUTPUT);
  Serial.begin(115200);
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
  
  // Test pattern: blink 3 times then stay on for 3 seconds
  for(int i = 0; i < 3; i++) {
    digitalWrite(SUCTION_CUP_WRITE_PIN, HIGH);
    delay(500);
    digitalWrite(SUCTION_CUP_WRITE_PIN, LOW);
    delay(500);
  }
  
  digitalWrite(SUCTION_CUP_WRITE_PIN, HIGH);
  delay(3000);
  Serial.println("Suction cup should be off now");
  digitalWrite(SUCTION_CUP_WRITE_PIN, LOW);
  
  startupMillis = millis();  // Record startup time
  Serial.println("-READY-");
}

void loop() {
  if (Serial.available() > 0) {
    Serial.println("-NOT-READY-");
    String data = Serial.readStringUntil('\n');
    
    if (data.startsWith("SYNC?")) {
      Serial.println(getCurrentTime());
      Serial.println("-READY-");
      return;
    }

    // Log timestamp when receiving command
    double currentTime = getCurrentTime();
    Serial.print("T:");
    Serial.print(currentTime, 6);
    Serial.print(" CMD:");
    Serial.println(data);

    if (data.startsWith("pickup")) {
      // Remove "pickup(" and ")" from the string
      data = data.substring(7, data.length() - 1);
      
      // Split the string by ").(", which separates our coordinate groups
      int firstSplit = data.indexOf(").(");
      int secondSplit = data.indexOf(").(", firstSplit + 1);
      int thirdSplit = data.indexOf(").(", secondSplit + 1);
      
      String dropPos = data.substring(0, firstSplit);
      String transPos = data.substring(firstSplit + 3, secondSplit);
      String restPos = data.substring(secondSplit + 3, thirdSplit);
      
      // Define a struct for positions
      struct Position {
        int x, y, z, s;
      };
      
      // Function to process each position string
      auto processPosition = [](String pos) -> Position {
        int firstComma = pos.indexOf(',');
        int secondComma = pos.indexOf(',', firstComma + 1);
        int thirdComma = pos.indexOf(',', secondComma + 1);
        
        Position p;
        p.x = pos.substring(0, firstComma).toInt();
        p.y = pos.substring(firstComma + 1, secondComma).toInt();
        p.z = pos.substring(secondComma + 1, thirdComma).toInt();
        p.s = pos.substring(thirdComma + 1).toInt();
        return p;
      };
      
      // Process each position
      Position drop = processPosition(dropPos);
      Position trans = processPosition(transPos);
      Position rest = processPosition(restPos);
      
      // Execute the movement sequence
      digitalWrite(SUCTION_CUP_WRITE_PIN, drop.s);
      moveCoordinated(drop.x, drop.y, drop.z);
      
      digitalWrite(SUCTION_CUP_WRITE_PIN, trans.s);
      moveCoordinated(trans.x, trans.y, trans.z);
      
      digitalWrite(SUCTION_CUP_WRITE_PIN, rest.s);
      moveCoordinated(rest.x, rest.y, rest.z);
    } else {
      // Handle simple position commands (legacy format)
      int firstComma = data.indexOf(',');
      int secondComma = data.indexOf(',', firstComma + 1);
      int thirdComma = data.indexOf(',', secondComma + 1);
      
      if (firstComma != -1 && secondComma != -1 && thirdComma != -1) {
        int x = data.substring(0, firstComma).toInt();
        int y = data.substring(firstComma + 1, secondComma).toInt();
        int z = data.substring(secondComma + 1, thirdComma).toInt();
        int suctionState = data.substring(thirdComma + 1).toInt();
        
        digitalWrite(SUCTION_CUP_WRITE_PIN, suctionState);
        moveCoordinated(x, y, z);
      }
    }
    
    Serial.println("-READY-");
  }
}