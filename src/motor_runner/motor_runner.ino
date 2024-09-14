const int StepX1 = 12;
const int DirX1 = 13;
const int StepX2 = 3;
const int DirX2 = 6;
const int StepY = 4;
const int DirY = 7;

int targetX = 0;
int targetY = 0;
int targetZ = 0;

void setup() {
  Serial.begin(9600);
  pinMode(StepX1, OUTPUT);
  pinMode(DirX1, OUTPUT);
  pinMode(StepY, OUTPUT);
  pinMode(DirY, OUTPUT);
  pinMode(StepX2, OUTPUT);
  pinMode(DirX2, OUTPUT);
  
  digitalWrite(DirX1, HIGH);
  digitalWrite(DirY, HIGH);
  digitalWrite(DirX2, HIGH);
  
  Serial.println("Arduino ready");
}

void loop() {
  if (Serial.available() >= 3) {
    targetX = Serial.read();
    targetY = Serial.read();
    targetZ = Serial.read();
    
    Serial.print("Received: X=");
    Serial.print(targetX);
    Serial.print(", Y=");
    Serial.print(targetY);
    Serial.print(", Z=");
    Serial.println(targetZ);
    
    moveToPosition(targetX, targetY, targetZ);
    
    Serial.println("Movement completed");
  }
}

void moveToPosition(int x, int y, int z) {
  Serial.println("Moving X");
  moveAxis(StepX1, DirX1, x);
  Serial.println("Moving Y");
  moveAxis(StepY, DirY, y);
  Serial.println("Moving Z");
  moveAxis(StepX2, DirX2, z);
}

void moveAxis(int stepPin, int dirPin, int steps) {
  digitalWrite(dirPin, steps > 0 ? HIGH : LOW);
  steps = abs(steps);
  
  Serial.print("Moving ");
  Serial.print(steps);
  Serial.println(" steps");
  
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(2000);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(2000);
  }
}

