#include <Servo.h>

Servo myservo0;
Servo myservo1;

const int SUCTION_CUP_READ_PIN = 12;
const int SUCTION_CUP_WRITE_PIN = 13; // these should be opposite of what is on controller arduino

void setup() 
{
  Serial.begin(9600);
  myservo0.attach(9);
  myservo1.attach(8);

  myservo0.write(0);  
  myservo1.write(0);

  pinMode(SUCTION_CUP_WRITE_PIN, OUTPUT);
  pinMode(SUCTION_CUP_READ_PIN, INPUT_PULLUP); 
}

void loop() 
{
  Serial.println("Checking...");
  
  const int numReadings = 20;
  int totalReadings = 0;
  
  // Take 25 readings and sum them up
  for (int i = 0; i < numReadings; i++) {
    totalReadings += digitalRead(SUCTION_CUP_READ_PIN);
    delay(10); // Short delay between readings
  }
  
  // Calculate the average
  float averageReading = (float)totalReadings / numReadings;
  
  Serial.print("Average reading: ");
  Serial.println(averageReading);
  
  // Use the average to determine the action
  if (averageReading > 0.99) { // If more than half of the readings were HIGH
    myservo0.write(180);  // Turn on the pump
    digitalWrite(SUCTION_CUP_WRITE_PIN, HIGH);
  } else {
    myservo0.write(0);    // Turn off the pump
    digitalWrite(SUCTION_CUP_WRITE_PIN, LOW);
  }
}