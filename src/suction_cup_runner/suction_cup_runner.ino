#include <Servo.h>

Servo myservo0;
Servo myservo1;

void setup() 
{
  Serial.begin(9600);
  myservo0.attach(9);
  myservo1.attach(8);

  myservo0.write(0);  
  myservo1.write(0);  
}

void loop() 
{
  if (Serial.available() > 0) {
    int command = Serial.read();
    if (command == '1') {
      myservo0.write(180);  // Turn on the pump
    } else if (command == '0') {
      myservo0.write(0);    // Turn off the pump
    }
  }
}