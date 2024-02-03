
#define stepPin 0
#define dirPin 3

void setup() {
  pinMode(stepPin,OUTPUT);
  pinMode(dirPin,OUTPUT);

}

void loop() {
  digitalWrite(dirPin,HIGH);
  // put your main code here, to run repeatedly:
  for(int x = 0; x < 200; x++) {
    digitalWrite(stepPin,HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPin,LOW);
    delayMicroseconds(1000);
  }
  delay(1000);

  digitalWrite(dirPin,LOW);
  // put your main code here, to run repeatedly:
  for(int x = 0; x < 400; x++) {
    digitalWrite(stepPin,HIGH);
    delayMicroseconds(700);
    digitalWrite(stepPin,LOW);
    delayMicroseconds(700);
  }
  delay(1000);

}
