const int LED_PIN_1 = 2;  // Change these pin numbers as per your setup
const int LED_PIN_2 = 3;
const int LED_PIN_3 = 4;

void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN_1, OUTPUT);
  pinMode(LED_PIN_2, OUTPUT);
  pinMode(LED_PIN_3, OUTPUT);
}

void loop() {
  if (Serial.available() >= 12) {  // Expecting 4 bytes per int (3 ints total)
    int x = Serial.parseInt();
    int y = Serial.parseInt();
    int z = Serial.parseInt();
    
    digitalWrite(LED_PIN_1, HIGH);
    delay(x);
    digitalWrite(LED_PIN_1, LOW);
    
    digitalWrite(LED_PIN_2, HIGH);
    delay(y);
    digitalWrite(LED_PIN_2, LOW);
    
    digitalWrite(LED_PIN_3, HIGH);
    delay(z);
    digitalWrite(LED_PIN_3, LOW);
    
    // Clear any remaining data in the buffer
    while(Serial.available() > 0) {
      Serial.read();
    }
  }
}