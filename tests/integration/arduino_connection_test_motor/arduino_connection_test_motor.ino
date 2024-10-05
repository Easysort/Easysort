

void setup() 
{
  Serial.begin(9600);
}

void loop() 
{
  if (Serial.available() > 0) {  // Check if at least 2 bytes are available
    String data = Serial.readStringUntil('\n');
    Serial.println(data);
    delay(1000);

    int comma = data.indexOf(',');
    if (comma != -1) {
      int num1 = data.substring(0, comma).toInt();
      int num2 = data.substring(comma + 1).toInt();

      if (num1 > 5 || num2 > 5) {
        Serial.println("paper__success__none");
      } else {
        Serial.println("glass__fail__pickup_failure");
      }
    }
  }
}