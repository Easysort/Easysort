
// CNC Shield Stepper  Control Demo
// Superb Tech
// www.youtube.com/superbtech
// https://www.youtube.com/watch?v=zUb8tiFCwmk&ab_channel=SuperbTech

const int StepX = 2;
const int DirX = 5;
const int StepY = 3;
const int DirY = 6;
const int StepZ = 4;
const int DirZ = 7;


void setup() {
  pinMode(StepX,OUTPUT);
  pinMode(DirX,OUTPUT);
  pinMode(StepY,OUTPUT);
  pinMode(DirY,OUTPUT);
  pinMode(StepZ,OUTPUT);
  pinMode( DirZ,OUTPUT);

}

void loop() {
 digitalWrite(DirX, HIGH); // set direction, HIGH for clockwise, LOW for anticlockwise
 digitalWrite(DirY, HIGH);
 digitalWrite(DirZ, HIGH);
 
 for(int x = 0; x<600; x++) { // loop for 200 steps
  digitalWrite(StepX,HIGH);
  digitalWrite(StepY,HIGH);
  digitalWrite(StepZ,HIGH);
  delayMicroseconds(1000);
  digitalWrite(StepZ,LOW); 
  digitalWrite(StepX,LOW); 
  digitalWrite(StepY,LOW); 
  delayMicroseconds(1000);
 }
 delay(2000); // delay for 1 second
 digitalWrite(DirX, LOW); // set direction, HIGH for clockwise, LOW for anticlockwise
 digitalWrite(DirY, LOW);
 digitalWrite(DirZ, LOW);
 for(int x = 0; x<200; x++) { // loop for 200 steps
  digitalWrite(StepX,HIGH);
  digitalWrite(StepY,HIGH);
  digitalWrite(StepZ,HIGH);
  delayMicroseconds(3000);
  digitalWrite(StepZ,LOW); 
  digitalWrite(StepX,LOW); 
  digitalWrite(StepY,LOW); 
  delayMicroseconds(3000);
 }
 delay(2000);
}
