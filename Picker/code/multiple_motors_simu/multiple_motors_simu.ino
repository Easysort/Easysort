
// CNC Shield Stepper  Control Demo
// Superb Tech
// www.youtube.com/superbtech
// https://www.youtube.com/watch?v=zUb8tiFCwmk&ab_channel=SuperbTech

// Demo of 2d robot movement both fast and slow, changing directions and with breaks

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

 // Right +, left -
 // Up +, down -
 
 for(int x = 0; x<400; x++) { // Fast right
  //digitalWrite(StepZ,HIGH);
  digitalWrite(StepY,HIGH);
  delayMicroseconds(1000);
  // digitalWrite(StepZ,LOW);
  digitalWrite(StepY,LOW); 
  delayMicroseconds(1000);
 }
for(int x = 0; x<200; x++) { // Fast right, up
  digitalWrite(StepX,HIGH);
  digitalWrite(StepY,HIGH);
  delayMicroseconds(1000);
  digitalWrite(StepX,LOW); 
  digitalWrite(StepY,LOW); 
  delayMicroseconds(1000);
 }
for(int x = 0; x<400; x++) { // Fast up
  digitalWrite(StepX,HIGH);
  delayMicroseconds(1000);
  digitalWrite(StepX,LOW); 
  delayMicroseconds(1000);
 }
 delay(2000); // delay for 2 second
 digitalWrite(DirX, LOW); // set direction, HIGH for clockwise, LOW for anticlockwise
 digitalWrite(DirY, LOW);
 digitalWrite(DirZ, LOW);
 for(int x = 0; x<400; x++) { // Slow left
  digitalWrite(StepY,HIGH);
  delayMicroseconds(3000);
  digitalWrite(StepY,LOW); 
  delayMicroseconds(3000);
 }
  for(int x = 0; x<200; x++) { // Slow down
  digitalWrite(StepX,HIGH);
  delayMicroseconds(3000);
  digitalWrite(StepX,LOW);  
  delayMicroseconds(3000);
 }
 digitalWrite(DirX, HIGH);
  for(int x = 0; x<200; x++) { // Slow left up
  digitalWrite(StepX,HIGH);
  digitalWrite(StepY,HIGH);
  delayMicroseconds(3000);
  digitalWrite(StepX,LOW); 
  digitalWrite(StepY,LOW); 
  delayMicroseconds(3000);
 }
 digitalWrite(DirX, LOW);
  for(int x = 0; x<600; x++) { // Slow down
  digitalWrite(StepX,HIGH);
  delayMicroseconds(3000);
  digitalWrite(StepX,LOW); 
  delayMicroseconds(3000);
 }
 delay(2000);
}
