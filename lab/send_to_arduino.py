import serial
import time
import random

# Establish a connection to the Arduino
# You may need to change the port name and baud rate as per your setup
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to establish

def generate_values():
    return [random.randint(100, 1000) for _ in range(3)]

while True:
    values = generate_values()
    print(f"Sending values: {values}")
    
    # Convert integers to bytes and send
    for value in values:
        ser.write(value.to_bytes(4, byteorder='big'))
    
    time.sleep(10)  # Wait for 10 seconds before sending the next set of values

ser.close()