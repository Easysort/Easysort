import serial
import time

MOTOR_ARDUINO = "/dev/cu.usbmodem1101"

def connect_to_arduino():
    try:
        ser = serial.Serial(MOTOR_ARDUINO, 9600, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        return ser
    except serial.SerialException as err:
        print(f"Failed to connect to Arduino at {MOTOR_ARDUINO}, {err}")
        return None

def send_position(ser, x, y, z):
    if ser:
        ser.write(bytes([x, y, z]))
        time.sleep(0.1)  # Give Arduino time to process and respond
        while ser.in_waiting:
            print(ser.readline().decode().strip())

def main():
    ser = connect_to_arduino()
    if not ser:
        return

    print("Motor Position Control")
    # 6 values for 3 motors could make much higher res
    print("Enter X, Y, Z coordinates (0-255 each) or 'q' to quit")

    while True:
        user_input_X = input("Enter position X: ").strip().lower()
        user_input_Y = input("Enter position Y: ").strip().lower()
        user_input_Z = input("Enter position Z: ").strip().lower()
        
        if any([user_input_X == 'q', user_input_Y == 'q', user_input_Z == 'q']):
            break
        
        try:
            x, y, z = map(int, (user_input_X, user_input_Y, user_input_Z))
            if all(0 <= val <= 255 for val in (x, y, z)):
                send_position(ser, x, y, z)
                print(f"Sent position: X={x}, Y={y}, Z={z}")
            else:
                print("Values must be between 0 and 255.")
        except ValueError:
            print("Invalid input. Please enter three integers separated by spaces.")

    ser.close()
    print("Connection closed.")

if __name__ == "__main__":
    main()
