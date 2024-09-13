import serial
import time

SUCTION_CUP_ARDUINO = "/dev/cu.usbserial-110"

def connect_to_arduino():
    try:
        ser = serial.Serial(SUCTION_CUP_ARDUINO, 9600, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        return ser
    except serial.SerialException:
        print(f"Failed to connect to Arduino at {SUCTION_CUP_ARDUINO}")
        return None

def control_pump(ser, command):
    if ser:
        ser.write(command.encode())

def main():
    ser = connect_to_arduino()
    if not ser:
        return

    print("Suction Cup Control")
    print("Commands: '1' to start pump, '0' to stop pump, 'q' to quit")

    while True:
        user_input = input("Enter command: ").strip().lower()
        
        if user_input == 'q':
            break
        elif user_input in ['0', '1']:
            control_pump(ser, user_input)
            print(f"Pump {'started' if user_input == '1' else 'stopped'}")
        else:
            print("Invalid command. Use '1' to start, '0' to stop, or 'q' to quit.")

    ser.close()
    print("Connection closed.")

if __name__ == "__main__":
    main()