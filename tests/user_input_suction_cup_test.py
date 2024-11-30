import serial
import time

from src.utilities.errors import ArduinoConnectionError, ArduinoCommunicationError
from src.utilities.logger import EasySortLogger

# SUCTION_CUP_ARDUINO = "/dev/cu.usbserial-140"
SUCTION_CUP_ARDUINO = "/dev/cu.usbmodem1401"

_LOGGER = EasySortLogger()

def establish_connection(port, connection) -> serial.Serial:
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        if not ser.is_open: raise ArduinoConnectionError(connection, port, "Failed to open serial connection")
        _LOGGER.info(f"Established connection to {connection} on {port}")
        return ser
    except serial.SerialException as err: raise ArduinoConnectionError(connection, port, err)

def send_information(port: str, ser: serial.Serial, msg: str | bytes | tuple) -> None:
    if type(msg) == str: msg = msg.encode()
    elif type(msg) == tuple: msg = f"{msg[0]},{msg[1]}\n".encode()
    # _LOGGER.info(f"Sending information to {port}: {msg}")
    try: ser.write(msg); _LOGGER.info(f"Sent information to {port}: {msg}")
    except serial.SerialException as err: raise ArduinoCommunicationError(port, err) # TODO: add error handling, so no crashing

def control_pump(ser, command):
    command = command.encode()
    _LOGGER.info(f"Sending command to {SUCTION_CUP_ARDUINO}: {command}")
    if ser:
        ser.write(command)

def main():
    index = 0
    ser = establish_connection(SUCTION_CUP_ARDUINO, "Suction Cup")
    if not ser:
        return

    print("Suction Cup Control")
    print("Commands: '1' to start pump, '0' to stop pump, 'q' to quit")

    while True:
        user_input = input("Enter command: ").strip().lower()
        print(f"In waiting: {ser.in_waiting}")
        
        if user_input == 'q':
            break
        elif user_input in ['0', '1']:
            send_information(SUCTION_CUP_ARDUINO, ser, (index, index))
            # control_pump(ser, '10,10\n')
            # print(f"Pump {'started' if user_input == '1' else 'stopped'}")
            # time.sleep(2)
            print(f"Arduino said: {ser.readline().decode()}")
            time.sleep(0.5)
            print(f"Arduino said: {ser.readline().decode()}")
            time.sleep(0.5)
            print(f"Arduino said: {ser.readline().decode()}")
        else:
            print(f"Arduino said: {ser.readline().decode()}")
            # print("Invalid command. Use '1' to start, '0' to stop, or 'q' to quit.")
        index += 1

    ser.close()
    print("Connection closed.")

if __name__ == "__main__":
    main()