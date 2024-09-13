import serial.tools.list_ports

SUCTION_CUP_ARDUINO = "/dev/cu.usbserial-110"
MOTOR_ARDUINO = "/dev/cu.usbmodem1101"

ARDUINO_PORTS = [SUCTION_CUP_ARDUINO, MOTOR_ARDUINO]

def list_arduino_devices():
    arduino_ports = []
    ports = list(serial.tools.list_ports.comports())
    
    for port in ports:
        if any(keyword in port.device.lower() for keyword in ARDUINO_PORTS):
            arduino_ports.append(port)
            port.type = "Motor" if port.device == MOTOR_ARDUINO else "Suction Cup"

    return arduino_ports

if __name__ == "__main__":
    arduino_devices = list_arduino_devices()
    
    if arduino_devices:
        print("Arduino devices found:")
        for device in arduino_devices:
            print(f"Port: {device.device}")
            print(f"Description: {device.description}")
            print(f"Hardware ID: {device.hwid}")
            print(f"Type: {device.type}")
            print("---")
    else:
        print("No Arduino devices found.")

