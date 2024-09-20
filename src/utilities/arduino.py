import serial
import time

from src.utilities.errors import ArduinoConnectionError, ArduinoCommunicationError
from src.utilities.environment import Environment
from src.utilities.logger import EasySortLogger

_LOGGER = EasySortLogger()

class ArduinoBaseConnector():
    """
    Handles connection and communication with Arduinos.

    note: 
        A response of 0 from arduino means success
        A response of 1 from arduino means failure

    args: 
        port (str): A string representing the port to connect to. See available ports in utilities/environment.py
    """
    def __init__(self, port: str):
        self.port = port
        # Determine the connection name based on the port
        self.connection = next((name for name, value in vars(Environment).items() if value == port), "UNKNOWN")
        self.ser = self.establish_connection()

    def establish_connection(self) -> serial.Serial:
        try:
            ser = serial.Serial(self.port, 9600, timeout=1)
            if not ser.is_open: raise ArduinoConnectionError(self.connection, self.port, "Failed to open serial connection")
            return ser
        except serial.SerialException as err: raise ArduinoConnectionError(self.connection, self.port, err)

    def send_information(self, msg: str | bytes):
        if type(msg) == str: msg = msg.encode()
        try: self.ser.write(msg); _LOGGER.info(f"Sent information to {self.port}: {msg}")
        except serial.SerialException as err: raise ArduinoCommunicationError(self.port, err)

    def receive_information(self, wait: bool = False, timeout_minutes: int = 1) -> str:
        if wait: 
            start_time = time.time()
            while (time.time() - start_time) < (timeout_minutes * 60):
                response = self.ser.readline().decode().strip()
                if response: return response
        return self.ser.readline().decode().strip()

class ArduinoMotorConnector(ArduinoBaseConnector):
    def __init__(self):
        super().__init__(Environment.MOTOR_ARDUINO)

    def navigate(self, x: float, y: float, wait: bool = True):
        """
        Sends the coordinates to the Arduino to move the motor and wait for response.
        """
        self.send_information(bytes([x, y]))
        return self.receive_information(wait=wait)

class ArduinoSuctionCupConnector(ArduinoBaseConnector):
    def __init__(self):
        super().__init__(Environment.SUCTION_CUP_ARDUINO)
    
    def on(self): self.send_information(b'1')
    def off(self): self.send_information(b'0')

