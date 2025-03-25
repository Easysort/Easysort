from typing import Optional
import time

import serial
import serial.tools.list_ports
from easysort.common.logger import EasySortLogger
from easysort.common.environment import Environment

_LOGGER = EasySortLogger()
_MAX_TIME_TO_WAIT_FOR_MOVEMENT_MESSAGE = 5 # seconds

class GantryConnector:
    """
    TODO:
    - Test with Jetson + AI
    - Add calibration calls to controller via connector
    - Add is_ready()
    """

    def __init__(self, port: str, name: str = "Arduino") -> None:
        self.port = port
        self.name = name
        self.ser = self.establish_connection()
        self.position = (0, 0, 0)
        self.suction_state = 0
        self._is_ready = False

    def go_to(self, x: int, y: int, z: int) -> None:
        self.position = (x, y, z)
        self.send_information((x, y, z, self.suction_state))

    def suction_on(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        self.position = (x or self.position[0], y or self.position[1], z or self.position[2])
        self.suction_state = 1
        self.send_information((x, y, z, self.suction_state))

    def suction_off(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        self.position = (x or self.position[0], y or self.position[1], z or self.position[2])
        self.suction_state = 0
        self.send_information((x, y, z, self.suction_state))

    def establish_connection(self) -> serial.Serial:
        try:
            ser = serial.Serial(self.port, 9600, timeout=1)
            if not ser.is_open: raise serial.SerialTimeoutException(f"Failed to open serial connection on port {self.port}")
            _LOGGER.info(f"Established connection to {self.name} on {self.port}")
            return ser
        except serial.SerialException as err:
            open_ports = [port.device for port in serial.tools.list_ports.comports()]
            raise serial.SerialException(f"No open port at {self.port}, instead use one of: {open_ports}") from err

    def send_information(self, msg: str | bytes | tuple) -> None:
        if not self.is_ready: return
        if isinstance(msg, str): msg = msg.encode()
        elif isinstance(msg, tuple): msg = f"{','.join(map(str, msg))}\n".encode()
        try: self.ser.write(msg)
        except serial.SerialException as err: _LOGGER.error(f"Error sending information to {self.port}: {err}")

    def receive_information(self, timeout_seconds: int = _MAX_TIME_TO_WAIT_FOR_MOVEMENT_MESSAGE) -> str:
        start_time = time.time()
        while (time.time() - start_time) < (timeout_seconds):
            if self.ser.in_waiting <= 0: time.sleep(0.1)
            msg = self.ser.readline().decode().strip()
            _LOGGER.info(f"Received message from {self.port}: {msg}")
            return msg
        _LOGGER.warning(f"Timeout occurred while waiting for response from {self.port}")
        return '' # TODO: How to handle this?

    @property
    def is_ready(self) -> bool:
        lines = self.ser.readlines()
        not_ready_lines = max([i for i, line in enumerate(lines) if line.decode().strip() == "-NOT-READY-"] or [-1])
        ready_lines = max([i for i, line in enumerate(lines) if line.decode().strip() == "-READY-"] or [-1])
        if not_ready_lines == -1 and ready_lines == -1: return self._is_ready
        self._is_ready = not_ready_lines <= ready_lines
        return self._is_ready

    def clear_buffer(self) -> None: self.ser.reset_input_buffer()
    def quit(self) -> None: self.ser.close()

if __name__ == "__main__":
    connector = GantryConnector(Environment.GANTRY_PORT)
    while True:
        press_enter = input("Press enter to continue")
        if press_enter == "":
            if connector.suction_state: connector.suction_off(2, 2, 0)
            else: connector.suction_on(0, 0, 0)
        elif press_enter == "q":
            connector.quit()
            break
