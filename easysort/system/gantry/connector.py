from typing import Optional
import time

import serial
import serial.tools.list_ports
from easysort.common.logger import EasySortLogger
from easysort.common.environment import Environment
from easysort.utils.detections import Detection
import numpy as np

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
        self.start_time = time.time()
        self.arduino_time_offset: float = 0.0
        self._sync_time()

    def go_to(self, x: int, y: int, z: int) -> None:
        self.send_information((x, y, z), self.suction_state)

    def suction_on(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        self.send_information((x or self.position[0], y or self.position[1], z or self.position[2]), 1)

    def suction_off(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        self.send_information((x or self.position[0], y or self.position[1], z or self.position[2]), 0)

    def establish_connection(self) -> serial.Serial:
        try:
            ser = serial.Serial(self.port, 115200, timeout=1)
            if not ser.is_open: raise serial.SerialTimeoutException(f"Failed to open serial connection on port {self.port}")
            _LOGGER.info(f"Established connection to {self.name} on {self.port}")
            return ser
        except serial.SerialException as err:
            open_ports = [port.device for port in serial.tools.list_ports.comports()]
            raise serial.SerialException(f"No open port at {self.port}, instead use one of: {open_ports}") from err

    def _sync_time(self) -> None:
        """
        Get time offset between Arduino startup and connector startup
        """
        while not self.is_ready: pass
        self.ser.reset_input_buffer()
        self.ser.write(b"SYNC?\n")
        while not self.ser.in_waiting: pass
        not_ready_msg = self.ser.readline().decode().strip()
        assert not_ready_msg == "-NOT-READY-"
        while not self.ser.in_waiting: pass
        message = self.ser.readline().decode().strip()
        try:
            arduino_ms = float(message) / 1000.0  # Convert Arduino milliseconds to seconds
            self.arduino_time_offset = (time.time() - self.start_time) - arduino_ms
            _LOGGER.info(f"Time sync established. Arduino offset: {self.arduino_time_offset}")
        except (ValueError, serial.SerialException) as e:
            _LOGGER.error(f"Time sync failed: {e}")

    def pickup_detection(self, detection: Detection) -> None:
        dx, dy, dz = detection.center_point
        if detection.timestamp is None:
            _LOGGER.error(f"Detection timestamp is None for detection: {detection}")
            return
        detection_time = self.start_time - detection.timestamp + self.arduino_time_offset
        tx, ty, tz = (10, 0, 0)
        rx, ry, rz = (0, 0, 0)
        ds, ts, rs = (1, 0, 0)
        instructions = f"pickup({dx},{dy},{dz},{ds}).({tx},{ty},{tz},{ts}).({rx},{ry},{rz},{rs}).({detection_time})"
        try: self.ser.write(instructions.encode())
        except serial.SerialException as err: _LOGGER.error(f"Error sending information to {self.port}: {err}")

    def send_information(self, position: tuple, suction_state: int) -> None:
        if not self.is_ready: return
        if self.position == position and self.suction_state == suction_state: return
        self.position = position
        self.suction_state = suction_state
        msg = f"{','.join(map(str, position))},{suction_state}\n".encode()
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

    def quit(self) -> None:
        self.send_information((0, 0, 0), 0)
        while not self.is_ready: pass
        self.ser.close()

if __name__ == "__main__":
    connector = GantryConnector(Environment.GANTRY_PORT)
    time.sleep(5)
    detection = Detection(box=np.array([0, 0, 100, 100]), timestamp=time.time())
    # while not connector.is_ready: pass
    # connector.go_to(-10, -10, -10)
    # while not connector.is_ready: pass
    # connector.suction_on()
    # while not connector.is_ready: pass
    # connector.go_to(0,0,0)
    # while not connector.is_ready: pass
    # connector.suction_off()
    # while not connector.is_ready: pass
    connector.quit()
    print("Done")