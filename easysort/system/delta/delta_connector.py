
import serial
import time
import asyncio
from dataclasses import dataclass

import serial.errors
from easysort.common.logger import EasySortLogger

_LOGGER = EasySortLogger()
_MAX_TIME_TO_WAIT_FOR_MOVEMENT_MESSAGE = 5 # seconds
_MAX_TIME_TO_WAIT_FOR_ECHO_CONFIRMATION = 1 # seconds
_MAX_RETRIES = 5

class DeltaConnector:
    def __init__(self, port: str, name: str = "Arduino") -> None: self.port = port; self.name = name; self.ser = self.establish_connection()
    def __call__(self, x, y) -> None: pass
    def is_ready(self) -> bool: pass

    def establish_connection(self) -> serial.Serial:
        try:
            ser = serial.Serial(self.port, 9600, timeout=1)
            if not ser.is_open: raise serial.SerialTimeoutException(f"Failed to open serial connection on port {self.port}")
            _LOGGER.info(f"Established connection to {self.name} on {self.port}")
            return ser
        except serial.SerialException as err: 
            open_ports = [port.device for port in serial.tools.list_ports.comports()]
            raise serial.SerialException(f"No open port at {self.port}, instead use one of: {open_ports}")


class ArduinoBaseConnector():
    # def __init__(self, port: str, robot_config: RobotConfig):
    #     self.port = port
        # Determine the connection name based on the port
        self.connection = next((key for key, value in vars(robot_config).items() if value == self.port), "UNKNOWN")
        # self.ser = self.establish_connection()

    # def establish_connection(self) -> serial.Serial:
    #     try:
    #         ser = serial.Serial(self.port, 9600, timeout=1)
    #         if not ser.is_open: raise ArduinoConnectionError(self.connection, self.port, "Failed to open serial connection")
    #         _LOGGER.info(f"Established connection to {self.connection} on {self.port}")
    #         return ser
    #     except serial.SerialException as err: raise ArduinoConnectionError(self.connection, self.port, err)

    async def send_information(self, msg: str | bytes | tuple, _retries: int = _MAX_RETRIES) -> None:
        if _retries == _MAX_RETRIES: self.clear_buffer()
        if _retries < 0: _LOGGER.warning(f"Max retries reached for {self.port}"); return # TODO: Handle this
        if type(msg) == str: msg = msg.encode()
        elif type(msg) == tuple: msg = f"{msg[0]},{msg[1]}\n".encode()
        try: 
            self.ser.write(msg)
            if await self.echo_confirmation_received(msg.decode().replace("\n", "")): _LOGGER.info(f"Sent information to {self.port}: {msg}, tried {abs(_retries-6)} times"); return 
            # TODO: This can make the input buffer large as the processing time is slow by arduino and multiple echos are tried. Currently fixed by self.clear_buffer(), but this is not a good solution.
            await self.send_information(msg, _retries - 1)
        except serial.SerialException as err: raise ArduinoCommunicationError(self.port, err) # TODO: add error handling, so no crashing

    async def echo_confirmation_received(self, msg: str, timeout_seconds: int = _MAX_TIME_TO_WAIT_FOR_ECHO_CONFIRMATION) -> bool:
        echo_msg = await self.receive_information(timeout_seconds) 
        return echo_msg == msg

    async def receive_information(self, timeout_seconds: int = _MAX_TIME_TO_WAIT_FOR_MOVEMENT_MESSAGE) -> str:
        # Yes, this async function allows other tasks to run while waiting.
        # The 'await asyncio.sleep(1)' releases control to the event loop,
        # allowing other coroutines to execute during the 1-second intervals.
        start_time = time.time()
        while (time.time() - start_time) < (timeout_seconds):
            if self.ser.in_waiting <= 0: await asyncio.sleep(0.1)
            msg = self.ser.readline().decode().strip()
            _LOGGER.info(f"Received message from {self.port}: {msg}")
            return msg
        _LOGGER.warning(f"Timeout occurred while waiting for response from {self.port}")
        return '' # TODO: How to handle this?
    
    def clear_buffer(self) -> None:
        self.ser.reset_input_buffer()
    
    def quit(self) -> None:
        self.ser.close(); return

class ArduinoMotorConnector(ArduinoBaseConnector):
    def __init__(self, robot_config: RobotConfig):
        super().__init__(robot_config.motor_arduino, robot_config)

    async def navigate_to(self, x: float, y: float) -> None:
        """
        Sends the coordinates to the Arduino to move the motor and then asynchronously waits for a response.
        """
        await self.send_information((x, y))
        return await self.receive_information()