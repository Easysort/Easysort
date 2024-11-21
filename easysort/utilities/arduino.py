import serial
import time
import asyncio
from dataclasses import dataclass

from easysort.utilities.errors import ArduinoConnectionError, ArduinoCommunicationError
from easysort.utilities.config import RobotConfig
from easysort.utilities.logger import EasySortLogger

_LOGGER = EasySortLogger()
_MAX_TIME_TO_WAIT_FOR_MOVEMENT_MESSAGE = 5 # seconds
_MAX_TIME_TO_WAIT_FOR_ECHO_CONFIRMATION = 1 # seconds
_MAX_RETRIES = 5


class ArduinoBaseConnector():
    """
    Handles connection and communication with Arduinos.

    note: 
        Arduino responses are kept in ArduinoResponse enum
    args: 
        port (str): A string representing the port to connect to. See available ports in utilities/environment.py

    TODO: add error handling, so no crashing
    """
    def __init__(self, port: str, robot_config: RobotConfig):
        self.port = port
        # Determine the connection name based on the port
        self.connection = next((key for key, value in vars(robot_config).items() if value == self.port), "UNKNOWN")
        self.ser = self.establish_connection()

    def establish_connection(self) -> serial.Serial:
        try:
            ser = serial.Serial(self.port, 9600, timeout=1)
            if not ser.is_open: raise ArduinoConnectionError(self.connection, self.port, "Failed to open serial connection")
            _LOGGER.info(f"Established connection to {self.connection} on {self.port}")
            return ser
        except serial.SerialException as err: raise ArduinoConnectionError(self.connection, self.port, err)

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

class ArduinoSuctionCupConnector(ArduinoBaseConnector):
    def __init__(self, robot_config: RobotConfig):
        super().__init__(robot_config.suction_cup_arduino, robot_config)
    
    # Suction Arduino now connected to motor arduino, therefore no need for on/off, 8/10/2024
    # async def on(self) -> None: await self.send_information(b'1')
    # async def off(self) -> None: await self.send_information(b'0')

