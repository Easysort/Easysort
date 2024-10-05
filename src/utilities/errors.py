from serial.tools import list_ports

from src.utilities.config import load_robot_overview_config, RobotConfig

# Arduino

class ArduinoConnectionError(Exception):
    def __init__(self, name: str, port: str, error: str):
        super().__init__(f"Failed to connect to {name} at {port}, possible ports are: {[port.device for port in list_ports.comports()]}")

class ArduinoCommunicationError(Exception):
    def __init__(self, name: str, port: str, error: str):
        super().__init__(f"Failed to communicate with {name} at {port}, {error}")

# Robot

class InvalidRobotIdError(Exception):
    def __init__(self, robot_id: str):
        super().__init__(f"Invalid robot_id: {robot_id}. Must be one of {load_robot_overview_config().names}")

class InvalidContainerError(Exception):
    def __init__(self, container: str, robot_config: RobotConfig):
        super().__init__(f"Invalid container: {container}. This robot cannot sort this type of trash. Must be one of {robot_config.SORT_CONTAINERS}")

