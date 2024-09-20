
# Arduino
from src.utilities.environment import Environment

class ArduinoConnectionError(Exception):
    def __init__(self, name: str, port: str, error: str):
        super().__init__(f"Failed to connect to {name} at {port}, {error}")

class ArduinoCommunicationError(Exception):
    def __init__(self, name: str, port: str, error: str):
        super().__init__(f"Failed to communicate with {name} at {port}, {error}")

# Robot

class InvalidRobotIdError(Exception):
    def __init__(self, robot_id: str):
        super().__init__(f"Invalid robot_id: {robot_id}. Must be one of {Environment.POSSIBLE_ROBOT_IDS}")

class InvalidContainerError(Exception):
    def __init__(self, container: str, robot_config: dict):
        super().__init__(f"Invalid container: {container}. This robot cannot sort this type of trash. Must be one of {robot_config.SORT_CONTAINERS}")

