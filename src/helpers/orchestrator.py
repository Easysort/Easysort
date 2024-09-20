from typing import List

from src.utilities.arduino import ArduinoMotorConnector, ArduinoSuctionCupConnector
from src.utilities.environment import Environment
from src.utilities.logger import EasySortLogger

_LOGGER = EasySortLogger()

class Orchestrator:
    def __init__(self, robot_id: str):
        if robot_id not in Environment.POSSIBLE_ROBOT_IDS: raise ValueError(f"Invalid robot_id. Must be one of {Environment.POSSIBLE_ROBOT_IDS}")
        self.robot_config = getattr(Environment, robot_id)
        self.motor_arduino = ArduinoMotorConnector()
        self.suction_cup_arduino = ArduinoSuctionCupConnector()

    def pick_up_flow(self, x: float, y: float, container: List[int]):
        """
        Picks up item and puts in container

        args:
            x, y, z is coordinates from camera to current position of item to pick up
            container is the container to put the item in
        """
        if container not in self.robot_config.SORT_CONTAINERS: _LOGGER.error(f"Invalid container: {container}. Must be one of {self.robot_config.SORT_CONTAINERS}"); return
        self.motor_arduino.move_to(x, y)
        self.suction_cup_arduino.on()
        self.motor_arduino.move_to(*container)
        self.suction_cup_arduino.off()
        return
        

