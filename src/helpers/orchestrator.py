from typing import List, Optional
from dataclasses import asdict

from src.utilities.arduino import ArduinoMotorConnector, ArduinoSuctionCupConnector
from src.utilities.config import load_robot_config, load_robot_overview_config
from src.utilities.logger import EasySortLogger
from src.helpers.datasaver import DataSaver
from src.utilities.config import RobotConfig

_LOGGER = EasySortLogger()

class Orchestrator:
    def __init__(self, robot_id: str, connect_motor: bool = True, connect_suction_cup: bool = True, _overwrite_config: Optional[dict] = None):
        overview_config = load_robot_overview_config()
        if robot_id not in overview_config.names: raise ValueError(f"Invalid robot_id. Must be one of {overview_config.names}")
        self.robot_config = load_robot_config(robot_id)
        if _overwrite_config: self._overwrite_config(_overwrite_config)
        self.motor_arduino = ArduinoMotorConnector(self.robot_config) if connect_motor else None
        self.suction_cup_arduino = ArduinoSuctionCupConnector(self.robot_config) if connect_suction_cup else None
        self.database = DataSaver("db.json")
        _LOGGER.info(f"Successfully set up connection to Arduino for {robot_id}")

    def _overwrite_config(self, overwrite_dict: dict):
        self.robot_config = RobotConfig(**{**asdict(self.robot_config), **overwrite_dict})

    def pick_up_flow(self, x: float, y: float, container: List[int]):
        """
        Picks up item and puts in container

        args:
            x, y coordinates from camera to current position of item to pick up. The z coordinate is controlled by the robot.
            container is the container to put the item in
        """
        if container not in self.robot_config.sort_containers: _LOGGER.error(f"Invalid container: {container}. Must be one of {self.robot_config.sort_containers}"); return
        self.motor_arduino.navigate_to(x, y)
        self.suction_cup_arduino.on()
        arduino_response = self.motor_arduino.navigate_to(*container)
        self.suction_cup_arduino.off()
        self.database.decode_and_save(arduino_response)
        return
    
    def quit(self):
        if self.motor_arduino: self.motor_arduino.quit()
        if self.suction_cup_arduino: self.suction_cup_arduino.quit()
        if self.database: self.database.quit()
        return
