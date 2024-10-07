from src.utilities.config import RobotConfig
from src.utilities.math import get_conveyor_speed

import time

class Coordinate:
    def __init__(self, x: float, y: float, robot_config: RobotConfig):
        self._init_time = time.time()
        self.robot_config = robot_config
        self._init_x = coordinate2steps(x, robot_config)
        self._init_y = coordinate2steps(y, robot_config)

    def get_position(self) -> tuple:
        t = time.time() - self._init_time
        x = self._init_x + self.robot_config.conveyor_speed * t 
        return (x, self._init_y)
    
    def x(self) -> float: return self.get_position()[0]
    def y(self) -> float: return self.get_position()[1]
    
    def get_pickup_position(self) -> tuple:
        # TODO
        return (self.x, self.y)

    def coordinate2steps(self) -> float:
        # TODO
        return 
