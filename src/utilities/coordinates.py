from src.utilities.config import RobotConfig
from dataclasses import dataclass

@dataclass
class Coordinate: # Coordinate on screen
    x: float
    y: float

class CentimeterCoordinate: # Coordinate in centimeters
    x: float
    y: float

class StepsCoordinate: # Coordinate in steps
    x: int
    y: int

def coordinate2steps(coordinate: Coordinate, robot_config: RobotConfig) -> StepsCoordinate:
    # SOME PROCESSING TODO
    return StepsCoordinate(coordinate.x, coordinate.y)
