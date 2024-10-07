from src.utilities.config import RobotConfig
from src.utilities.coordinates import Coordinate

class PickupCalculater():
    def __init__(self, robot_config: RobotConfig):
        self.robot_config = robot_config
        self.time_to_max_speed = robot_config.xy_max_speed / robot_config.xy_acceleration
        xy_steps_per_cm = robot_config.xy_steps_per_rev / (robot_config.cms_for_10_revolutions / 10)
        self.conveyor_speed = robot_config.conveyor_speed_cm_per_s / xy_steps_per_cm
        

    def calculate_pickup_position(self, coordinate: Coordinate) -> Coordinate:
        steps_coordinate = coordinate2steps(coordinate, self.robot_config)
        # TODO: Add some processing here
        return StepsCoordinate(steps_coordinate.x, steps_coordinate.y)
    
    def interpolate(self, steps_coordinate: StepsCoordinate) -> StepsCoordinate:
        return
    
def get_conveyor_speed(robot_config: RobotConfig) -> float:
    """
    Calculate the conveyor speed in steps per second
    """
    xy_steps_per_cm = robot_config.xy_steps_per_rev / (robot_config.cms_for_10_revolutions / 10)
    return robot_config.conveyor_speed_cm_per_s / xy_steps_per_cm
