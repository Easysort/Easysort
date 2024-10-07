from typing import List
import yaml
from dataclasses import dataclass
from typing import Type, TypeVar
from pathlib import Path

# from src.utilities.environment import Environment

T = TypeVar('T')
ROBOT_OVERVIEW_PATH = 'src/configs/robot_overview.yaml'
ROBOT_CONFIG_PATH = 'src/configs'

@dataclass
class RobotOverviewConfig:
    names: List[str]

@dataclass
class RobotConfig:
    name: str
    motor_arduino: str
    suction_cup_arduino: str
    sort_containers: List[str]
    sleep_position: List[int]
    ready_position: List[int]
    plastic_position: List[int]
    plastic_cans_position: List[int]
    metal_position: List[int]
    metal_cans_position: List[int]
    paper_position: List[int]
    max_x: int
    max_y: int
    xy_max_speed: int
    xy_acceleration: int
    xy_steps_per_rev: int
    z_max_speed: int
    z_acceleration: int
    z_steps_per_rev: int
    conveyor_speed_cm_per_s: int
    cms_for_10_revolutions: int


def load_config(path: str, config_type: Type[T]) -> T: return config_type(**yaml.safe_load(open(path, 'r')))
def load_robot_config(robot_name: str) -> RobotConfig: return load_config(Path(ROBOT_CONFIG_PATH) / (robot_name + ".yaml"), RobotConfig)
def load_robot_overview_config() -> RobotOverviewConfig: return load_config(Path(ROBOT_OVERVIEW_PATH), RobotOverviewConfig)
def get_matching_config_string(robot_config: RobotConfig, value: str | int | List[int]) -> str: return next((key for key, config in robot_config.items() if config == value), None)
