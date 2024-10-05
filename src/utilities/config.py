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


def load_config(path: str, config_type: Type[T]) -> T: return config_type(**yaml.safe_load(open(path, 'r')))
def load_robot_config(robot_name: str) -> RobotConfig: return load_config(Path(ROBOT_CONFIG_PATH) / (robot_name + ".yaml"), RobotConfig)
def load_robot_overview_config() -> RobotOverviewConfig: return load_config(Path(ROBOT_OVERVIEW_PATH), RobotOverviewConfig)
