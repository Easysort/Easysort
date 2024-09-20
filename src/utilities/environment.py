from typing import List

class Environment:
    
    ## ARDUINO PORTS ##
    MOTOR_ARDUINO: str = "/dev/cu.usbmodem1101"
    SUCTION_CUP_ARDUINO: str = "/dev/cu.usbserial-110"

    ## TRASHINATOR LOCATIONS ##
    POSSIBLE_ROBOT_IDS: List[str] = ["TRASHINATOR"]
    # Locations can be different for our robots, so they have their own config
    class TRASHINATOR:
        SORT_CONTAINERS: List[str] = ["plastic", "metal", "metal-can", "plastic-can", "paper"] 
        SLEEP_POSITION: List[int] = [0, 0]
        READY_POSITION: List[int] = [0, 0]
        PLASTIC_POSITION: List[int] = [0, 0]
        PLASTIC_CANS_POSITION: List[int] = [0, 0]
        METAL_POSITION: List[int] = [0, 0]
        METAL_CANS_POSITION: List[int] = [0, 0]
        PAPER_POSITION: List[int] = [0, 0]

