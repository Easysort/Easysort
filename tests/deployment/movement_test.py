import sys

sys.path.append("/Users/lucasvilsen/Documents/Documents/EasySort")

from src.helpers.orchestrator import Orchestrator

orchestrator = Orchestrator('Dave', connect_suction_cup=False)
orchestrator.pick_up_flow(10, 10, orchestrator.robot_config.paper_position)

