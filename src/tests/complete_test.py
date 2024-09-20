from src.helpers.orchestrator import Orchestrator
from src.utilities.environment import Environment

orchestrator = Orchestrator(robot_id="TRASHINATOR")

orchestrator.pick_up_flow(10, 10, Environment.TRASHINATOR.PAPER_POSITION)

