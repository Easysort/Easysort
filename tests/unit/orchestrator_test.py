import sys

sys.path.append("/Users/lucasvilsen/Documents/Documents/EasySort")

from src.helpers.orchestrator import Orchestrator

class OrchestratorTest:
    def test_get_database_representation(self):
        orchestrator = Orchestrator('Dave', connect_motor=False, connect_suction_cup=False)
        container = orchestrator.robot_config.paper_position
        
        assert orchestrator.get_database_representation(container, "success", "none") == "paper__success__none", \
            f"Expected paper__success__none, got {orchestrator.get_database_representation(container, 'success', 'none')}"
        assert orchestrator.database.is_valid_movement_message(orchestrator.get_database_representation(container, "success", "none"))
        assert orchestrator.get_database_representation(container, "fail", "pickup_fail") == "paper__fail__pickup_fail", \
            f"Expected paper__fail__pickup_fail, got {orchestrator.get_database_representation(container, 'fail', 'pickup_fail')}"
        assert orchestrator.database.is_valid_movement_message(orchestrator.get_database_representation(container, "fail", "pickup_fail"))

if __name__ == "__main__":
    test = OrchestratorTest()
    test.test_get_database_representation()

