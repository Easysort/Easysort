

# Open a opencv window, let the user run through the frames one by one. Space gives 0, 1 gives 1, 2 gives 2, etc. q saves and quits
# Save the results to a json file at specific path using ResultRegistry.POST

import cv2
from easysort.registry import ResultRegistry

class HumanEvaluator:
    def __init__(self, paths: list[str], project: str):
        self.paths = paths
        self.project = project
        self.model = "HumanEval"
        