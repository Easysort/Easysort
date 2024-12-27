# Delta Sorting Project
This is for our Delta Robot for rough sorting of certain waste fractions.

This connector outputs 3d positions for the robot to follow. The following positions are sent to the robot:

1. The 3d position of the object at time t0
2. The 3d position of the position where the object should be left after being picked up.

These position are specified in the config and used by the classifier.
The robot should then send a status update (see `easysort/common/datasaver.py`). When the status update has been recieved, the flow repeats.