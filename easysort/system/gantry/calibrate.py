
class Calibrator:
    """
    Works in the following way:
    1) Robot doesn't know where it is.
    2) Move robot in X+ direction until it can be seen by camera
    3) Move robot in Y+ direction until it is right over the camera
    4) Move robot in X+ direction until known position is reached
    5) Move robot in Z coordinates until known position reached
    6) Do a fast moving sequence twice and stop and the start location: If it is at the same location, we are good.
    7) If not, show error message and stop robot
    """
    def __init__(self) -> None:
        pass

    def calibrate(self) -> None:
        pass
