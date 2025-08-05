
import mss
from PIL import Image
import numpy as np
import cv2

from easysort.common.environment import Environment

TRUCK_LOCATIONS_CROPS = [
    (0, 0.25),
    (0.25, 0.5),
    
]

class TruckRegistrer:
    def run(self):
        try:
            while True:
                image = capture_screen(2)

                if Environment.DEBUG:
                    cv2.imshow("Truck Crops - Debug", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(1)
                if key == ord('q'): break

        except KeyboardInterrupt:
            print("Exiting...")

def capture_screen(screen_idx: int = 1) -> Image.Image:
    with mss.mss() as sct:
        assert 0 <= screen_idx < len(sct.monitors), "Invalid screen index"
        monitor = sct.monitors[screen_idx]  # 0 is all monitors; 1 is primary
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)
    
def is_truck_in_image(image: Image.Image) -> bool:
    """
    Check if a truck is present in the given image.
    This function should be implemented with actual logic to detect trucks.
    """
    # Placeholder for truck detection logic
    return False
    
if __name__ == "__main__":
    TruckRegistrer().run()