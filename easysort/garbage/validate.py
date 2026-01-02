from ultralytics import YOLO
from pathlib import Path
import cv2

import numpy as np

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6

def classify_facing_direction(person_kpts_xy, person_kpts_conf,
                              face_conf_thresh=0.4,
                              dx_forward_thresh=10):
    """
    person_kpts_xy: np.array shape (17, 2) -> x, y
    person_kpts_conf: np.array shape (17,) -> confidence per keypoint
    Returns one of: "forward", "left", "right", "back", or "unknown"
    """

    def visible(idx, t=face_conf_thresh):
        return person_kpts_conf[idx] > t

    # Basic checks: shoulders
    if not (visible(L_SHOULDER, 0.2) and visible(R_SHOULDER, 0.2)):
        return "unknown"

    # Shoulder midpoint
    sx = (person_kpts_xy[L_SHOULDER, 0] + person_kpts_xy[R_SHOULDER, 0]) / 2.0
    sy = (person_kpts_xy[L_SHOULDER, 1] + person_kpts_xy[R_SHOULDER, 1]) / 2.0

    has_nose = visible(NOSE)
    has_eyes = visible(L_EYE) or visible(R_EYE)
    has_ears = visible(L_EAR) or visible(R_EAR)

    # If no face keypoints but torso visible -> probably back
    if not (has_nose or has_eyes or has_ears):
        return "back"

    if not has_nose:
        # No nose but some face parts: treat as side/back, fall back to ears heuristic
        # You can choose "unknown" or use ears:
        if visible(L_EAR) and not visible(R_EAR):
            return "right"   # left side of face visible → turned right
        if visible(R_EAR) and not visible(L_EAR):
            return "left"    # right side of face visible → turned left
        return "unknown"

    # Nose visible: compute dx from shoulder center
    nx, ny = person_kpts_xy[NOSE]
    dx = nx - sx

    # Optional refinement using ears
    left_ear_vis  = visible(L_EAR)
    right_ear_vis = visible(R_EAR)

    # Front vs side based on how far nose is from torso center
    if abs(dx) <= dx_forward_thresh:
        # almost centered → forward
        return "forward"
    else:
        # side: sign of dx decides left/right in image coords
        if dx > 0:
            # nose to the right of torso center -> facing right side of image
            # if one ear strongly visible, we could check consistency, but simple is fine
            return "right"
        else:
            return "left"


class Validator:
    def __init__(self, folder: str):
        self.yolo_model = YOLO("yolo11n.pt")
        self.pose_model = YOLO("yolo11n-pose.pt")
        self.image_paths = self._unpack_folder(folder)

    def _unpack_folder(self, folder: str):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]
        images = ["tmp/114221.mp4_79.jpg"]
        return images

    def _detect(self, image_path: Path):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        if image is None: return None
        results_yolo = self.yolo_model(image, classes=[0], verbose=False)
        results_pose = self.pose_model(image, verbose=False)
        print("Number of people detected: ", len(results_yolo))
        print("Number of people detected by pose model: ", len(results_pose))
        for result in results_yolo:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for result in results_pose:
            if result.keypoints is None:
                continue

            kpts_xy = result.keypoints.xy.cpu().numpy()      # [num_people, 17, 2]
            kpts_conf = result.keypoints.conf.cpu().numpy()  # [num_people, 17]

            for person_xy, person_conf in zip(kpts_xy, kpts_conf):
                direction = classify_facing_direction(person_xy, person_conf)

                # Draw label near nose or shoulders
                nose_x, nose_y = person_xy[NOSE]
                cv2.putText(image, direction, (int(nose_x), int(nose_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        for image_path in self.image_paths:
            self._detect(image_path)
            break


if __name__ == "__main__":
    validator = Validator("tmp")
    validator.run()
