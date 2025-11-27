
# First check for people
# Then check for 80% in bboxes in crop and width over 50px
# Then check for facing direction
# Then heck for duplicate people
# Then send to chatgpt for information

from easysort.gpt_trainer import YoloTrainer
from ultralytics.engine.results import Results
from easysort.sampler import Crop, Sampler
from easysort.registry import Registry
from easysort.helpers import Sort

import numpy as np
import cv2
from typing import List, Dict
import os
import datetime
from tqdm import tqdm
import shutil

CROP_ROSKILDE = Crop(x=631, y=110, w=210, h=540) # Roskilde
CROP_JYLLINGE = Crop(x=640, y=0, w=260, h=480) # Jyllinge
MIN_WIDTH = 80
MIN_HEIGHT = 200
PERCENTAGE_IN_CROP = 0.5

BATCH_SIZE = 16

def classify_facing_direction(person_kpts_xy, person_kpts_conf, face_conf_thresh=0.4, dx_forward_thresh=10):
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6
    def visible(idx, t=face_conf_thresh): return person_kpts_conf[idx] > t
    if not (visible(L_SHOULDER, 0.2) and visible(R_SHOULDER, 0.2)): return "unknown"
    sx = (person_kpts_xy[L_SHOULDER, 0] + person_kpts_xy[R_SHOULDER, 0]) / 2.0
    sy = (person_kpts_xy[L_SHOULDER, 1] + person_kpts_xy[R_SHOULDER, 1]) / 2.0
    has_nose = visible(NOSE)
    has_eyes = visible(L_EYE) or visible(R_EYE)
    has_ears = visible(L_EAR) or visible(R_EAR)
    if not (has_nose or has_eyes or has_ears): return "back"
    if not has_nose:
        if visible(L_EAR) and not visible(R_EAR): return "right"
        if visible(R_EAR) and not visible(L_EAR): return "left"
        return "unknown"
    nx, ny = person_kpts_xy[NOSE]
    dx = nx - sx
    if abs(dx) <= dx_forward_thresh: return "forward"
    else:
        if dx > 0: return "right"
        else: return "left"

def box1_in_box2(A, B):
    # How much of A is inside B
    Ax1, Ay1, Ax2, Ay2 = A
    Bx1, By1, Bx2, By2 = B

    # Intersection coords
    Ix1 = max(Ax1, Bx1)
    Iy1 = max(Ay1, By1)
    Ix2 = min(Ax2, Bx2)
    Iy2 = min(Ay2, By2)

    Iw = max(0.0, Ix2 - Ix1)
    Ih = max(0.0, Iy2 - Iy1)
    A_inter = Iw * Ih

    A_A = max(0.0, Ax2 - Ax1) * max(0.0, Ay2 - Ay1)
    if A_A == 0: return 0.0

    return A_inter / A_A

def check_side_view_visibility(person_kpts_conf, conf_thresh=0.4):
    """Check if person is visible from side: needs head, shoulder, and knee/foot"""
    L_SHOULDER, R_SHOULDER = 5, 6
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    
    def visible(idx, t=conf_thresh):  return person_kpts_conf[idx] > t
    has_shoulder = visible(L_SHOULDER) or visible(R_SHOULDER)
    has_lower_body = visible(L_KNEE) or visible(R_KNEE) or visible(L_ANKLE) or visible(R_ANKLE)
    return has_shoulder and has_lower_body

class ArgoTentCheck:
    def __init__(self, output_dir: str):
        self.yolo_model = YoloTrainer("yolo11n.pt")
        self.pose_model = YoloTrainer("yolo11n-pose.pt")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._local_information = {}

    def check_people(self, image: np.ndarray) -> Results|None:
        results = self.yolo_model.model(image, classes=[0], verbose=False)
        return results if len(results) > 0 else None
    
    def bboxes_idx_to_poses_idx(self, box_results: Results, pose_results: Results) -> Dict[int, Dict[int, int]]:
        # matches[i] = {j: k} # i is box result index = k is pose result index, j is box index and k is pose index.
        matches = {}
        for i, result in enumerate(box_results):
            matches[i] = {}
            for j, box in enumerate(result.boxes):
                scores = {}
                box_x1, box_y1, box_x2, box_y2 = box.xyxy[0].int().tolist()
                pose_result = pose_results[i]
                if pose_result.keypoints is not None:
                    for l, person_xy in enumerate(pose_result.keypoints.xy.cpu().numpy()):
                        if l in scores: continue
                        scores[l] = 0
                        for x, y in person_xy:
                            if x > box_x1 and x < box_x2 and y > box_y1 and y < box_y2: scores[l] += 1
                        max_score_idx = max(scores.keys(), key=lambda x: scores[x])
                        matches[i][j] = max_score_idx if scores[max_score_idx] > 4 else None

        return matches

    def check_bboxes(self, results: Results, crop: Crop) -> Results|None:
        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes):
                box_x1, box_y1, box_x2, box_y2 = box.xyxy[0].int().tolist()
                keep = True
                if box_x2 - box_x1 < MIN_WIDTH or box_y2 - box_y1 < MIN_HEIGHT: keep = False
                if box1_in_box2(box.xyxy[0].int().tolist(), [crop.x, crop.y, crop.x + crop.w, crop.y + crop.h]) < PERCENTAGE_IN_CROP: keep = False
                self._local_information[f"{i}_{j}_keep"] = keep
                self._local_information[f"{i}_{j}_keep_params"] = {"width": box_x2 - box_x1, "height": box_y2 - box_y1, "percentage_in_crop": box1_in_box2(box.xyxy[0].int().tolist(), [crop.x, crop.y, crop.x + crop.w, crop.y + crop.h])}

    def check_facing_direction(self, image: np.ndarray, results: Results) -> Results|None:
        results_pose = self.pose_model.model(image, verbose=False)
        matches = self.bboxes_idx_to_poses_idx(results, results_pose)
        for i in range(len(results)):
            if results_pose[i].keypoints is None: continue
            kpts_xy = results_pose[i].keypoints.xy.cpu().numpy()
            kpts_conf = results_pose[i].keypoints.conf.cpu().numpy()

            for j, pose_idx in matches[i].items():
                if pose_idx is None: continue
                person_xy = kpts_xy[pose_idx]
                person_conf = kpts_conf[pose_idx]
                assert person_xy.shape == (17, 2)
                direction = classify_facing_direction(person_xy, person_conf)
                visible = check_side_view_visibility(person_conf)
                if not visible: self._local_information[f"{i}_{j}_keep"] = False
                self._local_information[f"{i}_{j}_visible"] = str(visible)
                self._local_information[f"{i}_{j}_direction"] = direction

        for i, result in enumerate(results):
            for j in range(len(result.boxes)):
                if f"{i}_{j}_direction" not in self._local_information:
                    self._local_information[f"{i}_{j}_direction"] = "none"
                if f"{i}_{j}_visible" not in self._local_information:
                    self._local_information[f"{i}_{j}_visible"] = "unknown"
        
        return results_pose, matches

    def check_duplicate_people(self, results: Results) -> Results|None:
        pass

    def send_to_chatgpt(self, image: np.ndarray, results: Results) -> Results|None:
        pass

    def check_image(self, images: List[np.ndarray], crop: Crop, save_paths: List[str]) -> List[np.ndarray]|None:
        # if image.shape[0] > 600: return None
        results = self.check_people(images)
        assert len(results) == len(images)
        self.check_bboxes(results, crop)
        results_pose, matches = self.check_facing_direction(images, results)
        self.check_duplicate_people(results)
        self.send_to_chatgpt(images, results)

        # cv2.rectangle(image, (crop.x, crop.y), (crop.x + crop.w, crop.y + crop.h), (0, 255, 255), 2)

        # if len(results[0].boxes) == 0: return None
        # for i, result in enumerate(results):
        #     if len(result.boxes) == 0: continue
        #     for j, box in enumerate(result.boxes):
        #         color = (0, 255, 0) if self._local_information.get(f"{i}_{j}_keep", False) else (0, 0, 255)
        #         cv2.rectangle(image, (box.xyxy[0].int().tolist()[0], box.xyxy[0].int().tolist()[1]), (box.xyxy[0].int().tolist()[2], box.xyxy[0].int().tolist()[3]), color, 2)
        #         cv2.putText(image, f"{i}_{j}: {self._local_information[f"{i}_{j}_direction"]}", (box.xyxy[0].int().tolist()[0], box.xyxy[0].int().tolist()[3] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        #         cv2.putText(image, str(self._local_information[f"{i}_{j}_keep_params"]), (box.xyxy[0].int().tolist()[0], box.xyxy[0].int().tolist()[3] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        #         if matches[i][j] is None: continue
        #         cv2.circle(image, (int(results_pose[i].keypoints.xy.cpu().numpy()[matches[i][j]][0][0]), int(results_pose[i].keypoints.xy.cpu().numpy()[matches[i][j]][0][1])), 8, (255, 0, 0), 2)
        #         cv2.putText(image, f"{i}_{j}", (int(results_pose[i].keypoints.xy.cpu().numpy()[matches[i][j]][0][0]), int(results_pose[i].keypoints.xy.cpu().numpy()[matches[i][j]][0][1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        
        for i, result in enumerate(results):
            if any(self._local_information.get(f"{i}_{j}_keep", False) for j in range(len(result.boxes))):
                cv2.imwrite(save_paths[i], images[i])
                print(f"Saved {save_paths[i]}")
                # Save all person crops with idx and direction in name
                for j in range(len(result.boxes)):
                    if self._local_information.get(f"{i}_{j}_keep", False):
                        person_crop = images[i][result.boxes[j].xyxy[0].int().tolist()[1]:result.boxes[j].xyxy[0].int().tolist()[3], result.boxes[j].xyxy[0].int().tolist()[0]:result.boxes[j].xyxy[0].int().tolist()[2]]
                        cv2.imwrite(f"{save_paths[i].replace('.jpg', '')}_{i}_{j}_{self._local_information[f'{i}_{j}_direction']}.jpg", person_crop)

        return 
    
    # def check_images(self, images: List[np.ndarray], save_path: str):
    #     for image_path in images:
    #         assert os.path.exists(image_path), "path does not exist: " + image_path
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         image = self.check_image(image, CROP_JYLLINGE)
    #         if image is None: continue

    #         cv2.imshow("image", image)
    #         key = cv2.waitKey(0)
    #         if key == ord('q'): break
    #     cv2.destroyAllWindows()

    def check_video(self, video_path: str):
        images = Sampler.unpack(video_path)
        print(f"Checking {len(images)} images from {video_path}")
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE]
            save_paths = [os.path.join(self.output_dir, f"{video_path.replace('/', '-').replace('.mp4', '')}_{i+j}.jpg") for j in range(len(batch))]
            crop = CROP_JYLLINGE if "Jyllinge" in video_path else CROP_ROSKILDE
            self.check_image(batch, crop, save_paths)

    def run(self, video_paths: List[str]):
        for video_path in tqdm(video_paths):
            self.check_video(video_path)

if __name__ == "__main__":
    files = Registry.LIST("argo")
    files = list(Sort.since(files, datetime.datetime(2025, 11, 17)))
    files = list(Sort.before(files, datetime.datetime(2025, 11, 24)))
    # os.makedirs("tmp2", exist_ok=True)
    # for file in files[:20]:
    #     shutil.copy(Registry._registry_path(file), "tmp2/" + file.replace("/", "-"))
    
    # print(f"Checking {len(files)} files, like: {files[0]}")
    checker = ArgoTentCheck(output_dir="output")
    checker.run(files)





