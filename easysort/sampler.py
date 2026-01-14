
# Should take in a video, sample x frames with y seconds between them, and z seconds between groups
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from easysort.helpers import REGISTRY_PATH
from easysort.registry import Registry

@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int

class Polygon:
    points: list[tuple[int, int]]

ROSKILDE_CROP = Crop(x=631, y=110, w=210, h=540)
JYLLINGE_CROP = Crop(x=640, y=0, w=260, h=480)
DEVICE_TO_CROP = {"Argo-Jyllinge-Entrance-01": JYLLINGE_CROP, "Argo-roskilde-03-01": ROSKILDE_CROP}

class Sampler:
    @staticmethod
    def unpack(video_path: Path|str, crop: Crop|str = None) -> list[np.ndarray]:
        if isinstance(video_path, str): video_path = Path(video_path)
        if video_path.suffix == ".jpg": return [cv2.imread(video_path) if crop is None else cv2.imread(video_path)[crop.y:crop.y+crop.h, crop.x:crop.x+crop.w]]
        if crop == "auto": crop = DEVICE_TO_CROP[video_path.parts[-6]]
        cap = cv2.VideoCapture(str(Registry._registry_path(video_path)))
        if not cap.isOpened(): raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or not fps: raise RuntimeError("Could not determine FPS for video.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)  # (H, W, 3)
        cap.release()
        if len(frames) != total_frames: print(f"Warning: Expected {total_frames} frames, got {len(frames)}")
        if crop is not None: frames = [frame[crop.y:crop.y+crop.h, crop.x:crop.x+crop.w] for frame in frames]
        return frames

def create_crop(image: np.ndarray) -> Crop|Polygon|None:
    """Interactive crop: 1=square, 2=polygon, q=quit, r=reset"""
    mode, sq_1, sq_2, polygon_points, mouse_pos = None, None, None, [], (0, 0)
    completed_polygons = []
    CLOSE_THRESHOLD = 20
    
    def get_center(points):
        x_coords, y_coords = zip(*points)
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mode, sq_1, sq_2, polygon_points, completed_polygons, mouse_pos
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == 1: 
                if sq_1 and not sq_2: sq_2 = (x, y)
                else: sq_1, sq_2 = (x, y), None
            elif mode == 2:
                if len(polygon_points) > 2:
                    first_pt = polygon_points[0]
                    dist = np.sqrt((x - first_pt[0])**2 + (y - first_pt[1])**2)
                    if dist < CLOSE_THRESHOLD:
                        completed_polygons.append(polygon_points.copy())
                        polygon_points = []
                        return
                polygon_points.append((x, y))
    
    cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("crop", mouse_callback)
    
    while True:
        display = image.copy()
        
        # Draw completed polygons with center highlighted
        for poly in completed_polygons:
            pts = np.array(poly, np.int32)
            mask = np.zeros(display.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            overlay = display.copy()
            overlay[mask > 0] = [0, 255, 0]
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            for pt in poly:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, f"({pt[0]}, {pt[1]})", (pt[0] + 10, pt[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            center = get_center(poly)
            cv2.circle(display, center, 10, (0, 255, 255), -1)
        
        # Draw current selection
        if mode == 1 and sq_1:
            cv2.circle(display, sq_1, 5, (0, 255, 0), -1)
            cv2.putText(display, f"({sq_1[0]}, {sq_1[1]})", (sq_1[0] + 10, sq_1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if sq_2:
                cv2.rectangle(display, (min(sq_1[0], sq_2[0]), min(sq_1[1], sq_2[1])), (max(sq_1[0], sq_2[0]), max(sq_1[1], sq_2[1])), (0, 255, 0), 2)
                cv2.circle(display, sq_2, 5, (0, 255, 0), -1)
                cv2.putText(display, f"({sq_2[0]}, {sq_2[1]})", (sq_2[0] + 10, sq_2[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif mode == 2 and polygon_points:
            for pt in polygon_points:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, f"({pt[0]}, {pt[1]})", (pt[0] + 10, pt[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(polygon_points) > 1:
                pts = np.array(polygon_points + [mouse_pos], np.int32)
                cv2.polylines(display, [pts], False, (0, 255, 0), 2)
            if len(polygon_points) > 2:
                cv2.line(display, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)
                first_pt = polygon_points[0]
                dist = np.sqrt((mouse_pos[0] - first_pt[0])**2 + (mouse_pos[1] - first_pt[1])**2)
                if dist < CLOSE_THRESHOLD: cv2.circle(display, first_pt, CLOSE_THRESHOLD, (0, 255, 255), 2)
        
        cv2.imshow("crop", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): mode, sq_1, sq_2, polygon_points, completed_polygons = None, None, None, [], []
        elif key == ord('1'): mode, polygon_points = 1, []
        elif key == ord('2'): mode, sq_1 = 2, None
    
    cv2.destroyAllWindows()
    if completed_polygons: return Polygon(points=completed_polygons[-1])
    if mode == 2 and len(polygon_points) > 2: return Polygon(points=polygon_points)
    if mode == 1 and sq_1 and sq_2: return Crop(x=min(sq_1[0], sq_2[0]), y=min(sq_1[1], sq_2[1]), w=max(sq_1[0], sq_2[0])-min(sq_1[0], sq_2[0]), h=max(sq_1[1], sq_2[1])-min(sq_1[1], sq_2[1]))
    return None

if __name__ == "__main__":
    image = cv2.imread("tmp/114221.mp4_79.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop = create_crop(image)
    print(crop)