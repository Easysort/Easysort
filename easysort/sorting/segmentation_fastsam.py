from typing import List

import numpy as np
from ultralytics import FastSAM

from easysort.common.timer import TimeIt
from easysort.utils.detections import Detection, Mask


class Segmentation:
    def __init__(self):
        self.fast_sam = FastSAM("easysort/models/fastsam/FastSAM-s.pt")
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    def match_mask_to_detection(self, detections: List[Detection], masks: List[np.ndarray]) -> List[Detection]:
        for det in detections:
            overlaps = {}
            for mask in masks:
                mask_inside_detection = mask[det.xyxy[1] : det.xyxy[3], det.xyxy[0] : det.xyxy[2]]
                mask_outside_detection = np.copy(mask)
                mask_outside_detection[det.xyxy[1] : det.xyxy[3], det.xyxy[0] : det.xyxy[2]] = 0
                score = np.sum(mask_inside_detection == 1) / det.area
                score -= np.sum(mask_outside_detection == 1) / det.area
                overlaps[score] = mask
            best_score = max(overlaps.keys()) if overlaps else 0
            det.mask = overlaps[best_score] if overlaps and best_score > 0.35 else None
        return detections

    @TimeIt("Segmentation")
    def __call__(self, image: np.ndarray, detections: List[Detection]):
        if len(detections) == 0:
            return detections
        midpoints = [((det.xyxy[0] + det.xyxy[2]) / 2, (det.xyxy[1] + det.xyxy[3]) / 2) for det in detections]
        results = self.fast_sam(image, points=midpoints)
        detections = self.match_mask_to_detection(detections, Mask.from_ultralytics(results, image.shape))
        return detections
