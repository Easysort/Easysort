
import cv2
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from easysort.utils.detections import Detection

def visualize_sorting_pipeline_image(image: np.ndarray, detections: List[Detection], show_plot: bool = True) -> np.ndarray:
    def get_color_for_class(class_name: str) -> tuple:
        # Generate a unique color based on the class name using HSV color space
        # Hash the string to get a consistent numeric value
        class_id = hash(class_name) % 360
        hue = (class_id * 137) % 360  # Use prime number to get good distribution
        sat = 255
        val = 255

        # Convert HSV to BGR
        h = float(hue) / 2  # OpenCV uses hue values 0-180
        hsv = np.array([[[h, sat, val]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(map(int, bgr[0,0]))

    def draw_detection(img: np.ndarray, detection: Detection) -> np.ndarray:
        color = get_color_for_class(detection.class_name)

        cv2.rectangle(img, (int(detection.xyxy[0]), int(detection.xyxy[1])),
                            (int(detection.xyxy[2]), int(detection.xyxy[3])), color, 2)

        text_pos = (int(detection.xyxy[0]), int(detection.xyxy[1] - 10))
        cv2.putText(img, f"{detection.class_name} (z={detection.center_point[2]})", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if detection.mask is not None and len(detection.mask) > 0:
            mask = cv2.resize(detection.mask.astype(np.uint8), (img.shape[1], img.shape[0]))
            overlay = img.copy()
            overlay[mask > 0] = color
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        cx, cy, _ = map(int, detection.center_point)
        size = 10
        cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), 2)
        cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), (0, 0, 255), 2)
        return img

    n_plots = len(detections) + 1
    main_view = image.copy()
    for det in detections: main_view = draw_detection(main_view, det)

    if show_plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, n_plots, 1)
        plt.imshow(cv2.cvtColor(main_view, cv2.COLOR_BGR2RGB))
        plt.title('All Detections')
        plt.axis('off')

        for idx, det in enumerate(detections, 1):
            plt.subplot(1, n_plots, idx + 1)
            det_view = draw_detection(image.copy(), det)
            plt.imshow(cv2.cvtColor(det_view, cv2.COLOR_BGR2RGB))
            plt.title(f'Detection {idx} ({det.class_name})')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    return main_view