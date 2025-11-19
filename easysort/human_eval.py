import cv2
from pathlib import Path
from easysort.registry import Registry

class HumanEvaluator:
    def __init__(self, paths: list[str], project: str):
        self.video_paths = paths
        self.project = project
        self.model = "HumanEval"
        self.results = {}
    
    def run(self):
        idx = 0
        window_name = "Human Evaluation - Press: Space=0, 1-9=1-9, q=quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while idx < len(self.video_paths):
            path = self.video_paths[idx]
            video_capture = Registry.GET(path, loader=lambda x: cv2.VideoCapture(x))
            
            while True:
                ret, frame = video_capture.read()
                if not ret: break
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(0) & 0xFF
        
        cv2.destroyAllWindows()
        self.save()
    
    def save(self):
        """Save results to ResultRegistry."""
        for path, label in self.results.items():
            key = Registry.construct_path(path, self.model, self.project, "label")
            Registry.POST(key, {"label": label})
        print(f"Saved {len(self.results)} labels to {self.project}")

if __name__ == "__main__":
    paths = Registry.LIST(prefix="argo", suffix=".mp4")
    HumanEvaluator(paths, "people_with_items").run()
