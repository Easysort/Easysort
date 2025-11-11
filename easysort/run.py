from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry
from typing import Dict, List
import json
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass
import cv2

# def create_montage(images: list[np.ndarray], cols: int = 10) -> np.ndarray:
#     """Create a grid montage of images."""
#     if not images:
#         return np.zeros((100, 100, 3), dtype=np.uint8)
    
#     rows = (len(images) + cols - 1) // cols
    
#     # Resize all images to same size
#     h, w = 150, 200  # Thumbnail size
#     resized = [cv2.resize(img, (w, h)) for img in images]
    
#     # Create grid
#     montage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
#     for idx, img in enumerate(resized):
#         row, col = idx // cols, idx % cols
#         montage[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
#     return montage

# def visualize_results(frames: list[np.ndarray], person_counts: list[int], output_dir: Path = Path("validation_output"), max_per_montage: int = 100):
#     """Visualize images with and without people. Creates multiple montages if needed."""
#     output_dir.mkdir(exist_ok=True)
    
#     # Separate frames by detection
#     with_people = [frames[i] for i, count in enumerate(person_counts) if count > 0]
#     without_people = [frames[i] for i, count in enumerate(person_counts) if count == 0]
    
#     print(f"\nResults:")
#     print(f"  Images with people: {len(with_people)}")
#     print(f"  Images without people: {len(without_people)}")
    
#     # Create multiple montages for images with people
#     if with_people:
#         num_montages = (len(with_people) + max_per_montage - 1) // max_per_montage
#         for i in range(num_montages):
#             start_idx = i * max_per_montage
#             end_idx = min(start_idx + max_per_montage, len(with_people))
#             chunk = with_people[start_idx:end_idx]
#             montage = create_montage(chunk, cols=10)
#             filename = f"with_people_{i+1}.jpg" if num_montages > 1 else "with_people.jpg"
#             cv2.imwrite(str(output_dir / filename), montage)
#             print(f"  Saved montage: {output_dir / filename} ({len(chunk)} images)")
    
#     # Create multiple montages for images without people
#     if without_people:
#         num_montages = (len(without_people) + max_per_montage - 1) // max_per_montage
#         for i in range(num_montages):
#             start_idx = i * max_per_montage
#             end_idx = min(start_idx + max_per_montage, len(without_people))
#             chunk = without_people[start_idx:end_idx]
#             montage = create_montage(chunk, cols=10)
#             filename = f"without_people_{i+1}.jpg" if num_montages > 1 else "without_people.jpg"
#             cv2.imwrite(str(output_dir / filename), montage)
#             print(f"  Saved montage: {output_dir / filename} ({len(chunk)} images)")
    
#     # Also save individual examples (first 10 of each)
#     examples_dir = output_dir / "examples"
#     examples_dir.mkdir(exist_ok=True)
    
#     for i, frame in enumerate(with_people[:10]):
#         cv2.imwrite(str(examples_dir / f"with_people_{i}.jpg"), frame)
    
#     for i, frame in enumerate(without_people[:10]):
#         cv2.imwrite(str(examples_dir / f"without_people_{i}.jpg"), frame)

def run():
    # DataRegistry.SYNC()
    path_counts: Dict[str, List[int]] = json.load(open("counts.json")) if os.path.exists("counts.json") else {}
    if os.getenv("VIEW", "0") > 0:
        path_counts = {path.replace("/mnt/c/Users/lucas/Desktop/data/", "/Volumes/Easysort128/data/"): path_counts[path] for path in path_counts}
    
    # Load bboxes if they exist
    path_bboxes: Dict[str, List[List]] = {}
    if os.path.exists("bboxes.json"):
        path_bboxes = json.load(open("bboxes.json"))
        path_bboxes = {path.replace("/mnt/c/Users/lucas/Desktop/data/", "/Volumes/Easysort128/data/"): path_bboxes[path] for path in path_bboxes}
    
    skip_paths = open("skip.txt").read().splitlines()
    all_counts = []
    errors = []
    yolo_trainer = YoloTrainer()
    
    for j,path in enumerate(tqdm(DataRegistry.LIST("argo")[:10], desc="Processing paths")):
        print(f"Processing {path}; Skipping {path in skip_paths}")
        if path in path_counts: continue
        if path in skip_paths: continue
        path_counts[path] = []
        path_bboxes[path] = []  # Store bboxes per frame
        frames = Sampler.unpack(path, crop="auto")
        print(f"Loaded {len(frames)} frames")
        batch_size = 32
        for i in tqdm(range(0, len(frames), batch_size), desc="Processing batches"):
            batch = frames[i:i+batch_size]
            # Get full YOLO results to extract bboxes
            results = yolo_trainer.model(batch, verbose=False)
            for result in results:
                count = 0
                bboxes = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    # Filter for person class (class 0)
                    for box, cls, conf in zip(boxes, classes, confidences):
                        if int(cls) == 0:  # Person
                            count += 1
                            # Save bbox as [x1, y1, x2, y2, confidence]
                            bboxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)])
                
                path_counts[path].append(count)
                path_bboxes[path].append(bboxes)  # List of bboxes for this frame
        
        if j % 20 == 0:
            print(f"Saving counts and bboxes...")
            with open("counts.json", "w") as f:
                json.dump(path_counts, f)
            with open("bboxes.json", "w") as f:
                json.dump(path_bboxes, f)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total paths: {len(DataRegistry.LIST('argo'))}")
    print(f"  Errors: {len(errors)}")
    print(f"  Frames with people: {sum(1 for path in path_counts for c in path_counts[path] if c > 0)}")
    print(f"  Frames without people: {sum(1 for path in path_counts for c in path_counts[path] if c == 0)}")

    with open("counts.json", "w") as f:
        json.dump(path_counts, f)
    with open("bboxes.json", "w") as f:
        json.dump(path_bboxes, f)


    if os.getenv("VIEW", "0") == "0": return
    # Show all frames with people
    all_frames_with_people = []
    for path in tqdm(DataRegistry.LIST("argo")[:10], desc="Collecting frames"):
        if path not in path_counts: 
            continue
        frames = Sampler.unpack(path, crop="auto")
        for i, frame in enumerate(frames):
            if i < len(path_counts[path]):
                all_frames_with_people.append({
                    'frame': frame,
                    'path': path,
                    'frame_idx': i,
                    'has_person': path_counts[path][i] > 0,
                    'person_count': path_counts[path][i],
                    'bboxes': path_bboxes.get(path, [[]] * len(path_counts[path]))[i] if path in path_bboxes else []
                })
    
    if not all_frames_with_people:
        print("No frames with people found")
        return
    
    print(f"\nTotal frames: {len(all_frames_with_people)}")
    print(f"Frames with people: {sum(1 for f in all_frames_with_people if f['has_person'])}")
    print("\nControls: 'n' = next, 'b' = previous, 'q' = quit")
    
    cur_idx = 0
    window_name = "Human Detection Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while 0 <= cur_idx < len(all_frames_with_people):
        item = all_frames_with_people[cur_idx]
        img = item['frame'].copy()
        h, w = img.shape[:2]
        
        # Draw bounding boxes
        for bbox in item['bboxes']:
            if len(bbox) >= 4:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                conf = bbox[4] if len(bbox) > 4 else 1.0
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw confidence score
                cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display frame info
        frame_num = f"Frame {cur_idx + 1}/{len(all_frames_with_people)}"
        path_text = f"Path: {item['path'].split('/')[-1]}"
        frame_idx_text = f"Frame index: {item['frame_idx']}"
        
        # Human detection status - large and prominent
        if item['has_person']:
            status_text = f"HUMAN DETECTED: {item['person_count']} person(s)"
            color = (0, 255, 0)  # Green
        else:
            status_text = "NO HUMAN"
            color = (0, 0, 255)  # Red
        
        # Draw status - large text
        cv2.putText(img, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        cv2.putText(img, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)  # Black outline
        
        # Frame info
        cv2.putText(img, frame_num, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, path_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, frame_idx_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Instructions
        instructions = "Press 'n' for next | 'b' for previous | 'q' to quit"
        cv2.putText(img, instructions, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, img)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key in (ord('n'), ord('N')):  # next
            cur_idx += 1
        elif key in (ord('b'), ord('B')):  # previous
            cur_idx = max(0, cur_idx - 1)
        elif key in (ord('q'), 27):  # quit
            break
    
    cv2.destroyAllWindows()

    # # Save json with path: counts
    # with open("counts.json", "w") as f:
    #     json.dump(path_counts, f)

    # gpt_trainer = GPTTrainer()
    # prompt = """You need to analyze the image and determine if there is a person, if they are walking left or right, what item they are carrying, how many of each item they are carrying, and the estimated weight of the item."""

    # @dataclass
    # class GPTResult:
    #     person: bool
    #     person_carrying_item: bool
    #     person_walking_direction: str #left or right
    #     item_description: [str]
    #     item_count: [int]
    #     estimated_weight_of_item: [float]


    # for path in tqdm(DataRegistry.LIST("argo")[:1], desc="Processing paths"):
    #     if path not in path_counts: continue
    #     frames = [[f] for i, f in enumerate(Sampler.unpack(path, crop=Crop(x=640, y=0, w=260, h=480))) if path_counts[path][i] > 0][:10]
    #     print(f"Calling OpenAI with {len(frames)} people frames out of {len(path_counts[path])} total frames")
    #     gpt_results = gpt_trainer._openai_call(model=gpt_trainer.default_model, prompt=prompt, image_paths=frames, output_schema=GPTResult)
    #     for i, (frame, result) in enumerate(zip(frames, gpt_results)):
    #         ResultRegistry.POST(path.replace(".mp4", "") + "_" + str(i) + ".json", result.__dict__)
    #         ResultRegistry.POST(path.replace(".mp4", "") + "_" + str(i) + ".jpg", frame[0])


if __name__ == "__main__":
    run()