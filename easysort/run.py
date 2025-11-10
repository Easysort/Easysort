from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry
from typing import Dict, List
import json
import numpy as np
from tqdm import tqdm
import os

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
    DataRegistry.SYNC()
    path_counts: Dict[str, List[int]] = json.load(open("counts.json")) if os.path.exists("counts.json") else {}
    all_counts = []
    errors = []
    yolo_trainer = YoloTrainer()
    for j,path in enumerate(tqdm(DataRegistry.LIST("argo"), desc="Processing paths")):
        try:
            if path in path_counts: continue
            path_counts[path] = []
            frames = Sampler.unpack(path, crop=Crop(x=640, y=0, w=260, h=480))
            print(f"Loaded {len(frames)} frames")
            batch_size = 32
            for i in tqdm(range(0, len(frames), batch_size), desc="Processing batches"):
                batch = frames[i:i+batch_size]
                counts = yolo_trainer._is_person_in_image(batch)
                all_counts.extend(counts)
                path_counts[path].extend(counts)
            if j % 20 == 0:
                print(f"Saving counts...")
                with open("counts.json", "w") as f:
                    json.dump(path_counts, f)
        except Exception as e:
            errors.append(path)
            print(f"Error processing {path}: {e}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total paths: {len(DataRegistry.LIST('argo'))}")
    print(f"  Total frames: {len(frames)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Frames with people: {sum(1 for c in all_counts if c > 0)}")
    print(f"  Frames without people: {sum(1 for c in all_counts if c == 0)}")
    print(f"  Average people per frame (when detected): {np.mean([c for c in all_counts if c > 0]) if any(c > 0 for c in all_counts) else 0:.2f}")

    # Save json with path: counts
    with open("counts.json", "w") as f:
        json.dump(path_counts, f)

    # gpt_trainer = GPTTrainer()
    # images = [Sampler.unpack(path, crop=Crop(x=640, y=0, w=260, h=480)) for path in DataRegistry.LIST("argo")]

if __name__ == "__main__":
    run()