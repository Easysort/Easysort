from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry
from typing import Dict, List
import json
import numpy as np
from tqdm import tqdm
import os
from dataclasses import dataclass

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
    skip_paths = open("skip.txt").read().splitlines()
    all_counts = []
    errors = []
    yolo_trainer = YoloTrainer()
    for j,path in enumerate(tqdm(DataRegistry.LIST("argo"), desc="Processing paths")):
        print(f"Processing {path}; Skipping {path in skip_paths}")
        if path in path_counts: continue
        if path in skip_paths: continue
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
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total paths: {len(DataRegistry.LIST('argo'))}")
    print(f"  Errors: {len(errors)}")
    print(f"  Frames with people: {sum(1 for path in path_counts for c in path_counts[path] if c > 0)}")
    print(f"  Frames without people: {sum(1 for path in path_counts for c in path_counts[path] if c == 0)}")

    # Save json with path: counts
    with open("counts.json", "w") as f:
        json.dump(path_counts, f)

    gpt_trainer = GPTTrainer()
    prompt = """You need to analyze the image and determine if there is a person, if they are walking left or right, what item they are carrying, how many of each item they are carrying, and the estimated weight of the item."""

    @dataclass
    class GPTResult:
        person: bool
        person_carrying_item: bool
        person_walking_direction: str #left or right
        item_description: [str]
        item_count: [int]
        estimated_weight_of_item: [float]


    for path in tqdm(DataRegistry.LIST("argo")[:1], desc="Processing paths"):
        if path not in path_counts: continue
        frames = [[f] for i, f in enumerate(Sampler.unpack(path, crop=Crop(x=640, y=0, w=260, h=480))) if path_counts[path][i] > 0]
        print(f"Calling OpenAI with {len(frames)} people frames out of {len(path_counts[path])} total frames")
        gpt_results = gpt_trainer._openai_call(model=gpt_trainer.default_model, prompt=prompt, image_paths=frames, output_schema=GPTResult)
        for i, (frame, result) in enumerate(zip(frames, gpt_results)):
            ResultRegistry.POST(path + "_" + str(i) + ".json", result.__dict__)
            ResultRegistry.POST(path + "_" + str(i) + ".jpg", frame)


if __name__ == "__main__":
    run()