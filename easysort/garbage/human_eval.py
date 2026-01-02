import cv2
from pathlib import Path
from typing import List
import random
from PIL import Image
from tqdm import tqdm
from easysort.sampler import Sampler
from easysort.gpt_trainer import GPTTrainer
import os
import tkinter as tk
from ultralytics import YOLO
from dataclasses import dataclass
import numpy as np
import json

@dataclass
class HumanDetection:
    person_facing_direction: str  # left, right, forward, backward
    human_visibility: bool

# Global prompt
GPT_PROMPT = "Analyze the image. Determine: 1) person_facing_direction (Which way is the person facing: left, right, forward, backward), 2) human_visibility (true if you can see the most of the person (okay if they are behind objects, or partly hidden behind left-center beam/tent) and they are close to entrance (center of the image), false otherwise)."

def label_folder(folder: str, suffix: str = ".label"):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in exts]
    total_images = len(images)
    
    # Count already labeled images
    labeled_count = sum(1 for img_path in images if img_path.with_suffix(suffix).exists())
    
    # Get screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    # Calculate maximum 16:9 size that fits on screen
    aspect_ratio = 16 / 9
    if screen_width / screen_height > aspect_ratio:
        # Screen is wider than 16:9, limit by height
        display_height = int(screen_height * 0.9)  # Use 90% of screen height
        display_width = int(display_height * aspect_ratio)
    else:
        # Screen is taller than 16:9, limit by width
        display_width = int(screen_width * 0.9)  # Use 90% of screen width
        display_height = int(display_width / aspect_ratio)
    
    window_name = "Label: 0-9, b=back, q=quit, 1=in, 2=out"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # Initialize models
    yolo_model = YOLO("yolo11n.pt")
    gpt_trainer = GPTTrainer()
    
    # Sort all image paths
    images = sorted(images)
    
    if not images:
        cv2.destroyAllWindows()
        return
    
    current_idx = 0
    going_back = False
    gpt_cache = {}  # Cache GPT results
    
    def get_batch_detections(start_idx, batch_size=40):
        batch_images = []
        batch_indices = []
        for i in range(start_idx, min(start_idx + batch_size, len(images))):
            img_path = images[i]
            label_path = img_path.with_suffix(suffix)
            if label_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            yolo_results = yolo_model(img, classes=[0], verbose=False)
            if sum(len(r.boxes) for r in yolo_results) == 0:
                continue
            batch_images.append([img])
            batch_indices.append(i)
        
        if not batch_images:
            return {}
        
        detections = gpt_trainer._openai_call(gpt_trainer.model, GPT_PROMPT, batch_images, HumanDetection, max_workers=20)
        
        # Save to .gpt files and return cache
        result = {}
        for i, detection in enumerate(detections):
            idx = batch_indices[i]
            img_path = images[idx]
            gpt_path = img_path.with_suffix(".gpt")
            gpt_path.write_text(json.dumps({
                "person_facing_direction": detection.person_facing_direction,
                "human_visibility": detection.human_visibility
            }))
            result[idx] = detection
        return result
    
    while 0 <= current_idx < len(images):
        img_path = images[current_idx]
        label_path = img_path.with_suffix(suffix)
        
        # Skip already labeled images when going forward (unless going back)
        if not going_back and label_path.exists():
            current_idx += 1
            continue
        
        # Load and detect people
        img = cv2.imread(str(img_path))
        if img is None:
            current_idx += 1
            continue
        
        yolo_results = yolo_model(img, classes=[0], verbose=False)
        num_detections = sum(len(r.boxes) for r in yolo_results)
        
        # Auto-label as 0 if no people detected and skip showing (but not when going back)
        if not going_back and num_detections == 0:
            label_path.write_text("0")
            current_idx += 1
            continue
        
        going_back = False
        
        # Get GPT detection (from .gpt file, cache, or fetch batch) - only if people detected
        gpt_detection = None
        if num_detections > 0:
            gpt_path = img_path.with_suffix(".gpt")
            
            # Try to load from .gpt file first
            if gpt_path.exists():
                try:
                    gpt_data = json.loads(gpt_path.read_text())
                    gpt_detection = HumanDetection(
                        person_facing_direction=gpt_data["person_facing_direction"],
                        human_visibility=gpt_data["human_visibility"]
                    )
                    gpt_cache[current_idx] = gpt_detection
                except:
                    pass
            
            # If no .gpt file, check cache or fetch batch
            if gpt_detection is None:
                if current_idx not in gpt_cache:
                    # Show waiting message
                    cv2.putText(img, "Waiting for GPT detections...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    img_resized = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow(window_name, img_resized)
                    cv2.waitKey(1)
                    
                    # Fetch batch
                    batch_results = get_batch_detections(current_idx)
                    gpt_cache.update(batch_results)
                
                gpt_detection = gpt_cache.get(current_idx)
        
        # Recalculate labeled count
        labeled_count = sum(1 for img_path in images if img_path.with_suffix(suffix).exists())
        
        # Update window title with progress
        progress_text = f"Space=accept, 0-9=overwrite, b=back, q=quit, 1=in, 2=out | {labeled_count}/{total_images} labeled"
        cv2.setWindowTitle(window_name, progress_text)
        
        # Draw YOLO detections
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display GPT detection
        if gpt_detection:
            direction = gpt_detection.person_facing_direction
            visibility = gpt_detection.human_visibility
            # Calculate label GPT would assign
            if direction == "right" and visibility:
                gpt_label = "1"
            elif direction == "left" and visibility:
                gpt_label = "2"
            else:
                gpt_label = "0"
            text = f"GPT: {direction}, visible={visibility}, label={gpt_label}"
            cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        img_resized = cv2.resize(img, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, img_resized)
        
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('b'), ord('B')):
            if current_idx > 0:
                current_idx -= 1
                going_back = True
            continue
        elif key == ord(' '):  # Space to accept GPT detection
            if gpt_detection:
                direction = gpt_detection.person_facing_direction
                visibility = gpt_detection.human_visibility
                if direction == "right" and visibility:
                    label = "1"
                elif direction == "left" and visibility:
                    label = "2"
                else:
                    label = "0"
                label_path.write_text(label)
                labeled_count += 1
                current_idx += 1
        elif ord('0') <= key <= ord('9'):
            label_path.write_text(str(key - ord('0')))
            labeled_count += 1
            current_idx += 1
        else:
            current_idx += 1
    
    cv2.destroyAllWindows()

def unpack_and_save(paths: List[str]):
    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    for _ in tqdm(range(10)):
        path = random.choice(paths)
        frames = Sampler.unpack(path, crop="auto")
        for frame_idx, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(f"{output_dir}/{path.split('/')[-1]}_{frame_idx}.jpg")

def reset_folder(folder: str):
    files = os.listdir(folder)
    label_files = [file for file in files if file.endswith(".label")]
    for label_file in label_files:
        Path(folder, label_file).unlink()

def delete_last_labels(folder: str, x: int, suffix: str = ".label"):
    """Delete the last x label files based on modification time."""
    folder_path = Path(folder)
    label_files = sorted(folder_path.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    for label_file in label_files[:x]:
        label_file.unlink()
    print(f"Deleted {min(x, len(label_files))} label(s)")

def generate_gpt_detections(folder: str, label_suffix: str = ".label", gpt_suffix: str = ".gpt", batch_size: int = 40):
    """Generate GPT detections for all unlabeled images and save to .gpt files."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in exts])
    
    yolo_model = YOLO("yolo11n.pt")
    gpt_trainer = GPTTrainer()
    
    # Find images that need GPT detection
    images_to_process = []
    for img_path in tqdm(images):
        label_path = img_path.with_suffix(label_suffix)
        gpt_path = img_path.with_suffix(gpt_suffix)
        if label_path.exists() or gpt_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        yolo_results = yolo_model(img, classes=[0], verbose=False)
        if sum(len(r.boxes) for r in yolo_results) > 0:
            images_to_process.append((img_path, img))
    
    if not images_to_process:
        print("No images need GPT detection")
        return
    
    print(f"Processing {len(images_to_process)} images in batches of {batch_size}...")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(images_to_process), batch_size), desc="Batches"):
        batch = images_to_process[batch_start:batch_start + batch_size]
        batch_images = [[img] for _, img in batch]
        batch_paths = [img_path for img_path, _ in batch]
        
        detections = gpt_trainer._openai_call(gpt_trainer.model, GPT_PROMPT, batch_images, HumanDetection, max_workers=20)
        
        # Save to .gpt files
        for img_path, detection in zip(batch_paths, detections):
            gpt_path = img_path.with_suffix(gpt_suffix)
            gpt_path.write_text(json.dumps({
                "person_facing_direction": detection.person_facing_direction,
                "human_visibility": detection.human_visibility
            }))
    
    print(f"Generated {len(images_to_process)} GPT detections")


if __name__ == "__main__":
    folder = "tmp"
    label_folder(folder)
