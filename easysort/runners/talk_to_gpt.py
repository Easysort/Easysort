from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry

from tqdm import tqdm
import os
from pathlib import Path
from easysort.helpers import DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH
import json
from dataclasses import dataclass
import cv2
import traceback
import gc

# Define GPTResult dataclass at module level
@dataclass
class GPTResult:
    person: bool
    person_facing_direction: str
    person_carrying_item: bool
    people_descriptions: list[str]
    item_description: list[str]
    item_category: list[str]
    item_count: list[int]
    estimated_weight_of_item_kg: list[float]
    estimated_co2_emission_from_item_production_kg: list[float]

def extract_video_path_and_frame(result_path: str):
    """
    Extract the original video path and frame index from a result path.
    
    Result path format: .../results/argo/Device/2025/11/16/07/072703/yolov8s.pt/argo-people/108.json
    Video path format: .../data/argo/Device/2025/11/16/07/072703.mp4
    """
    # Convert result path to video path
    path_obj = Path(result_path)
    
    # Remove the result-specific parts: /yolov8s.pt/argo-people/108.json
    # Get the directory structure up to the timestamp
    parts = path_obj.parts
    
    # Find where 'argo' starts and build the path
    try:
        argo_idx = parts.index('argo')
        # Path structure: .../argo/Device/2025/11/13/20/204901/yolov8s.pt/argo-people/108.json
        # We need: .../argo/Device/2025/11/13/20/204901.mp4
        
        # Get everything up to the timestamp (which is 4 parts from the end: before yolov8s.pt/argo-people/filename)
        # parts[-1] = filename (108.json)
        # parts[-2] = argo-people
        # parts[-3] = yolov8s.pt
        # parts[-4] = timestamp (204901)
        device_and_date_parts = parts[argo_idx + 1:-4]  # Remove yolov8s.pt, argo-people, and filename
        timestamp = parts[-4]  # The timestamp directory name (e.g., "204901")
        
        # Build video path
        video_relative = Path("argo") / Path(*device_and_date_parts) / f"{timestamp}.mp4"
        video_path = os.path.join(DATA_REGISTRY_PATH, str(video_relative))
        
        # Extract frame number from filename (e.g., "108.json" -> 108)
        frame_num = int(path_obj.stem)
        
        return video_path, frame_num
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not parse result path: {result_path}") from e

def check_result_exists(video_path, gpt_model, gpt_project, frame_idx):
    """
    Check if a result already exists in the ResultRegistry.
    """
    # Construct the result file path
    result_path = Path(video_path)
    if "/mnt/" in str(result_path):
        result_path = Path(str(result_path).replace(DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH))
    
    result_file = Path(RESULTS_REGISTRY_PATH) / str(result_path.with_suffix("")) / gpt_model / gpt_project / f"{frame_idx}.json"
    return result_file.exists()

def load_single_frame(video_path, frame_idx, crop=None):
    """
    Load a single frame from video without loading the entire video into memory.
    Returns frame or None if error.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, f'Failed to open video: {video_path}'
        
        # Get total frames to validate
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx >= total_frames:
            cap.release()
            return None, f'Frame index {frame_idx} out of range (video has {total_frames} frames)'
        
        if frame_idx < 0:
            cap.release()
            return None, f'Frame index {frame_idx} is negative'
        
        # Seek to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return None, f'Failed to read frame {frame_idx} from video'
        
        # Apply crop if needed (get crop from Sampler logic)
        if crop == "auto":
            from easysort.sampler import DEVICE_TO_CROP, DATA_REGISTRY_PATH
            path_obj = Path(video_path)
            try:
                device_name = path_obj.parts[len(Path(DATA_REGISTRY_PATH).parts) + 1]
                crop = DEVICE_TO_CROP.get(device_name)
            except (IndexError, KeyError):
                crop = None
        
        if crop is not None:
            frame = frame[crop.y:crop.y+crop.h, crop.x:crop.x+crop.w]
        
        # Validate frame
        if frame.size == 0:
            return None, 'Frame is empty after cropping'
        
        # Compress/resize image to reduce API costs
        # Resize to max 1024px on longest side (OpenAI's recommended size for vision models)
        h, w = frame.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return frame, None
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return None, f'{str(e)}\nTraceback:\n{error_trace}'

def load_frame(result_path_str, gpt_model, gpt_project):
    """
    Load a single frame from video. Returns frame data or None if skipped/error.
    No caching - loads frame directly to avoid memory issues.
    """
    try:
        # Extract video path and frame index
        video_path, frame_idx = extract_video_path_and_frame(result_path_str)
        
        # Check if result already exists
        if check_result_exists(video_path, gpt_model, gpt_project, str(frame_idx)):
            return None, f'Result already exists for frame {frame_idx}'
        
        # Check if video exists
        if not os.path.exists(video_path):
            return None, f'Video not found: {video_path}'
        
        # Load single frame (no caching - direct load)
        frame, error = load_single_frame(video_path, frame_idx, crop="auto")
        if error:
            return None, error
        
        return {
            'result_path': result_path_str,
            'video_path': video_path,
            'frame_idx': frame_idx,
            'frame': frame
        }, None
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return None, f'{str(e)}\nTraceback:\n{error_trace}'

if __name__ == "__main__":
    TEST = False
    # DataRegistry.SYNC()
    yolo_model = "yolov8s.pt"
    yolo_trainer = YoloTrainer(yolo_model)
    gpt_model = "gpt-5-mini-2025-08-07" # Change to "gpt-4o" if you need better quality
    gpt_trainer = GPTTrainer(gpt_model)

    # Use a cheaper vision model - gpt-4o-mini is much cheaper and faster than gpt-5
    # Options: "gpt-4o-mini" (cheapest), "gpt-4o" (better quality, still reasonable cost)
    yolo_person_cls_idx = 0
    project = "argo-people"
    
    # Load paths and frame indices from the still frames file
    still_frames_file = "paths_still_frames.txt"
    if not os.path.exists(still_frames_file):
        print(f"Error: {still_frames_file} not found. Run argo_people_items.py first.")
        exit(1)
    
    with open(still_frames_file, "r") as f:
        result_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(result_paths)} frames to process")
    
    # If TEST mode, only process first 10 frames
    if TEST:
        result_paths = result_paths[:500]
        print(f"TEST mode: Processing only first {len(result_paths)} frames")
        # Create test output directory
        test_output_dir = Path("test_gpt_outputs")
        test_output_dir.mkdir(parents=True, exist_ok=True)
        test_output_dir_str = str(test_output_dir.absolute())
        print(f"Test outputs will be saved to: {test_output_dir_str}")
    else:
        result_paths = result_paths  # Original limit
        test_output_dir_str = None
    
    # Define your prompt here
    prompt = """
    Analyze the image and determine if there is a person, what items they are carrying. The item descriptions should only be items that people are carrying in the image.
    Sometimes people are carrying boxes, bags, etc. If there are items in the bags/boxes, describe them as items too.
    If there are multiple people, focus on the person in the center of the image.
    The description should desribe what the person looks like and what they are wearing.
    You should state if the person is facing 'left', 'right', 'forward' or 'backwards'. Make your best guess based on the image. If you are a bit unsure, be conservative in your estimates.
    You need to give the items a category out of the following list: [Køkkenting, Fritid & Have, Møbler, Boligting, Legetøj, Andet]. In some cases, people are carrying multiple of the same items. There you can use the item_count. Else just say 1.
    """
    
    # GPTResult dataclass is defined at module level
    
    gpt_project = "gpt-analysis"  # Project name for saving results
    
    # Batch size for API calls (OpenAI's _openai_call handles parallelism internally)
    batch_size = 15
    
    # Filter paths: skip already processed
    print(f"\nFiltering paths (skipping already processed)...")
    paths_to_process = []
    skipped_count = 0
    already_exists_count = 0
    
    for result_path in tqdm(result_paths, desc="Filtering paths"):
        try:
            video_path, frame_idx = extract_video_path_and_frame(result_path)
            if check_result_exists(video_path, gpt_model, gpt_project, str(frame_idx)):
                skipped_count += 1
                already_exists_count += 1
                if TEST:
                    print(f"\nSkipped (already exists): {result_path}")
            else:
                paths_to_process.append(result_path)
        except Exception as e:
            print(f"\nERROR filtering {result_path}: {e}")
    
    print(f"  {len(paths_to_process)} paths to process, {skipped_count} skipped")
    
    # Create GPT trainer (single instance, reused for all batches)
    # Process in batches: load -> call OpenAI -> save
    processed_count = 0
    error_count = 0
    
    for batch_start in tqdm(range(0, len(paths_to_process), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(paths_to_process))
        batch_paths = paths_to_process[batch_start:batch_end]
        
        # Load batch_size frames sequentially
        batch_frames = []
        batch_metadata = []
        
        for result_path in batch_paths:
            frame_data, error = load_frame(result_path, gpt_model, gpt_project)
            if error:
                if 'already exists' in error:
                    skipped_count += 1
                    already_exists_count += 1
                else:
                    error_count += 1
                    print(f"\nERROR loading {result_path}: {error}")
                continue
            
            if frame_data:
                batch_frames.append(frame_data['frame'])
                batch_metadata.append({
                    'result_path': frame_data['result_path'],
                    'video_path': frame_data['video_path'],
                    'frame_idx': frame_data['frame_idx']
                })
        
        if not batch_frames:
            continue  # Skip empty batch
        
        gpt_results = None
        try:
            # Call OpenAI (handles parallelism internally)
            gpt_results = gpt_trainer._openai_call(
                model=gpt_model,
                prompt=prompt,
                image_paths=[[frame] for frame in batch_frames],  # Each image as a list
                output_schema=GPTResult,
                max_workers=batch_size  # Use batch_size workers for parallel API calls
            )
            
            # Save results sequentially
            for i, (result, metadata) in enumerate(zip(gpt_results, batch_metadata)):
                try:
                    result_dict = result.__dict__
                    
                    # Save to ResultRegistry
                    ResultRegistry.POST(
                        path=metadata['video_path'],
                        model=gpt_model,
                        project=gpt_project,
                        identifier=str(metadata['frame_idx']),
                        data=result_dict
                    )
                    
                    # If in test mode, also save image and response to test directory
                    if TEST and test_output_dir_str:
                        test_output_dir = Path(test_output_dir_str)
                        test_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save image
                        image_filename = f"frame_{metadata['frame_idx']}_image.jpg"
                        image_path = test_output_dir / image_filename
                        cv2.imwrite(str(image_path), batch_frames[i])
                        
                        # Save GPT response as JSON
                        response_filename = f"frame_{metadata['frame_idx']}_response.json"
                        response_path = test_output_dir / response_filename
                        with open(response_path, 'w') as f:
                            json.dump(result_dict, f, indent=2)
                    
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"\nERROR saving result for {metadata['result_path']}: {e}")
                    if TEST:
                        print(traceback.format_exc())
        except Exception as e:
            print(f"\nERROR in batch {batch_start//batch_size + 1}: {e}")
            error_count += len(batch_frames)
            if TEST:
                print(traceback.format_exc())
        finally:
            # Explicitly clear batch data to free memory before next batch
            # This ensures cleanup happens even if there's an error
            del batch_frames
            del batch_metadata
            if gpt_results is not None:
                del gpt_results
            gc.collect()  # Force garbage collection to free memory
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count} frames")
    print(f"  Skipped: {skipped_count} frames (including {already_exists_count} that already existed)")
    print(f"  Errors: {error_count} frames")
    print(f"  Results saved to project: {gpt_project}")
    if TEST:
        print(f"  Test outputs saved to: {test_output_dir.absolute()}")