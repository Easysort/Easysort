from easysort.sampler import Sampler, Crop
from easysort.gpt_trainer import GPTTrainer, YoloTrainer
from easysort.registry import DataRegistry, ResultRegistry

from tqdm import tqdm
from ultralytics.engine.results import Results
import datetime
from easysort.helpers import Sort
import os
from pathlib import Path
from easysort.helpers import DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH
import json
import numpy as np
from collections import defaultdict


def get_box_center(box):
    """
    Get the center point of a bounding box.
    Box format: [x1, y1, x2, y2, conf]
    Returns: (cx, cy) tuple
    """
    x1, y1, x2, y2 = box[:4]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (cx, cy)


def get_box_area(box):
    """Get the area of a bounding box."""
    x1, y1, x2, y2 = box[:4]
    return (x2 - x1) * (y2 - y1)


def calculate_distance(box1, box2):
    """
    Calculate the Euclidean distance between centers of two boxes.
    Returns: float distance
    """
    center1 = get_box_center(box1)
    center2 = get_box_center(box2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)


def match_boxes(prev_boxes, curr_boxes, max_distance_threshold=50):
    """
    Match boxes between consecutive frames using center distance.
    
    Args:
        prev_boxes: List of boxes from previous frame
        curr_boxes: List of boxes from current frame
        max_distance_threshold: Maximum distance (in pixels) to consider a match
    
    Returns:
        List of tuples (prev_idx, curr_idx) representing matches
    """
    matches = []
    used_curr = set()
    
    # Sort by area (larger boxes first) for better matching
    prev_with_idx = sorted(enumerate(prev_boxes), key=lambda x: get_box_area(x[1]), reverse=True)
    curr_with_idx = sorted(enumerate(curr_boxes), key=lambda x: get_box_area(x[1]), reverse=True)
    
    for prev_idx, prev_box in prev_with_idx:
        best_match = None
        best_distance = float('inf')
        
        for curr_idx, curr_box in curr_with_idx:
            if curr_idx in used_curr:
                continue
            
            distance = calculate_distance(prev_box, curr_box)
            if distance < best_distance and distance <= max_distance_threshold:
                best_distance = distance
                best_match = curr_idx
        
        if best_match is not None:
            matches.append((prev_idx, best_match))
            used_curr.add(best_match)
    
    return matches


def find_still_sequences(tracked_people, min_sequence_length=3, max_movement_per_frame=30):
    """
    Find sequences where people are standing still.
    
    Args:
        tracked_people: Dict mapping person_id -> list of (frame_num, box) tuples
        min_sequence_length: Minimum number of consecutive frames to consider a sequence
        max_movement_per_frame: Maximum movement (in pixels) between frames to consider "still"
    
    Returns:
        List of tuples (person_id, start_frame, end_frame, middle_frame)
    """
    still_sequences = []
    
    for person_id, track in tracked_people.items():
        if len(track) < min_sequence_length:
            continue
        
        # Sort by frame number
        track = sorted(track, key=lambda x: x[0])
        
        # Find consecutive sequences with low movement
        sequence_start = 0
        for i in range(1, len(track)):
            prev_frame, prev_box = track[i-1]
            curr_frame, curr_box = track[i]
            
            # Check if frames are consecutive
            if curr_frame != prev_frame + 1:
                # Gap in sequence, check if previous sequence is valid
                if i - sequence_start >= min_sequence_length:
                    # Calculate middle frame
                    middle_idx = sequence_start + (i - sequence_start) // 2
                    middle_frame = track[middle_idx][0]
                    still_sequences.append((person_id, track[sequence_start][0], track[i-1][0], middle_frame))
                sequence_start = i
                continue
            
            # Calculate movement between consecutive frames
            movement = calculate_distance(prev_box, curr_box)
            
            if movement > max_movement_per_frame:
                # Movement too high, check if previous sequence is valid
                if i - sequence_start >= min_sequence_length:
                    middle_idx = sequence_start + (i - sequence_start) // 2
                    middle_frame = track[middle_idx][0]
                    still_sequences.append((person_id, track[sequence_start][0], track[i-1][0], middle_frame))
                sequence_start = i
        
        # Check final sequence
        if len(track) - sequence_start >= min_sequence_length:
            middle_idx = sequence_start + (len(track) - sequence_start) // 2
            middle_frame = track[middle_idx][0]
            still_sequences.append((person_id, track[sequence_start][0], track[-1][0], middle_frame))
    
    return still_sequences


def find_moving_sequences(tracked_people, min_sequence_length=2, max_movement_per_frame=30):
    """
    Find sequences where people are moving (not still).
    
    Args:
        tracked_people: Dict mapping person_id -> list of (frame_num, box) tuples
        min_sequence_length: Minimum number of consecutive frames to consider a sequence
        max_movement_per_frame: Maximum movement (in pixels) between frames to consider "still"
    
    Returns:
        List of tuples (person_id, start_frame, end_frame, selected_frame)
        where selected_frame is the middle frame for 2-4 frame sequences,
        or the frame closest to image center for longer sequences
    """
    moving_sequences = []
    
    for person_id, track in tracked_people.items():
        if len(track) < min_sequence_length:
            continue
        
        # Sort by frame number
        track = sorted(track, key=lambda x: x[0])
        
        # Find consecutive sequences with movement
        sequence_start = 0
        for i in range(1, len(track)):
            prev_frame, prev_box = track[i-1]
            curr_frame, curr_box = track[i]
            
            # Check if frames are consecutive
            if curr_frame != prev_frame + 1:
                # Gap in sequence, check if previous sequence is valid
                if i - sequence_start >= min_sequence_length:
                    sequence_frames = track[sequence_start:i]
                    selected_frame = _select_frame_from_sequence(sequence_frames)
                    if selected_frame is not None:
                        moving_sequences.append((person_id, track[sequence_start][0], track[i-1][0], selected_frame))
                sequence_start = i
                continue
            
            # Calculate movement between consecutive frames
            movement = calculate_distance(prev_box, curr_box)
            
            if movement <= max_movement_per_frame:
                # Movement too low (still), check if previous sequence is valid
                if i - sequence_start >= min_sequence_length:
                    sequence_frames = track[sequence_start:i]
                    selected_frame = _select_frame_from_sequence(sequence_frames)
                    if selected_frame is not None:
                        moving_sequences.append((person_id, track[sequence_start][0], track[i-1][0], selected_frame))
                sequence_start = i
        
        # Check final sequence
        if len(track) - sequence_start >= min_sequence_length:
            sequence_frames = track[sequence_start:]
            selected_frame = _select_frame_from_sequence(sequence_frames)
            if selected_frame is not None:
                moving_sequences.append((person_id, track[sequence_start][0], track[-1][0], selected_frame))
    
    return moving_sequences


def _select_frame_from_sequence(sequence_frames, image_width=None, image_height=None):
    """
    Select the best frame from a moving sequence.
    For 2-4 frames: return middle frame
    For >4 frames: return frame where person is closest to image center
    
    Args:
        sequence_frames: List of (frame_num, box) tuples
        image_width: Image width (if None, estimated from boxes)
        image_height: Image height (if None, estimated from boxes)
    
    Returns:
        frame_num of selected frame, or None if sequence is invalid
    """
    if len(sequence_frames) < 2:
        return None
    
    # For 2-4 frame sequences, return middle frame
    if len(sequence_frames) <= 4:
        middle_idx = len(sequence_frames) // 2
        return sequence_frames[middle_idx][0]
    
    # For >4 frames, find frame where person is closest to image center
    # Estimate image dimensions from boxes if not provided
    if image_width is None or image_height is None:
        max_x = max(box[2] for _, box in sequence_frames)  # x2
        max_y = max(box[3] for _, box in sequence_frames)  # y2
        # Add some padding (assume boxes don't go to absolute edge)
        estimated_width = int(max_x * 1.1)
        estimated_height = int(max_y * 1.1)
        image_width = image_width if image_width is not None else estimated_width
        image_height = image_height if image_height is not None else estimated_height
    
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    # Find frame where box center is closest to image center
    best_frame = None
    best_distance = float('inf')
    
    for frame_num, box in sequence_frames:
        box_center = get_box_center(box)
        distance_to_center = np.sqrt(
            (box_center[0] - image_center_x)**2 + 
            (box_center[1] - image_center_y)**2
        )
        if distance_to_center < best_distance:
            best_distance = distance_to_center
            best_frame = frame_num
    
    return best_frame


if __name__ == "__main__":
    # DataRegistry.SYNC()
    yolo_model = "yolov8s.pt"
    yolo_trainer = YoloTrainer(yolo_model)
    gpt_trainer = GPTTrainer()
    gpt_model = gpt_trainer.model
    yolo_person_cls_idx = 0
    project = "argo-people"
    all_paths = DataRegistry.LIST("argo")
    data = list(Sort.since(all_paths, datetime.datetime(2025, 11, 10)))
    data = list(Sort.before(data, datetime.datetime(2025, 11, 17)))
    print(f"Processing {len(data)} paths out of {len(all_paths)}")

    number_of_frames = 0
    total_frames = 0
    
    # Minimum box area threshold (in pixels squared) - adjust as needed
    MIN_BOX_AREA = 500  # Example: 500 pixels^2, adjust based on your needs
    
    paths_with_people = set()
    if os.path.exists("paths_with_people.txt"):
        with open("paths_with_people.txt", "r") as f:
            paths_with_people = set(line.strip() for line in f if line.strip())

    paths_to_check = list(set(["/".join(path.split("/")[:-1]) for path in tqdm(paths_with_people)]))

    # Collect all frames that should be checked (kept after filtering)
    frames_to_check = []
    
    # Track totals across all paths
    total_original_frames = 0
    total_kept_frames = 0
    total_small_boxes_removed = 0

    for path in tqdm(paths_to_check, desc="Processing paths"):
        # Load all frames for this path
        items = {}
        frame_numbers = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if file_path in paths_with_people:
                try:
                    frame_num = int(file.split(".")[0])
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    # Assert data is a list of boxes
                    assert isinstance(data, list), f"Expected list, got {type(data)}"
                    if len(data) > 0:
                        # Filter out small boxes
                        filtered_boxes = []
                        for box in data:
                            assert len(box) == 5, f"Expected 5 elements in box, got {len(box)}"
                            assert all(isinstance(x, (int, float)) for x in box), "Box elements must be numeric"
                            box_area = get_box_area(box)
                            if box_area >= MIN_BOX_AREA:
                                filtered_boxes.append(box)
                            else:
                                total_small_boxes_removed += 1
                        
                        # Only add frame if it has at least one valid (non-small) box
                        if len(filtered_boxes) > 0:
                            items[frame_num] = filtered_boxes
                            frame_numbers.append(frame_num)
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Warning: Skipping invalid file {file_path}: {e}")
                    continue
        
        if len(items) == 0:
            continue
        
        # Sort frame numbers to process in order
        frame_numbers = sorted(frame_numbers)
        assert len(frame_numbers) == len(items), "Frame number count mismatch"
        
        print(f"\nProcessing {len(frame_numbers)} frames from {path}")
        original_frame_count = len(frame_numbers)
        total_original_frames += original_frame_count  # Track total
        
        # Track people across frames
        # Structure: person_id -> list of (frame_num, box) tuples
        tracked_people = defaultdict(list)
        next_person_id = 0
        frame_to_person_map = {}  # (frame_num, box_idx) -> person_id
        
        prev_frame_num = None
        prev_boxes = None
        prev_box_to_person = {}  # box_idx -> person_id
        
        for frame_num in frame_numbers:
            curr_boxes = items[frame_num]
            
            if len(curr_boxes) == 0:
                prev_frame_num = frame_num
                prev_boxes = []
                prev_box_to_person = {}
                continue
            
            curr_box_to_person = {}
            
            if prev_frame_num is not None and prev_boxes:
                # Match boxes
                matches = match_boxes(prev_boxes, curr_boxes)
                
                matched_curr_indices = set()
                for prev_idx, curr_idx in matches:
                    # Assign same person_id
                    person_id = prev_box_to_person[prev_idx]
                    curr_box_to_person[curr_idx] = person_id
                    tracked_people[person_id].append((frame_num, curr_boxes[curr_idx]))
                    matched_curr_indices.add(curr_idx)
                
                # New people for unmatched boxes
                for curr_idx, box in enumerate(curr_boxes):
                    if curr_idx not in matched_curr_indices:
                        person_id = next_person_id
                        next_person_id += 1
                        curr_box_to_person[curr_idx] = person_id
                        tracked_people[person_id].append((frame_num, box))
            else:
                # First frame
                for curr_idx, box in enumerate(curr_boxes):
                    person_id = next_person_id
                    next_person_id += 1
                    curr_box_to_person[curr_idx] = person_id
                    tracked_people[person_id].append((frame_num, box))
            
            # Update for next iteration
            prev_frame_num = frame_num
            prev_boxes = curr_boxes
            prev_box_to_person = curr_box_to_person
        
        # Find moving sequences (people moving, not still)
        moving_sequences = find_moving_sequences(tracked_people, min_sequence_length=2, max_movement_per_frame=30)
        
        # Estimate image dimensions from all boxes for center calculation
        all_boxes = []
        for frame_num in frame_numbers:
            all_boxes.extend(items[frame_num])
        if len(all_boxes) > 0:
            max_x = max(box[2] for box in all_boxes)  # x2
            max_y = max(box[3] for box in all_boxes)  # y2
            estimated_image_width = int(max_x * 1.1)
            estimated_image_height = int(max_y * 1.1)
        else:
            estimated_image_width = None
            estimated_image_height = None
        
        # Determine which frames to keep (selected frames from moving sequences)
        frames_to_keep = set()
        for person_id, start_frame, end_frame, selected_frame in moving_sequences:
            frames_to_keep.add(selected_frame)
        
        # Also need to handle still sequences - discard them
        still_sequences = find_still_sequences(tracked_people, min_sequence_length=3, max_movement_per_frame=30)
        frames_in_still_sequences = set()
        for person_id, start_frame, end_frame, middle_frame in still_sequences:
            for frame_num in range(start_frame, end_frame + 1):
                frames_in_still_sequences.add(frame_num)
        
        # Remove frames that are in still sequences
        frames_to_keep = frames_to_keep - frames_in_still_sequences
        
        total_kept_frames += len(frames_to_keep)  # Track total kept
        
        # Save path and frame index for kept frames
        for frame_num in frames_to_keep:
            frame_path = os.path.join(path, f"{frame_num}.json")
            frames_to_check.append(frame_path)
        
        # Filter items to only keep good frames
        filtered_items = {frame_num: boxes for frame_num, boxes in items.items() if frame_num in frames_to_keep}
        
        # Count statistics
        frames_removed = original_frame_count - len(frames_to_keep)
        removal_percentage = (frames_removed / original_frame_count * 100) if original_frame_count > 0 else 0
        
        print(f"  Original frames: {original_frame_count}")
        print(f"  Moving sequences found: {len(moving_sequences)}")
        print(f"  Still sequences found: {len(still_sequences)}")
        print(f"  Frames kept (from moving sequences): {len(frames_to_keep)}")
        print(f"  Frames removed: {frames_removed} ({removal_percentage:.1f}%)")
        
        # Update items with filtered data
        old_items_count = len(items)
        items = filtered_items
        new_items_count = len(items)
        print(f"  Items removed: {old_items_count - new_items_count} ({((old_items_count - new_items_count) / old_items_count * 100):.1f}%)")
        print(f"  Items kept: {new_items_count} ({new_items_count / old_items_count * 100:.1f}%)")
    
    # Save all frames to check to a file
    output_file = "paths_still_frames.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(frames_to_check))
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total original frames: {total_original_frames}")
    print(f"Total small boxes removed: {total_small_boxes_removed}")
    print(f"Total frames kept: {total_kept_frames}")
    print(f"Total frames removed: {total_original_frames - total_kept_frames}")
    if total_original_frames > 0:
        kept_percentage = (total_kept_frames / total_original_frames * 100)
        removed_percentage = ((total_original_frames - total_kept_frames) / total_original_frames * 100)
        print(f"Frames kept: {kept_percentage:.1f}%")
        print(f"Frames removed: {removed_percentage:.1f}%")
    print(f"Saved {len(frames_to_check)} frame paths to {output_file}")
    print(f"{'='*60}")
    