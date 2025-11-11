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
from pathlib import Path

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
    if int(os.getenv("VIEW", "0")) > 0:
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

    # Define crop region and size limits (needed for both viewing and processing)
    crop = Crop(x=400, y=70, w=500, h=600)
    crop_x1, crop_y1 = crop.x, crop.y
    crop_x2, crop_y2 = crop.x + crop.w, crop.y + crop.h
    
    # Size filtering parameters
    min_bbox_area = 5000  # Minimum area in pixels (adjust as needed)
    min_bbox_width = 50   # Minimum width
    min_bbox_height = 100 # Minimum height
    
    def bbox_inside_crop_percentage(bbox):
        """Calculate what percentage of bbox is inside the crop region."""
        if len(bbox) < 4:
            return 0.0
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Calculate intersection
        inter_x1 = max(bbox_x1, crop_x1)
        inter_y1 = max(bbox_y1, crop_y1)
        inter_x2 = min(bbox_x2, crop_x2)
        inter_y2 = min(bbox_y2, crop_y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        # Intersection area
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Bbox area
        bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
        
        if bbox_area == 0:
            return 0.0
        
        # Percentage of bbox inside crop
        return (inter_area / bbox_area) * 100.0
    
    def bbox_size_check(bbox):
        """Check if bbox meets size requirements."""
        if len(bbox) < 4:
            return False, 0
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        width = bbox_x2 - bbox_x1
        height = bbox_y2 - bbox_y1
        area = width * height
        return (area >= min_bbox_area and width >= min_bbox_width and height >= min_bbox_height), area
    
    def bbox_overlap_percentage(bbox1, bbox2):
        """Calculate IoU (Intersection over Union) between two bboxes."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        
        # Intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return (inter_area / union_area) * 100.0

    # Process all frames to determine colors (always runs)
    print("\nProcessing all frames to determine red/blue/green status...")
    all_frames_raw = []
    for path in tqdm(path_counts.keys(), desc="Collecting frames"):
        if path not in path_counts: 
            continue
        frames = Sampler.unpack(path, crop="auto")
        for i, frame in enumerate(frames):
            if i < len(path_counts[path]):
                bboxes = path_bboxes.get(path, [[]] * len(path_counts[path]))[i] if path in path_bboxes else []
                all_frames_raw.append({
                    'frame': frame,
                    'path': path,
                    'frame_idx': i,
                    'has_person': path_counts[path][i] > 0,
                    'person_count': path_counts[path][i],
                    'bboxes': bboxes
                })
    
    if not all_frames_raw:
        print("No frames with people found")
        return
    
    # Process bboxes: calculate overlap, size check, and apply filtering
    print("Processing bboxes...")
    all_frames_processed = []
    for frame_data in tqdm(all_frames_raw, desc="Processing bboxes"):
        bbox_info_list = []
        for bbox in frame_data['bboxes']:
            if len(bbox) >= 4:
                overlap_pct = bbox_inside_crop_percentage(bbox)
                meets_size, area = bbox_size_check(bbox)
                conf = bbox[4] if len(bbox) > 4 else 1.0
                
                bbox_info_list.append({
                    'bbox': bbox,
                    'overlap_pct': overlap_pct,
                    'meets_size': meets_size,
                    'area': area,
                    'confidence': conf
                })
        
        # If multiple detections in same frame, pick the one with highest overlap percentage
        if len(bbox_info_list) > 1:
            bbox_info_list.sort(key=lambda x: x['overlap_pct'], reverse=True)
            # Keep only the best one
            bbox_info_list = [bbox_info_list[0]]
        
        frame_data['bboxes'] = bbox_info_list
        all_frames_processed.append(frame_data)
    
    # Temporal grouping: group consecutive detections (2-4 frames in a row)
    def process_temporal_group(group):
        """Process a temporal group: mark 98%+ as green, otherwise pick best."""
        if len(group) < 2:
            # Single frame, no grouping needed
            if group[0]['bboxes']:
                group[0]['bboxes'][0]['temporal_group_best'] = True
            return
        
        # Check for 98%+ detections - these are always kept (green)
        high_overlap_indices = []
        for idx, frame in enumerate(group):
            if frame['bboxes']:
                bbox_info = frame['bboxes'][0]
                if bbox_info['overlap_pct'] >= 98.0 and bbox_info['meets_size']:
                    high_overlap_indices.append(idx)
                    bbox_info['temporal_group_best'] = True  # Always keep 98%+
        
        # If we have 98%+ detections, mark others as filtered
        if high_overlap_indices:
            for idx, frame in enumerate(group):
                if frame['bboxes'] and idx not in high_overlap_indices:
                    frame['bboxes'][0]['temporal_group_best'] = False
        else:
            # No 98%+ detections, pick the best one
            best_idx = max(range(len(group)), 
                          key=lambda idx: group[idx]['bboxes'][0]['overlap_pct'] 
                          if group[idx]['bboxes'] else 0)
            for idx, frame in enumerate(group):
                if frame['bboxes']:
                    frame['bboxes'][0]['temporal_group_best'] = (idx == best_idx)
    
    print("Applying temporal grouping...")
    all_frames_grouped = []
    current_group = []
    
    for i, frame_data in enumerate(all_frames_processed):
        if not frame_data['bboxes']:
            # No detections, end current group if exists
            if current_group:
                process_temporal_group(current_group)
                all_frames_grouped.extend(current_group)
                current_group = []
            all_frames_grouped.append(frame_data)
            continue
        
        # Has detection - add to current group
        if not current_group:
            # Start new group
            current_group.append(frame_data)
        else:
            # Check if group is getting too large (max 4)
            if len(current_group) >= 4:
                # Process current group and start new one
                process_temporal_group(current_group)
                all_frames_grouped.extend(current_group)
                current_group = [frame_data]
            else:
                # Continue group (consecutive detection)
                current_group.append(frame_data)
    
    # Handle remaining group
    if current_group:
        process_temporal_group(current_group)
        all_frames_grouped.extend(current_group)
    
    # Determine final color for each bbox
    print("Determining final colors...")
    all_frames_with_people = []
    for frame_data in all_frames_grouped:
        if not frame_data['bboxes']:
            all_frames_with_people.append(frame_data)
            continue
        
        bbox_info = frame_data['bboxes'][0]
        overlap_pct = bbox_info['overlap_pct']
        meets_size = bbox_info['meets_size']
        is_temporal_best = bbox_info.get('temporal_group_best', True)  # Default True if not in a group
        
        # Determine color and reason
        # 98%+ detections are always green regardless of temporal grouping
        if overlap_pct >= 98.0 and meets_size:
            color = 'green'  # Always green if 98%+ and meets size
            reason = 'high_overlap'
        elif not is_temporal_best:
            # Not the best in temporal group - mark as blue
            color = 'blue'
            reason = 'temporal_filtered'
        elif not meets_size:
            color = 'blue'  # Too small
            reason = 'size_filtered'
        elif overlap_pct >= 80.0:
            color = 'green'  # 80%+ inside crop
            reason = 'inside_crop'
        else:
            color = 'red'  # <80% inside crop
            reason = 'outside_crop'
        
        bbox_info['color'] = color
        bbox_info['reason'] = reason
        all_frames_with_people.append(frame_data)
    
    # Calculate and print statistics
    total_bboxes = sum(1 for item in all_frames_with_people if item['bboxes'])
    green_bboxes = sum(1 for item in all_frames_with_people 
                      for bbox_info in item['bboxes'] 
                      if bbox_info.get('color') == 'green')
    red_bboxes = sum(1 for item in all_frames_with_people 
                    for bbox_info in item['bboxes'] 
                    if bbox_info.get('color') == 'red')
    blue_bboxes = sum(1 for item in all_frames_with_people 
                     for bbox_info in item['bboxes'] 
                     if bbox_info.get('color') == 'blue')
    
    print(f"\n{'='*60}")
    print(f"Bbox Statistics (Crop: x={crop.x}, y={crop.y}, w={crop.w}, h={crop.h}):")
    print(f"  Total bboxes: {total_bboxes}")
    print(f"  Green (80%+ inside crop or 98%+ with size): {green_bboxes} ({green_bboxes/total_bboxes*100:.1f}%)" if total_bboxes > 0 else "  Green: 0")
    print(f"  Red (<80% inside crop): {red_bboxes} ({red_bboxes/total_bboxes*100:.1f}%)" if total_bboxes > 0 else "  Red: 0")
    print(f"  Blue (filtered by size or temporal): {blue_bboxes} ({blue_bboxes/total_bboxes*100:.1f}%)" if total_bboxes > 0 else "  Blue: 0")
    print(f"  Size filter: min_area={min_bbox_area}, min_width={min_bbox_width}, min_height={min_bbox_height}")
    print(f"{'='*60}\n")

    # Only show viewer if VIEW is enabled
    if os.getenv("VIEW", "0") != "0":
        # Collect all frames with bbox info
        all_frames_raw = []
        for path in tqdm(list(path_counts.keys())[10:], desc="Collecting frames"):
            print(path)
            if path not in path_counts: 
                continue
            frames = Sampler.unpack(path, crop="auto")
            for i, frame in enumerate(frames):
                if i < len(path_counts[path]):
                    bboxes = path_bboxes.get(path, [[]] * len(path_counts[path]))[i] if path in path_bboxes else []
                    all_frames_raw.append({
                        'frame': frame,
                        'path': path,
                        'frame_idx': i,
                        'has_person': path_counts[path][i] > 0,
                        'person_count': path_counts[path][i],
                        'bboxes': bboxes
                    })
        
        if not all_frames_raw:
            print("No frames with people found")
            return
        
        # Process bboxes: calculate overlap, size check, and apply filtering
        all_frames_processed = []
        for frame_data in all_frames_raw:
            bbox_info_list = []
            for bbox in frame_data['bboxes']:
                if len(bbox) >= 4:
                    overlap_pct = bbox_inside_crop_percentage(bbox)
                    meets_size, area = bbox_size_check(bbox)
                    conf = bbox[4] if len(bbox) > 4 else 1.0
                    
                    bbox_info_list.append({
                        'bbox': bbox,
                        'overlap_pct': overlap_pct,
                        'meets_size': meets_size,
                        'area': area,
                        'confidence': conf
                    })
            
            # If multiple detections in same frame, pick the one with highest overlap percentage
            if len(bbox_info_list) > 1:
                bbox_info_list.sort(key=lambda x: x['overlap_pct'], reverse=True)
                # Keep only the best one
                bbox_info_list = [bbox_info_list[0]]
            
            frame_data['bboxes'] = bbox_info_list
            all_frames_processed.append(frame_data)
        
        # Temporal grouping: group consecutive detections (2-4 frames in a row)
        
        def process_temporal_group(group):
            """Process a temporal group: mark 98%+ as green, otherwise pick best."""
            if len(group) < 2:
                # Single frame, no grouping needed
                if group[0]['bboxes']:
                    group[0]['bboxes'][0]['temporal_group_best'] = True
                return
            
            # Check for 98%+ detections - these are always kept (green)
            high_overlap_indices = []
            for idx, frame in enumerate(group):
                if frame['bboxes']:
                    bbox_info = frame['bboxes'][0]
                    if bbox_info['overlap_pct'] >= 98.0 and bbox_info['meets_size']:
                        high_overlap_indices.append(idx)
                        bbox_info['temporal_group_best'] = True  # Always keep 98%+
            
            # If we have 98%+ detections, mark others as filtered
            if high_overlap_indices:
                for idx, frame in enumerate(group):
                    if frame['bboxes'] and idx not in high_overlap_indices:
                        frame['bboxes'][0]['temporal_group_best'] = False
            else:
                # No 98%+ detections, pick the best one
                best_idx = max(range(len(group)), 
                              key=lambda idx: group[idx]['bboxes'][0]['overlap_pct'] 
                              if group[idx]['bboxes'] else 0)
                for idx, frame in enumerate(group):
                    if frame['bboxes']:
                        frame['bboxes'][0]['temporal_group_best'] = (idx == best_idx)
        
        all_frames_grouped = []
        current_group = []
        
        for i, frame_data in enumerate(all_frames_processed):
            if not frame_data['bboxes']:
                # No detections, end current group if exists
                if current_group:
                    # Process group: mark 98%+ as green, otherwise pick best
                    process_temporal_group(current_group)
                    all_frames_grouped.extend(current_group)
                    current_group = []
                all_frames_grouped.append(frame_data)
                continue
            
            # Has detection - add to current group
            if not current_group:
                # Start new group
                current_group.append(frame_data)
            else:
                # Check if group is getting too large (max 4)
                if len(current_group) >= 4:
                    # Process current group and start new one
                    process_temporal_group(current_group)
                    all_frames_grouped.extend(current_group)
                    current_group = [frame_data]
                else:
                    # Continue group (consecutive detection)
                    current_group.append(frame_data)
        
        # Handle remaining group
        if current_group:
            process_temporal_group(current_group)
            all_frames_grouped.extend(current_group)
        
        # Determine final color for each bbox
        all_frames_with_people = []
        for frame_data in all_frames_grouped:
            if not frame_data['bboxes']:
                all_frames_with_people.append(frame_data)
                continue
            
            bbox_info = frame_data['bboxes'][0]
            overlap_pct = bbox_info['overlap_pct']
            meets_size = bbox_info['meets_size']
            is_temporal_best = bbox_info.get('temporal_group_best', True)  # Default True if not in a group
            
            # Determine color and reason
            # 98%+ detections are always green regardless of temporal grouping
            if overlap_pct >= 98.0 and meets_size:
                color = 'green'  # Always green if 98%+ and meets size
                reason = 'high_overlap'
            elif not is_temporal_best:
                # Not the best in temporal group - mark as blue
                color = 'blue'
                reason = 'temporal_filtered'
            elif not meets_size:
                color = 'blue'  # Too small
                reason = 'size_filtered'
            elif overlap_pct >= 80.0:
                color = 'green'  # 80%+ inside crop
                reason = 'inside_crop'
            else:
                color = 'red'  # <80% inside crop
                reason = 'outside_crop'
            
            bbox_info['color'] = color
            bbox_info['reason'] = reason
            all_frames_with_people.append(frame_data)
        
        # Calculate statistics
        total_bboxes = sum(1 for item in all_frames_with_people if item['bboxes'])
        green_bboxes = sum(1 for item in all_frames_with_people 
                          for bbox_info in item['bboxes'] 
                          if bbox_info.get('color') == 'green')
        red_bboxes = sum(1 for item in all_frames_with_people 
                        for bbox_info in item['bboxes'] 
                        if bbox_info.get('color') == 'red')
        blue_bboxes = sum(1 for item in all_frames_with_people 
                         for bbox_info in item['bboxes'] 
                         if bbox_info.get('color') == 'blue')
        
        print(f"\nTotal frames: {len(all_frames_with_people)}")
        print(f"Frames with people: {sum(1 for f in all_frames_with_people if f['has_person'])}")
        print(f"\nBbox Statistics (Crop: x={crop.x}, y={crop.y}, w={crop.w}, h={crop.h}):")
        print(f"  Total bboxes: {total_bboxes}")
        print(f"  Green (80%+ inside crop or 98%+ with size): {green_bboxes} ({green_bboxes/total_bboxes*100:.1f}%)" if total_bboxes > 0 else "  Green: 0")
        print(f"  Red (<80% inside crop): {red_bboxes} ({red_bboxes/total_bboxes*100:.1f}%)" if total_bboxes > 0 else "  Red: 0")
        print(f"  Blue (filtered by size): {blue_bboxes} ({blue_bboxes/total_bboxes*100:.1f}%)" if total_bboxes > 0 else "  Blue: 0")
        print(f"\nSize filter: min_area={min_bbox_area}, min_width={min_bbox_width}, min_height={min_bbox_height}")
        print("\nControls: 'n' = next, 'b' = previous, 'q' = quit")
        
        cur_idx = 0
        window_name = "Human Detection Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while 0 <= cur_idx < len(all_frames_with_people):
            item = all_frames_with_people[cur_idx]
            img = item['frame'].copy()
            h, w = img.shape[:2]
            
            # Draw crop region
            cv2.rectangle(img, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 255, 0), 2)  # Yellow crop outline
            cv2.putText(img, "Crop Region", (crop_x1, crop_y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Draw bounding boxes with color based on filtering
            for bbox_info in item['bboxes']:
                bbox = bbox_info['bbox']
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    conf = bbox_info['confidence']
                    overlap_pct = bbox_info['overlap_pct']
                    color_name = bbox_info.get('color', 'green')
                    
                    # Map color names to BGR
                    if color_name == 'green':
                        color = (0, 255, 0)
                    elif color_name == 'red':
                        color = (0, 0, 255)
                    elif color_name == 'blue':
                        color = (255, 0, 0)
                    else:
                        color = (255, 255, 255)
                    
                    # Draw rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    # Draw confidence and overlap percentage
                    label = f"{conf:.2f} ({overlap_pct:.1f}%)"
                    cv2.putText(img, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            
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
            instructions = "Press 'n' for next | 'b' for previous | 'L' for next human | 'q' to quit"
            cv2.putText(img, instructions, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            
            cv2.imshow(window_name, img)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key in (ord('n'), ord('N')):  # next
                cur_idx += 1
            elif key in (ord('b'), ord('B')):  # previous
                cur_idx = max(0, cur_idx - 1)
            elif key in (ord('l'), ord('L')):  # next frame with human
                # Find next frame with human detected
                found = False
                for next_idx in range(cur_idx + 1, len(all_frames_with_people)):
                    if all_frames_with_people[next_idx]['has_person']:
                        cur_idx = next_idx
                        found = True
                        break
                if not found:
                    print("No more frames with humans detected")
            elif key in (ord('q'), 27):  # quit
                break
        
        cv2.destroyAllWindows()

    # Process green frames and call OpenAI (always runs)
    print("Processing green frames with OpenAI...")
    gpt_trainer = GPTTrainer()
    prompt = """
    You need to analyze the image and determine if there is a person, if they are walking left or right, what item they are carrying, how many of each item they are carrying, the estimated weight and co2 emission from the item production.
    If there are multiple people in the image, focus on the person in the center of the image.
    """

    @dataclass
    class GPTResult:
        person: bool
        person_carrying_item: bool
        person_walking_direction: str #left or right
        item_description: [str]
        item_count: [int]
        estimated_weight_of_item_kg: [float]
        estimated_co2_emission_from_item_production_kg: [float] 

    # Collect all green frames
    green_frames = []
    for item in all_frames_with_people:
        if item['bboxes']:
            bbox_info = item['bboxes'][0]
            if bbox_info.get('color') == 'green':
                green_frames.append({
                    'path': item['path'],
                    'frame': item['frame'],
                    'frame_idx': item['frame_idx'],
                    'bbox': bbox_info['bbox']
                })
    
    print(f"Total green frames to process: {len(green_frames)}")
    
    if not green_frames:
        print("No green frames to process")
        return
    
    # Process in batches of 50
    batch_size = 50
    total_batches = (len(green_frames) + batch_size - 1) // batch_size
    
    print(f"Processing {total_batches} batches of up to {batch_size} frames each...")
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(green_frames))
        batch = green_frames[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch)} frames)...")
        
        # Prepare images for OpenAI (each as a list with single image)
        image_batches = [[item['frame']] for item in batch]
        
        # Call OpenAI
        print("Calling OpenAI...")
        gpt_results = gpt_trainer._openai_call(
            model=gpt_trainer.default_model,
            prompt=prompt,
            image_paths=image_batches,
            output_schema=GPTResult
        )
        
        # Save results
        print("Saving results...")
        for item, result in zip(batch, gpt_results):
            path = item['path']
            frame_idx = item['frame_idx']
            
            # Create identifier: path_name_frame_number
            path_name = Path(path).stem  # Get filename without extension
            identifier = f"{path_name}_frame_{frame_idx}"
            
            # Save JSON result
            ResultRegistry.POST(
                identifier,
                gpt_trainer.default_model,
                "person-analysis",
                result.__dict__
            )
        
        print(f"Saved {len(batch)} results to ResultRegistry")
    
    print(f"\nCompleted! Processed {len(green_frames)} green frames total.")


if __name__ == "__main__":
    run()