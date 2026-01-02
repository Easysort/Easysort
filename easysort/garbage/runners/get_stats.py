from easysort.registry import ResultRegistry, DataRegistry
from easysort.helpers import RESULTS_REGISTRY_PATH, DATA_REGISTRY_PATH
from pathlib import Path
import json
import os
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Configuration
GPT_MODEL = "gpt-5-mini-2025-08-07"
GPT_PROJECT = "gpt-analysis"
STILL_FRAMES_FILE = "paths_still_frames.txt"

# Multipliers to apply to all values
ITEMS_MULTIPLIER = 2.0
WEIGHT_MULTIPLIER = 0.8
CO2_MULTIPLIER = 0.4

# Category sort order
CATEGORY_ORDER = [
    "køkkenting",
    "fritid og have",
    "møbler",
    "boligting",
    "lejetøj",
    "andet"
]

def sort_categories(categories):
    """Sort categories according to CATEGORY_ORDER, with others at the end."""
    def sort_key(category):
        try:
            return CATEGORY_ORDER.index(category)
        except ValueError:
            return len(CATEGORY_ORDER)  # Put unknown categories at the end
    
    return sorted(categories, key=sort_key)

def get_time_interval(hour: int) -> str:
    """Convert hour to 3-hour interval string."""
    interval_start = (hour // 3) * 3
    return f"{interval_start:02d}:00-{interval_start+3:02d}:00"

def extract_video_path_and_frame(result_path: str):
    """
    Extract the original video path and frame index from a YOLO result path.
    
    Result path format: .../results/argo/Device/2025/11/16/07/072703/yolov8s.pt/argo-people/108.json
    Video path format: .../data/argo/Device/2025/11/16/07/072703.mp4
    """
    path_obj = Path(result_path)
    parts = path_obj.parts
    
    try:
        argo_idx = parts.index('argo')
        # Path structure: .../argo/Device/2025/11/13/20/204901/yolov8s.pt/argo-people/108.json
        # We need: .../argo/Device/2025/11/13/20/204901.mp4
        
        # Get everything up to the timestamp (which is 4 parts from the end: before yolov8s.pt/argo-people/filename)
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

def get_gpt_result_path(video_path: str, frame_idx: int):
    """
    Construct the GPT result file path from video path and frame index.
    Uses the same logic as ResultRegistry.POST.
    """
    result_path = Path(video_path)
    if "/mnt/" in str(result_path) or DATA_REGISTRY_PATH in str(result_path):
        result_path = Path(str(result_path).replace(DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH))
    
    result_file = Path(RESULTS_REGISTRY_PATH) / str(result_path.with_suffix("")) / GPT_MODEL / GPT_PROJECT / f"{frame_idx}.json"
    return result_file

def parse_gpt_result_path(gpt_result_path: str):
    """
    Parse GPT result file path to extract device, date, time, and frame info.
    Path format: .../results/argo/Device/Year/Month/Day/Hour/Timestamp/{gpt_model}/{gpt_project}/{frame_idx}.json
    """
    path_obj = Path(gpt_result_path)
    parts = path_obj.parts
    
    try:
        argo_idx = parts.index('argo')
        # parts[argo_idx+1] = Device
        # parts[argo_idx+2] = Year
        # parts[argo_idx+3] = Month
        # parts[argo_idx+4] = Day
        # parts[argo_idx+5] = Hour
        # parts[argo_idx+6] = Timestamp
        # parts[-1] = {frame_idx}.json
        
        device = parts[argo_idx + 1]
        year = int(parts[argo_idx + 2])
        month = int(parts[argo_idx + 3])
        day = int(parts[argo_idx + 4])
        hour = int(parts[argo_idx + 5])
        timestamp = parts[argo_idx + 6]
        frame_idx = int(path_obj.stem)
        
        date = datetime(year, month, day)
        time_interval = get_time_interval(hour)
        
        return {
            'device': device,
            'date': date,
            'date_str': date.strftime('%Y-%m-%d'),
            'hour': hour,
            'time_interval': time_interval,
            'timestamp': timestamp,
            'frame_idx': frame_idx
        }
    except (ValueError, IndexError) as e:
        return None

def load_all_results():
    """Load all GPT analysis results from ResultRegistry using paths_still_frames.txt."""
    # Load paths from still_frames_file
    if not os.path.exists(STILL_FRAMES_FILE):
        print(f"Error: {STILL_FRAMES_FILE} not found. Run argo_people_items.py first.")
        return []
    
    print(f"Loading frame paths from {STILL_FRAMES_FILE}...")
    with open(STILL_FRAMES_FILE, "r") as f:
        yolo_result_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(yolo_result_paths)} frame paths in {STILL_FRAMES_FILE}")
    
    # Extract video paths and frame indices, then construct GPT result paths
    all_results = []
    errors = 0
    skipped = 0
    not_found = 0
    
    for yolo_result_path in tqdm(yolo_result_paths, desc="Processing paths", unit=" paths"):
        try:
            # Extract video path and frame index from YOLO result path
            video_path, frame_idx = extract_video_path_and_frame(yolo_result_path)
            
            # Construct GPT result path
            gpt_result_path = get_gpt_result_path(video_path, frame_idx)
            
            # Check if GPT result exists
            if not gpt_result_path.exists():
                not_found += 1
                continue
            
            # Parse path info from GPT result path
            path_info = parse_gpt_result_path(str(gpt_result_path))
            if path_info is None:
                skipped += 1
                tqdm.write(f"Warning: Could not parse path info from {gpt_result_path}")
                continue
            
            # Load JSON data
            with open(gpt_result_path, 'r') as f:
                data = json.load(f)
            
            all_results.append({
                'path_info': path_info,
                'data': data
            })
            
        except ValueError as e:
            # Invalid path format
            skipped += 1
            continue
        except Exception as e:
            errors += 1
            tqdm.write(f"Error processing {yolo_result_path}: {e}")
            continue
    
    print(f"\nLoaded: {len(all_results)} GPT results")
    if not_found > 0:
        print(f"  Not found: {not_found} (GPT results not yet generated)")
    if errors > 0:
        print(f"  Errors: {errors}")
    if skipped > 0:
        print(f"  Skipped: {skipped} (invalid paths)")
    
    return all_results

def calculate_statistics(all_results):
    """Calculate all statistics from results (all directions, not just left)."""
    stats = {
        'by_device': defaultdict(lambda: {
            'total_objects': 0,
            'total_weight': 0.0,
            'total_co2': 0.0,
            'items_by_category': defaultdict(int),  # category -> count
            'weight_by_category': defaultdict(float),  # category -> total weight
            'co2_by_category': defaultdict(float),  # category -> total CO2
            'by_date': defaultdict(int),  # date -> total objects
            'by_time_interval': defaultdict(int)  # time_interval -> total objects
        })
    }
    
    for result in tqdm(all_results, desc="Calculating statistics", unit=" results"):
        data = result['data']
        path_info = result['path_info']
        
        # Check if person exists
        if not data.get('person', False):
            continue
        
        device = path_info['device']
        date_str = path_info['date_str']
        time_interval = path_info['time_interval']
        
        # Count items for ALL directions (not just left)
        item_categories = data.get('item_category', [])
        item_counts = data.get('item_count', [])
        item_weights = data.get('estimated_weight_of_item_kg', [])
        item_co2s = data.get('estimated_co2_emission_from_item_production_kg', [])
        
        # Process items
        for i, category in enumerate(item_categories):
            if category:
                count = item_counts[i] if i < len(item_counts) else 1
                weight = item_weights[i] if i < len(item_weights) else 0.0
                co2 = item_co2s[i] if i < len(item_co2s) else 0.0
                
                # Update totals
                stats['by_device'][device]['total_objects'] += count
                stats['by_device'][device]['total_weight'] += weight * count
                stats['by_device'][device]['total_co2'] += co2 * count
                
                # Update by category
                stats['by_device'][device]['items_by_category'][category] += count
                stats['by_device'][device]['weight_by_category'][category] += weight * count
                stats['by_device'][device]['co2_by_category'][category] += co2 * count
                
                # Track total objects per date and time interval
                stats['by_device'][device]['by_date'][date_str] += count
                stats['by_device'][device]['by_time_interval'][time_interval] += count
    
    return stats

def print_statistics(stats):
    """Print statistics organized by device."""
    print("\n" + "="*80)
    print("GPT ANALYSIS STATISTICS".center(80))
    print("="*80)
    print(f"\nMultipliers Applied:")
    print(f"  Items: {ITEMS_MULTIPLIER}")
    print(f"  Weight: {WEIGHT_MULTIPLIER}")
    print(f"  CO₂: {CO2_MULTIPLIER}")
    
    # For each device
    for device in sorted(stats['by_device'].keys()):
        device_stats = stats['by_device'][device]
        
        print(f"\n{'='*80}")
        print(f"DEVICE: {device}".center(80))
        print("="*80)
        
        # Apply multipliers
        total_objects = device_stats['total_objects'] * ITEMS_MULTIPLIER
        total_weight = device_stats['total_weight'] * WEIGHT_MULTIPLIER
        total_co2 = device_stats['total_co2'] * CO2_MULTIPLIER
        
        # 1. Total number of objects, total weight, total CO2
        print(f"\nTotal Objects: {total_objects:,.0f}")
        print(f"Total Weight: {total_weight:,.2f} kg")
        print(f"Total CO₂: {total_co2:,.2f} kg")
        
        # 2. Weight and number of objects for each category
        if device_stats['items_by_category']:
            print(f"\nBy Category:")
            print(f"{'Category':<30} {'Objects':>12} {'Weight (kg)':>15}")
            print("-" * 80)
            for category in sort_categories(device_stats['items_by_category'].keys()):
                items = device_stats['items_by_category'][category] * ITEMS_MULTIPLIER
                weight = device_stats['weight_by_category'][category] * WEIGHT_MULTIPLIER
                print(f"{category:<30} {items:>12,.0f} {weight:>15,.2f}")
        
        # 3. CO2 * 0.8, 1, and 1.5 (total, not by category)
        print(f"\nCO₂ Multipliers (Total):")
        print(f"  CO₂ × 0.8: {total_co2 * 0.8:,.2f} kg")
        print(f"  CO₂ × 1.0: {total_co2 * 1.0:,.2f} kg")
        print(f"  CO₂ × 1.5: {total_co2 * 1.5:,.2f} kg")
        
        # 4. Total objects per day
        print(f"\nTotal Objects per Day:")
        for date_str in sorted(device_stats['by_date'].keys()):
            total_objects_day = device_stats['by_date'][date_str] * ITEMS_MULTIPLIER
            print(f"  {date_str}: {total_objects_day:,.0f} objects")
        
        # 5. Total objects per time interval
        print(f"\nTotal Objects per Time Interval:")
        for time_interval in sorted(device_stats['by_time_interval'].keys()):
            total_objects_interval = device_stats['by_time_interval'][time_interval] * ITEMS_MULTIPLIER
            print(f"  {time_interval}: {total_objects_interval:,.0f} objects")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("="*80)
    print("GPT ANALYSIS STATISTICS GENERATOR".center(80))
    print("="*80)
    print("\nStep 1: Loading all GPT analysis results...")
    all_results = load_all_results()
    
    if not all_results:
        print("No results found!")
        exit(1)
    
    print(f"\nStep 2: Calculating statistics from {len(all_results)} results...")
    stats = calculate_statistics(all_results)
    
    print("\nStep 3: Generating report...")
    print_statistics(stats)
