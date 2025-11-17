import os
import tempfile
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
from easysort.sampler import Sampler, Crop
from easysort.registry import DataRegistry
from tqdm import tqdm

# -----------------------------
# 1) Run YOLO tracking + filter to right->left walkers (not standing)
#    Returns:
#       - track_frames: dict[track_id] -> list of (frame_idx, xyxy)
#       - groups: list[list[int]] frame indices per qualifying track (what you asked for)
# -----------------------------
def get_rtl_tracks_and_indices(
    frames,                       # list[np.ndarray] BGR, chronological @ ~1 FPS
    model_path="yolov8s.pt",      # or your weights
    tracker_cfg="bytetrack.yaml", # or "botsort.yaml"
    fps=1,
    min_track_len=3,              # need N hits before classifying direction
    min_speed_px_s=25,            # reject standers (tune per scene)
    direction_agree=0.7,          # % of steps with dx<0 to count as R->L
    conf=0.3,
    iou=0.5,
    person_cls_id=0               # COCO person class id; change for custom datasets
):
    assert len(frames) > 0, "No frames provided"
    h, w = frames[0].shape[:2]

    # Write to temp mp4 for stable tracker ingestion
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmpdir = tempfile.mkdtemp(prefix="rtl_")
    video_path = os.path.join(tmpdir, "in.mp4")
    vw = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        vw.write(f)
    vw.release()

    model = YOLO(model_path)
    results = model.track(
        source=video_path,
        tracker=tracker_cfg,
        stream=True,
        persist=True,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    # Per-track short history for velocity
    # tid -> deque of (t_sec, cx, cy, frame_idx, xyxy)
    hist = defaultdict(lambda: deque(maxlen=60))
    # Also keep per-frame bboxes for later viewing
    # tid -> list of (frame_idx, xyxy)
    track_frames = defaultdict(list)

    frame_idx = -1
    for res in results:
        frame_idx += 1
        if res.boxes is None or len(res.boxes) == 0:
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id
        ids = ids.cpu().numpy().astype(int) if ids is not None else np.array([-1]*len(xyxy))
        t_sec = frame_idx / float(fps)

        for b, c, tid in zip(xyxy, cls, ids):
            if tid < 0 or c != person_cls_id:
                continue
            x1, y1, x2, y2 = b
            cx = 0.5*(x1 + x2)
            cy = 0.5*(y1 + y2)
            hist[tid].append((t_sec, cx, cy, frame_idx, b.astype(int)))
            track_frames[tid].append((frame_idx, b.astype(int)))

    def track_stats(points):
        if len(points) < 2:
            return None
        vxs, speeds, neg_steps = [], [], 0
        for (t0, x0, y0, *_), (t1, x1, y1, *_) in zip(points[:-1], points[1:]):
            dt = max(1e-6, t1 - t0)
            vx = (x1 - x0) / dt
            spd = ((x1-x0)**2 + (y1-y0)**2) ** 0.5 / dt
            vxs.append(vx)
            speeds.append(spd)
            if vx < 0:
                neg_steps += 1
        if not vxs:
            return None
        return {
            "avg_vx": float(np.mean(vxs)),
            "avg_speed": float(np.mean(speeds)),
            "agree": neg_steps / len(vxs)
        }

    # Filter to right->left walkers
    qualifying_tids = []
    for tid, points in hist.items():
        if len(points) < min_track_len:
            continue
        s = track_stats(list(points))
        if s and s["avg_vx"] < 0 and s["avg_speed"] >= min_speed_px_s and s["agree"] >= direction_agree:
            qualifying_tids.append(tid)

    # Build list of lists of frame indices, grouped by track
    groups = []
    for tid in qualifying_tids:
        frame_indices = [fidx for (fidx, _) in track_frames[tid]]
        # Ensure unique & sorted (tracking may output duplicates rarely)
        frame_indices = sorted(set(frame_indices))
        groups.append(frame_indices)

    return track_frames, groups, qualifying_tids


# -----------------------------
# 2) OpenCV viewer to play each qualifying track’s frames
#    Controls:
#       SPACE : pause/resume
#       n     : next track
#       b     : previous track
#       r     : replay current track
#       q/ESC : quit
# -----------------------------
def view_tracks(
    frames_dict,              # dict[path] -> list of frames, OR list of frames (backward compat)
    track_frames,             # dict[tid] -> list of (frame_idx, xyxy)
    track_order=None,         # list of tids; default sorts by first appearance
    track_paths=None,          # dict[tid] -> path (required if frames_dict is a dict)
    window_name="Track Viewer",
    play_fps=6,               # playback speed in the viewer (not your sampling FPS)
    draw_thickness=2
):
    # Handle backward compatibility: if frames_dict is a list, treat as single video
    if isinstance(frames_dict, list):
        frames_dict = {"_default": frames_dict}
        track_paths = {tid: "_default" for tid in track_frames.keys()}
    
    assert track_paths is not None, "track_paths required when frames_dict is a dict"
    
    # default order by first frame of each track
    if track_order is None:
        track_order = sorted(
            track_frames.keys(),
            key=lambda tid: min([fi for (fi, _) in track_frames[tid]]) if track_frames[tid] else 1e12
        )
    
    if not track_order:
        print("No tracks to display")
        return

    delay = max(1, int(1000 / play_fps))
    cur = 0
    paused = False
    auto_loop = True  # Auto-loop each track

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while 0 <= cur < len(track_order):
        tid = track_order[cur]
        seq = sorted(track_frames[tid], key=lambda x: x[0])  # by frame index
        
        # Get the correct frames for this track
        path = track_paths[tid]
        frames = frames_dict[path]

        i = 0
        loop_count = 0
        while True:  # Loop until user presses 'n' to continue
            frame_idx, box = seq[i]
            if frame_idx >= len(frames):
                print(f"Warning: frame_idx {frame_idx} out of range for track {tid}, skipping")
                i = (i + 1) % len(seq)  # Wrap around
                continue
                
            img = frames[frame_idx].copy()
            x1, y1, x2, y2 = map(int, box.tolist() if hasattr(box, "tolist") else box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), draw_thickness)
            
            # Large, prominent track ID display
            h, w = img.shape[:2]
            track_text = f"TRACK ID: {tid}"
            frame_info = f"Frame {i+1}/{len(seq)} | Loop #{loop_count + 1}"
            path_display = f"Path: {path.split('/')[-1]}"  # Show just filename
            
            # Draw track ID - large and prominent
            cv2.putText(img, track_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(img, track_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)  # Black outline
            
            # Frame info
            cv2.putText(img, frame_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, path_display, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Instructions at bottom
            instructions = "Press 'n' for next track | 'b' for previous | 'q' to quit | SPACE to pause"
            cv2.putText(img, instructions, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow(window_name, img)
            
            # Wait for key - if paused, wait indefinitely, otherwise wait for delay
            wait_time = 0 if paused else delay
            key = cv2.waitKey(wait_time) & 0xFF

            # Check if a key was actually pressed (not timeout)
            if key != 255:  # 255 means no key was pressed (timeout)
                if key == ord(' '):   # SPACE toggles pause
                    paused = not paused
                    continue
                elif key in (ord('n'), ord('N')):  # next track
                    cur += 1
                    break
                elif key in (ord('b'), ord('B')):  # previous track
                    cur = max(0, cur - 1)
                    break
                elif key in (ord('r'), ord('R')):  # replay current track (restart loop)
                    i = 0
                    loop_count = 0
                    continue
                elif key in (ord('q'), 27):   # 'q' or ESC
                    cv2.destroyWindow(window_name)
                    return
            
            # If not paused and no key pressed, advance frame
            if not paused:
                i += 1
                # Auto-loop: when we reach the end, restart from beginning
                if i >= len(seq):
                    i = 0
                    loop_count += 1

    cv2.destroyWindow(window_name)


def view_detected_people(
    frames_dict,              # dict[path] -> list of frames
    detections_dict,          # dict[path] -> list of YOLO results (one per frame)
    window_name="People Detector",
    play_fps=6,
    draw_thickness=2
):
    """View all frames with detected people, one by one."""
    # Flatten all frames and detections into a single list with path info
    all_frames_with_detections = []
    total_people = 0
    
    for path, frames in frames_dict.items():
        detections = detections_dict[path]
        for frame_idx, (frame, result) in enumerate(zip(frames, detections)):
            if result.boxes is not None:
                person_count = int((result.boxes.cls.cpu().numpy() == 0).sum())
                if person_count > 0:
                    all_frames_with_detections.append({
                        'frame': frame,
                        'path': path,
                        'frame_idx': frame_idx,
                        'result': result,
                        'person_count': person_count
                    })
                    total_people += person_count
    
    if not all_frames_with_detections:
        print("No people detected in any frames")
        return
    
    print(f"\nTotal frames with people: {len(all_frames_with_detections)}")
    print(f"Total people detected: {total_people}")
    print("\nControls: 'n' = next, 'b' = previous, 'q' = quit, SPACE = pause")
    
    delay = max(1, int(1000 / play_fps))
    cur_idx = 0
    paused = False
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while 0 <= cur_idx < len(all_frames_with_detections):
        item = all_frames_with_detections[cur_idx]
        img = item['frame'].copy()
        result = item['result']
        h, w = img.shape[:2]
        
        # Draw all bounding boxes
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                if int(cls) == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), draw_thickness)
                    # Draw confidence
                    cv2.putText(img, f"{conf:.2f}", (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Display info
        frame_num = f"Frame {cur_idx + 1}/{len(all_frames_with_detections)}"
        people_count = f"People in frame: {item['person_count']}"
        total_text = f"Total people detected: {total_people}"
        path_text = f"Path: {item['path'].split('/')[-1]}"
        frame_idx_text = f"Frame index: {item['frame_idx']}"
        
        # Large text for frame number
        cv2.putText(img, frame_num, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, frame_num, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)  # Outline
        
        # Other info
        cv2.putText(img, people_count, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, total_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, path_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, frame_idx_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Instructions
        instructions = "Press 'n' for next | 'b' for previous | 'q' to quit | SPACE to pause"
        cv2.putText(img, instructions, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, img)
        
        wait_time = 0 if paused else delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key != 255:
            if key == ord(' '):  # SPACE toggles pause
                paused = not paused
                continue
            elif key in (ord('n'), ord('N')):  # next
                cur_idx += 1
            elif key in (ord('b'), ord('B')):  # previous
                cur_idx = max(0, cur_idx - 1)
            elif key in (ord('q'), 27):  # quit
                break
        elif not paused:
            # Auto-advance if not paused
            cur_idx += 1
            if cur_idx >= len(all_frames_with_detections):
                cur_idx = 0  # Loop back to start
    
    cv2.destroyWindow(window_name)


# -----------------------------
# Example usage with batch processing
# -----------------------------
if __name__ == "__main__":
    from easysort.gpt_trainer import YoloTrainer
    
    all_paths = DataRegistry.LIST("argo")
    batch_size =5
    yolo_trainer = YoloTrainer()
    
    # Process in batches
    for batch_start in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[batch_start:batch_start + batch_size]
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1} ({len(batch_paths)} paths)")
        print(f"{'='*60}")
        
        frames_dict = {}  # Store frames per path
        detections_dict = {}  # Store YOLO results per path
        
        for path in tqdm(batch_paths, desc=f"Processing batch {batch_start // batch_size + 1}"):
            frames = Sampler.unpack(path, crop=Crop(x=640, y=0, w=260, h=480))
            frames_dict[path] = frames
            
            # Run YOLO detection (not tracking) on all frames
            print(f"Detecting people in {len(frames)} frames...")
            results = yolo_trainer.model(frames, verbose=False)  # Simple detection
            detections_dict[path] = results
        
        print(f"\nfor the following paths: {chr(10).join(batch_paths)}")
        
        # Show all detected people
        print("\nViewing detected people... (Press 'q' or ESC when done)")
        view_detected_people(
            frames_dict,
            detections_dict,
            play_fps=6
        )
        
        # Wait for user to continue
        print("\nPress any key to continue to next batch...")
        cv2.waitKey(0)
    
    print("\nFinished processing all paths!")
