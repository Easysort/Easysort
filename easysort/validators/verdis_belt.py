"""Verdis Belt Validator - validates AI predictions for belt images."""
from easysort.registry import RegistryBase, RegistryConnector
from easysort.helpers import REGISTRY_LOCAL_IP, current_timestamp
from easysort.runner import Runner
from flask import Flask, request, jsonify
from pathlib import Path
from dataclasses import dataclass, field
from easyprod.scripts.verdis.belt import ALLOWED_CATEGORIES, BeltRunnerJob, POLY_POINTS
from collections import Counter
import base64, io, random, json, gc
from tqdm import tqdm

waste_type_belt_prompt = f"Classify waste on belt. Categories: {ALLOWED_CATEGORIES}"

CACHE_FILE = Path("verdis_belt_ai_cache.json")

@dataclass
class VerdisBeltGroundTruth:
    category: str
    motion: bool
    id: str = field(default_factory=lambda: "5adf5d6f-539a-4533-9461-ff8b390fd9cf")
    metadata: RegistryBase.BaseDefaultTypes.BASEMETADATA = field(default_factory=lambda: RegistryBase.BaseDefaultTypes.BASEMETADATA(model="human", created_at=current_timestamp()))

class VerdisBeltValidator:
    prefix = "verdis/gadstrup/5"
    categories = ALLOWED_CATEGORIES
    port = 8080

    def __init__(self, registry: RegistryBase, review_mode: bool = False, run_ai: bool = False, 
                 datetime_from: str | None = None, datetime_to: str | None = None):
        self.registry, self.app = registry, Flask(__name__)
        self._idx = 0
        self._review_mode = review_mode
        self._run_ai = run_ai
        self._datetime_from = datetime_from  # Format: "YYYYMMDD_HHMMSS" or "YYYYMMDD"
        self._datetime_to = datetime_to
        self._motion_detector = BeltRunnerJob()
        self._setup_routes()
        
        # Load data and initialize folders
        self._ai_cache: dict[str, str] = {}  # folder_path -> category
        self._gt_cache: dict[str, dict] = {}  # folder_path -> {category, motion}
        self._load_and_display_distributions()
        self._folders = self._get_review_folders() if review_mode else self._get_prioritized_folders()
        
        if review_mode:
            print(f"\n*** REVIEW MODE: Reviewing {len(self._folders)} existing ground truths ***\n")
        if datetime_from or datetime_to:
            print(f"\n*** DATE RANGE MODE: {datetime_from or 'start'} to {datetime_to or 'end'} ***\n")

    def _load_cache(self) -> dict[str, str]:
        """Load AI results cache from local JSON file."""
        if CACHE_FILE.exists():
            try:
                return json.loads(CACHE_FILE.read_text())
            except:
                pass
        return {}

    def _save_cache(self):
        """Save AI results cache to local JSON file."""
        CACHE_FILE.write_text(json.dumps(self._ai_cache, indent=2))

    def _extract_datetime_from_image(self, img_path: Path) -> tuple[str, str] | None:
        """Extract date (YYYYMMDD) and time (HHMMSS) from image filename."""
        name = img_path.stem
        parts = name.split("_")
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                if i + 1 < len(parts) and len(parts[i + 1]) >= 6 and parts[i + 1][:6].isdigit():
                    return part, parts[i + 1][:6]
        return None

    def _is_on_quarter_hour(self, time_str: str) -> bool:
        """Check if time (HHMMSS) falls on XX:00, XX:15, XX:30, XX:45 (with 2min tolerance)."""
        if len(time_str) < 4:
            return False
        minutes = int(time_str[2:4])
        # 00: 58-59 or 00-02, 15: 13-17, 30: 28-32, 45: 43-47
        return (minutes <= 2 or minutes >= 58 or 
                (13 <= minutes <= 17) or 
                (28 <= minutes <= 32) or 
                (43 <= minutes <= 47))

    def _load_and_display_distributions(self):
        """Load AI results (with caching) and ground truth, display distributions."""
        files = self.registry.backend.LIST(self.prefix)
        self._files = files
        
        # Get hashes from registry
        hash_lookup = self.registry._get_hash_lookup()
        ai_hash = hash_lookup.get("3407561e-4531-473b-b7bb-7f75aebd7f76", "")
        gt_hash = hash_lookup.get(self.registry.get_id(VerdisBeltGroundTruth), "")
        
        # Find result files by hash in filename
        ai_files = [f for f in files if ai_hash and ai_hash in f.name]
        gt_files = [f for f in files if gt_hash and gt_hash in f.name]
        
        print(f"\nFound {len(ai_files)} AI result files, {len(gt_files)} ground truth files")
        
        # Build folder -> all images mapping (ALL folders first)
        all_folder_imgs: dict[Path, list[Path]] = {}
        for f in files:
            if f.suffix.lower() in (".jpg", ".png", ".jpeg"):
                all_folder_imgs.setdefault(f.parent, []).append(f)
        for folder in all_folder_imgs:
            all_folder_imgs[folder] = sorted(all_folder_imgs[folder])
        
        # Extract datetime for ALL folders first
        folder_datetime_all: dict[Path, tuple[str, str]] = {}
        folder_first_img: dict[Path, Path] = {}
        for folder, imgs in all_folder_imgs.items():
            if not imgs:
                continue
            dt = self._extract_datetime_from_image(imgs[0])
            if dt is None:
                for img in imgs[1:3]:
                    dt = self._extract_datetime_from_image(img)
                    if dt:
                        break
            if dt:
                folder_datetime_all[folder] = dt
                folder_first_img[folder] = imgs[0]
        
        # Check if we're in datetime range mode
        use_datetime_range = self._datetime_from or self._datetime_to
        
        def folder_in_range(date_str: str, time_str: str) -> bool:
            """Check if folder datetime is within the specified range."""
            folder_dt = f"{date_str}_{time_str}"
            if self._datetime_from:
                # Pad datetime_from if only date provided
                dt_from = self._datetime_from if "_" in self._datetime_from else f"{self._datetime_from}_000000"
                if folder_dt < dt_from:
                    return False
            if self._datetime_to:
                # Pad datetime_to if only date provided (use end of day)
                dt_to = self._datetime_to if "_" in self._datetime_to else f"{self._datetime_to}_235959"
                if folder_dt > dt_to:
                    return False
            return True
        
        self._folder_to_img = {}
        self._folder_to_all_imgs = {}
        self._folder_datetime: dict[Path, tuple[str, str]] = {}
        
        if use_datetime_range:
            # DATE RANGE MODE: Include ALL folders in range (no 15-minute filtering)
            for folder, (date_str, time_str) in folder_datetime_all.items():
                if folder_in_range(date_str, time_str):
                    self._folder_to_img[folder] = folder_first_img[folder]
                    self._folder_to_all_imgs[folder] = all_folder_imgs[folder]
                    self._folder_datetime[folder] = folder_datetime_all[folder]
            
            self._all_folders = sorted(self._folder_to_img.keys(), key=lambda f: folder_datetime_all[f])
            print(f"Date range filter: {len(self._all_folders)} folders in range (all folders, no 15-min filtering)")
        else:
            # NORMAL MODE: Group by quarter-hour slot, pick one per slot
            def get_quarter_slot(time_str: str) -> int:
                """Return quarter slot (0-3) for a time string HHMMSS."""
                minutes = int(time_str[2:4])
                if minutes < 15:
                    return 0
                elif minutes < 30:
                    return 1
                elif minutes < 45:
                    return 2
                else:
                    return 3
            
            slots: dict[str, list[tuple[Path, str]]] = {}  # slot_key -> [(folder, time_str), ...]
            for folder, (date_str, time_str) in folder_datetime_all.items():
                hour = time_str[:2]
                quarter = get_quarter_slot(time_str)
                slot_key = f"{date_str}-{hour}-{quarter}"
                slots.setdefault(slot_key, []).append((folder, time_str))
            
            # For each slot, pick the folder closest to the quarter-hour mark
            for slot_key, folder_times in slots.items():
                # Sort by time, pick the first (earliest in the quarter)
                folder_times.sort(key=lambda x: x[1])
                chosen_folder, chosen_time = folder_times[0]
                
                self._folder_to_img[chosen_folder] = folder_first_img[chosen_folder]
                self._folder_to_all_imgs[chosen_folder] = all_folder_imgs[chosen_folder]
                self._folder_datetime[chosen_folder] = folder_datetime_all[chosen_folder]
            
            skipped_count = len(all_folder_imgs) - len(self._folder_to_img)
            self._all_folders = sorted(self._folder_to_img.keys())
            print(f"Filtered to {len(self._all_folders)} folders (1 per 15-min slot, skipped {skipped_count})")
        
        # Load cache
        self._ai_cache = self._load_cache()
        cached_folders = set(self._ai_cache.keys())
        
        # Map AI files to folders that HAVE AI results
        # AI result stored at: folder/image_stem/hash.json, so use parent.parent
        folders_with_ai = {str(ai_file.parent.parent) for ai_file in ai_files}
        
        # Find folders WITHOUT AI results (only from our 15-minute filtered set)
        folders_without_ai = [f for f in self._all_folders if str(f) not in folders_with_ai]
        
        # Display missing AI results by date (from image timestamps)
        if folders_without_ai:
            missing_by_date: dict[str, list[Path]] = {}
            for folder in folders_without_ai:
                dt = self._folder_datetime.get(folder)
                date_str = dt[0] if dt else "unknown"
                missing_by_date.setdefault(date_str, []).append(folder)
            
            print(f"\n{'='*90}")
            print(f"MISSING AI RESULTS: {len(folders_without_ai)} folders across {len(missing_by_date)} dates")
            print(f"{'='*90}")
            for date in sorted(missing_by_date.keys(), reverse=True):
                folders_list = missing_by_date[date]
                count = len(folders_list)
                formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if date != "unknown" else "unknown"
                example = folders_list[0]
                example_img = self._folder_to_img.get(example)
                example_name = example_img.name if example_img else example.name
                
                # Count by hour
                hour_counts: dict[str, int] = {}
                for f in folders_list:
                    dt = self._folder_datetime.get(f)
                    if dt:
                        hour = dt[1][:2]
                        hour_counts[hour] = hour_counts.get(hour, 0) + 1
                hours_str = " ".join(f"{h}h:{c}" for h, c in sorted(hour_counts.items()))
                
                print(f"  {formatted_date}: {count:>4} folders  (e.g. {example_name})")
                print(f"      hours: {hours_str}")
            print(f"{'='*90}")
            
            if self._run_ai:
                self._run_ai_on_missing(folders_without_ai)
                files = self.registry.backend.LIST(self.prefix)
                self._files = files
                ai_files = [f for f in files if ai_hash and ai_hash in f.name]
                folders_with_ai = {str(ai_file.parent.parent) for ai_file in ai_files}
            else:
                print(f"\nRun with --run-ai to generate AI results for missing folders\n")
        
        # Find new AI results not in cache
        new_folders_to_cache = [str(f) for f in self._all_folders if str(f) in folders_with_ai and str(f) not in cached_folders]
        print(f"Cache: {len(cached_folders)} cached, {len(new_folders_to_cache)} new to fetch")
        
        for folder_str in tqdm(new_folders_to_cache, desc="Fetching new AI results"):
            try:
                folder = Path(folder_str)
                if folder in self._folder_to_img:
                    result = self.registry.GET(self._folder_to_img[folder], BeltRunnerJob.RegistryResult, throw_error=False)
                    if result:
                        self._ai_cache[folder_str] = result.category
            except:
                pass
        
        self._save_cache()
        
        # Count AI categories (only for filtered folders)
        ai_categories = Counter(v for k, v in self._ai_cache.items() if Path(k) in self._folder_to_img)
        ai_count = sum(1 for k in self._ai_cache if Path(k) in self._folder_to_img)
        
        self._print_distribution("AI RESULTS DISTRIBUTION", ai_categories, ai_count, len(self._all_folders))
        
        # Load ground truth - GT stored at folder/image_stem/hash.json, so use parent.parent
        # Load ALL ground truths (not just 15-minute filtered) for distribution display
        self._gt_cache = {}
        all_gt_cache = {}  # For distribution display (all GTs)
        for gt_file in tqdm(gt_files, desc="Loading ground truth"):
            try:
                img_folder = gt_file.parent.parent
                # Get any image in this folder to retrieve GT
                imgs = sorted([f for f in self._files if f.parent == img_folder and f.suffix.lower() in (".jpg", ".png", ".jpeg")])
                if imgs:
                    gt = self.registry.GET(imgs[0], VerdisBeltGroundTruth, throw_error=False)
                    if gt:
                        all_gt_cache[str(img_folder)] = {"category": gt.category, "motion": gt.motion}
                        # Also add to main cache if in 15-minute filtered set
                        if img_folder in self._folder_to_img:
                            self._gt_cache[str(img_folder)] = {"category": gt.category, "motion": gt.motion}
            except:
                pass
        
        gt_categories = Counter(v["category"] for v in all_gt_cache.values())
        gt_motion = Counter("motion" if v["motion"] else "no_motion" for v in all_gt_cache.values())
        
        self._print_distribution("LABELED DATA DISTRIBUTION (Ground Truth - All)", gt_categories, len(all_gt_cache), len(gt_files), gt_motion)
        
        # Also show filtered GT count
        print(f"  ({len(self._gt_cache)} of these are in 15-minute filtered set)")

    def _run_ai_on_missing(self, folders: list[Path]):
        """Run AI classification on folders missing results, following vejebod.py procedure."""
        import cv2
        import numpy as np
        
        def highlight_belt_from_array(img_array: np.ndarray, points: list = POLY_POINTS) -> np.ndarray:
            """Dim area outside polygon, highlight belt region with yellow border (from numpy array)."""
            h, w = img_array.shape[:2]
            pts = np.array([[max(0, min(w-1, x)), max(0, min(h-1, y))] for x, y in points], np.int32)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            dim = (img_array * 0.25).astype(np.uint8)
            dim[mask == 255] = img_array[mask == 255]
            cv2.polylines(dim, [pts], True, (0, 255, 255), 2)
            return dim
        
        print(f"\n{'='*50}")
        print(f"RUNNING AI ON {len(folders)} MISSING FOLDERS")
        print(f"{'='*50}\n")
        
        # Collect all first images from missing folders
        paths_to_process = []
        for folder in folders:
            if folder in self._folder_to_img:
                paths_to_process.append(self._folder_to_img[folder])
        
        if not paths_to_process:
            print("No images to process")
            return
        
        print(f"Processing {len(paths_to_process)} images...")
        
        # Create runner for GPT calls
        runner = Runner()
        batch_size = 100
        num_batches = (len(paths_to_process) + batch_size - 1) // batch_size
        
        for bi, start in enumerate(range(0, len(paths_to_process), batch_size), start=1):
            batch_paths = paths_to_process[start:start + batch_size]
            print(f"\nBatch {bi}/{num_batches}: {len(batch_paths)} images")
            
            # Preprocess images with highlight_belt
            images = []
            valid_paths = []
            for p in tqdm(batch_paths, desc=f"Preprocessing (batch {bi}/{num_batches})"):
                try:
                    # Get image from registry and convert to numpy array (BGR for cv2)
                    pil_img = self.registry.GET(p, self.registry.DefaultMarkers.ORIGINAL_MARKER)
                    img_array = np.array(pil_img)
                    # Convert RGB to BGR for cv2
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    highlighted = highlight_belt_from_array(img_array)
                    if highlighted is not None:
                        images.append([highlighted])
                        valid_paths.append(p)
                except Exception as e:
                    print(f"Error preprocessing {p}: {e}")
            
            if not images:
                continue
            
            # Run GPT classification
            gpt_results = runner.gpt(images, BeltRunnerJob.RegistryResult, waste_type_belt_prompt)
            
            # Save results
            print(f"Saving GPT results (batch {bi}/{num_batches})...")
            for path, result in zip(valid_paths, gpt_results):
                try:
                    self.registry.POST(path, result, BeltRunnerJob.RegistryResult)
                    # Update local cache
                    self._ai_cache[str(path.parent)] = result.category
                except Exception as e:
                    print(f"Error saving result for {path}: {e}")
            
            # Memory cleanup
            del images, gpt_results
            gc.collect()
        
        # Save updated cache
        self._save_cache()
        
        # Run motion detection on groups
        print(f"\nRunning motion detection...")
        groups: dict[Path, list[Path]] = {}
        for folder in folders:
            if folder in self._folder_to_all_imgs:
                groups[folder] = self._folder_to_all_imgs[folder]
        
        for folder, folder_imgs in tqdm(groups.items(), desc="Motion detection"):
            if folder_imgs:
                try:
                    motion_result = self._motion_detector._motion_detection(folder_imgs)
                    self.registry.POST(folder_imgs[0], motion_result, BeltRunnerJob.MotionDetectionResult)
                except Exception as e:
                    print(f"Error with motion detection for {folder}: {e}")
        
        print(f"\n{'='*50}")
        print(f"AI PROCESSING COMPLETE")
        print(f"{'='*50}\n")

    def _print_distribution(self, title: str, categories: Counter, count: int, total: int, motion: Counter = None):
        """Print a distribution with progress bars."""
        print(f"\n{'='*50}")
        print(title)
        print(f"{'='*50}")
        print(f"Total: {count}/{total}")
        if categories:
            for cat in self.categories:
                c = categories.get(cat, 0)
                pct = (c / count * 100) if count > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  {cat:<15} {bar} {c:>4} ({pct:>5.1f}%)")
            if motion:
                print(f"\nMotion distribution:")
                for m in ["motion", "no_motion"]:
                    c = motion.get(m, 0)
                    pct = (c / count * 100) if count > 0 else 0
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(f"  {m:<15} {bar} {c:>4} ({pct:>5.1f}%)")
        else:
            print("  No data found")
        print(f"{'='*50}\n")

    def _get_prioritized_folders(self) -> list[Path]:
        """Get unlabeled folders, prioritized by what categories need more labels, but shuffled for variety."""
        # Find already labeled folders
        labeled_folders = set(self._gt_cache.keys())
        unlabeled = [f for f in self._all_folders if str(f) not in labeled_folders]
        
        # Split into folders with AI results and without
        with_ai = [f for f in unlabeled if str(f) in self._ai_cache]
        without_ai = [f for f in unlabeled if str(f) not in self._ai_cache]
        
        # Calculate target distribution (equal per category)
        gt_categories = Counter(v["category"] for v in self._gt_cache.values())
        total_labeled = len(self._gt_cache)
        target_per_cat = max(1, (total_labeled + len(with_ai)) // len(self.categories))
        
        # Calculate how many more we need per category
        needed = {cat: max(0, target_per_cat - gt_categories.get(cat, 0)) for cat in self.categories}
        
        # Group folders by category and shuffle within each group
        by_category: dict[str, list[Path]] = {cat: [] for cat in self.categories}
        by_category["unknown"] = []
        for folder in with_ai:
            cat = self._ai_cache.get(str(folder), "unknown")
            if cat in by_category:
                by_category[cat].append(folder)
            else:
                by_category["unknown"].append(folder)
        
        # Shuffle within each category
        for cat_folders in by_category.values():
            random.shuffle(cat_folders)
        
        # Interleave categories, weighted by need (take more from categories that need more labels)
        # Sort categories by need (highest first)
        sorted_cats = sorted(self.categories, key=lambda c: -needed.get(c, 0))
        
        result = []
        while any(by_category[cat] for cat in sorted_cats):
            for cat in sorted_cats:
                if by_category[cat]:
                    # Take 1-3 items based on relative need
                    take = min(len(by_category[cat]), max(1, needed.get(cat, 0) // 10 + 1))
                    for _ in range(take):
                        if by_category[cat]:
                            result.append(by_category[cat].pop(0))
        
        # Add any unknown category folders
        result.extend(by_category.get("unknown", []))
        
        # Shuffle folders without AI results and add at the end
        random.shuffle(without_ai)
        result.extend(without_ai)
        
        print(f"Folders: {len(labeled_folders)} labeled, {len(with_ai)} with AI (prioritized+shuffled), {len(without_ai)} without AI (shuffled)")
        print(f"Category needs: {dict(sorted(needed.items(), key=lambda x: -x[1]))}")
        
        return result

    def _get_review_folders(self) -> list[Path]:
        """Get labeled folders for review mode."""
        labeled = [f for f in self._all_folders if str(f) in self._gt_cache]
        random.shuffle(labeled)
        return labeled

    def get_folders(self) -> list[Path]:
        """Get prioritized folders (already computed at init)."""
        return self._folders

    def _get_images(self, folder: Path) -> list[Path]:
        files = self.registry.backend.LIST(str(folder))
        return sorted([f for f in files if f.suffix.lower() in (".jpg", ".png", ".jpeg")])[:3]

    def _image_b64(self, key: Path) -> str:
        data = self.registry.GET(key, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        buf = io.BytesIO(); data.save(buf, format="JPEG"); return base64.b64encode(buf.getvalue()).decode()

    def _get_prediction(self, folder: Path) -> dict | None:
        images = self._get_images(folder)
        if not images: return None
        first = images[0]
        # Check if original image exists
        if not self.registry.backend.EXISTS(first):
            print(f"Skipping {folder.name}: image not found")
            return None
        
        # Use cached category if available
        folder_str = str(folder)
        category = self._ai_cache.get(folder_str)
        
        # Fetch from registry if not cached
        if category is None:
            try:
                cat_result = self.registry.GET(first, BeltRunnerJob.RegistryResult, throw_error=False)
                category = cat_result.category if cat_result else None
                # Update cache if found
                if category:
                    self._ai_cache[folder_str] = category
                    self._save_cache()
            except (AssertionError, Exception) as e:
                print(f"Skipping {folder.name}: {e}")
                return None
        
        # Always fetch motion (small, not worth caching separately)
        try:
            motion_result = self.registry.GET(first, BeltRunnerJob.MotionDetectionResult, throw_error=False)
        except:
            motion_result = None
        
        # Generate motion detection if not found
        if motion_result is None:
            motion_result = self._motion_detector._motion_detection(images)
        
        return {
            "category": category,
            "motion": motion_result.motion_detected if motion_result else False
        }

    def _save(self, folder: Path, category: str, motion: bool):
        images = self._get_images(folder)
        if images:
            gt = VerdisBeltGroundTruth(category=category, motion=motion)
            self.registry.POST(images[0], gt, VerdisBeltGroundTruth, overwrite=True)
            # Update ground truth cache
            self._gt_cache[str(folder)] = {"category": category, "motion": motion}
            print(f"Saved: {folder.name} -> category={category}, motion={motion}")

    def _setup_routes(self):
        @self.app.route("/")
        def index(): return HTML.replace("{{CATEGORIES}}", str(self.categories))

        @self.app.route("/next")
        def next_item():
            # Skip folders with missing/invalid images
            while self._idx < len(self._folders):
                folder = self._folders[self._idx]
                images = self._get_images(folder)
                pred = self._get_prediction(folder)
                if pred is not None and images:
                    try:
                        img_data = [{"key": str(img), "b64": self._image_b64(img)} for img in images]
                        response = {
                            "done": False, "folder": str(folder), "idx": self._idx, "total": len(self._folders),
                            "images": img_data, "prediction": pred, "review_mode": self._review_mode
                        }
                        # In review mode, also include the existing ground truth
                        if self._review_mode:
                            gt = self._gt_cache.get(str(folder))
                            response["ground_truth"] = gt
                        return jsonify(response)
                    except Exception as e:
                        print(f"Skipping {folder.name}: {e}")
                self._idx += 1
            return jsonify({"done": True, "total": len(self._folders)})

        @self.app.route("/label", methods=["POST"])
        def label():
            data = request.json
            self._save(Path(data["folder"]), data["category"], data["motion"])
            self._idx += 1
            return jsonify({"ok": True})

        @self.app.route("/accept", methods=["POST"])
        def accept():
            """Accept AI prediction as ground truth."""
            data = request.json
            pred = data.get("prediction", {})
            if pred.get("category"):
                self._save(Path(data["folder"]), pred["category"], pred.get("motion", False))
                self._idx += 1
                return jsonify({"ok": True})
            return jsonify({"ok": False, "error": "No AI prediction to accept"})

        @self.app.route("/skip", methods=["POST"])
        def skip(): self._idx += 1; return jsonify({"ok": True})

        @self.app.route("/back", methods=["POST"])
        def back(): self._idx = max(0, self._idx - 1); return jsonify({"ok": True})

    def run(self, host="0.0.0.0"):
        print(f"Verdis Belt Validator at http://{host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=False, threaded=True)

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Verdis Belt Validator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#1a1a1a;color:#fff;height:100vh;display:flex;flex-direction:column}
#images{flex:1;display:flex;justify-content:center;align-items:center;position:relative;overflow:hidden;background:#000}
#images img{max-height:85vh;max-width:95%;object-fit:contain;border-radius:4px;position:absolute;opacity:0;transition:opacity 0.15s}
#images img.active{opacity:1}
#images.loading img{opacity:0 !important}
#frame-indicator{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);display:flex;gap:8px}
#frame-indicator span{width:10px;height:10px;border-radius:50%;background:#555}
#frame-indicator span.active{background:#fff}
#info-overlay{position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.7);padding:8px 12px;border-radius:4px;font-size:12px;display:none}
#info-overlay.show{display:block}
#info-overlay .gt{color:#4f8}
#info-overlay .ai{color:#4af}
#bar{padding:12px;background:#222;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}
.btn{padding:8px 14px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:500}
.cat{background:#333;color:#888}.cat:hover{background:#555;color:#fff}
.cat.match{background:#0d4a6d;color:#fff;outline:3px solid #0af;outline-offset:-1px}
.cat.gt-match{background:#0d6d4a;color:#fff;outline:3px solid #0f8;outline-offset:-1px}
.accept{background:#2d5a2d;color:#fff}.accept:hover{background:#3d6a3d}
.skip{background:#666}.back{background:#555}
.motion{background:#3a5a3a}.motion.active{background:#4a8a4a;border:2px solid #6f6}
.no-motion{background:#5a3a3a}.no-motion.active{background:#8a4a4a;border:2px solid #f66}
#progress{color:#888;margin-left:auto}
#done{display:none;font-size:24px;text-align:center;padding:48px}
kbd{background:#333;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px}
.review-banner{background:#553;color:#ff8;padding:4px 12px;text-align:center;font-size:12px}
</style></head><body>
<div id="review-banner" class="review-banner" style="display:none">REVIEW MODE - Checking existing ground truths</div>
<div id="images"><div id="frame-indicator"></div><div id="info-overlay"></div></div>
<div id="bar">
  <button class="btn back" onclick="back()"><kbd>B</kbd>Back</button>
  <button class="btn accept" id="btn-accept" onclick="accept()"><kbd>A</kbd>Accept AI</button>
  <span id="cats"></span>
  <button class="btn skip" onclick="skip()"><kbd>S</kbd>Skip</button>
  <span id="motion-btns">
    <button class="btn motion" id="btn-motion" onclick="setMotion(true)"><kbd>M</kbd>Moving</button>
    <button class="btn no-motion" id="btn-no-motion" onclick="setMotion(false)"><kbd>N</kbd>Not Moving</button>
  </span>
  <span id="progress">-/-</span>
</div>
<div id="done">✓ All done!</div>
<script>
const CATS={{CATEGORIES}};
let currentFolder=null, currentMotion=false, currentFrame=0, frameCount=0, loopInterval=null, currentPrediction=null;

function updateMotionBtns(){
  document.getElementById('btn-motion').classList.toggle('active',currentMotion===true);
  document.getElementById('btn-no-motion').classList.toggle('active',currentMotion===false);
}

function setMotion(val){currentMotion=val;updateMotionBtns()}

function showFrame(idx){
  currentFrame=idx%frameCount;
  document.querySelectorAll('#images img').forEach((img,i)=>img.classList.toggle('active',i===currentFrame));
  document.querySelectorAll('#frame-indicator span').forEach((s,i)=>s.classList.toggle('active',i===currentFrame));
}

function startLoop(){
  if(loopInterval)clearInterval(loopInterval);
  loopInterval=setInterval(()=>showFrame(currentFrame+1),500);
}

function render(data){
  if(data.done){document.getElementById('done').style.display='block';document.getElementById('bar').style.display='none';document.getElementById('images').style.display='none';if(loopInterval)clearInterval(loopInterval);return}
  frameCount=data.images.length;
  currentPrediction=data.prediction;
  
  // Show review banner if in review mode
  document.getElementById('review-banner').style.display=data.review_mode?'block':'none';
  
  // Build overlay HTML (always create element, just hide when not needed)
  const showOverlay=data.review_mode&&data.ground_truth;
  const overlayHtml=`<div id="info-overlay"${showOverlay?' class="show"':''}>${showOverlay?`<div class="gt">GT: ${data.ground_truth.category} (${data.ground_truth.motion?'moving':'still'})</div><div class="ai">AI: ${data.prediction.category||'none'} (${data.prediction.motion?'moving':'still'})</div>`:''}</div>`;
  
  document.getElementById('images').innerHTML=data.images.map((i,idx)=>`<img src="data:image/jpeg;base64,${i.b64}" class="${idx===0?'active':''}">`).join('')+
    `<div id="frame-indicator">${data.images.map((_,i)=>`<span class="${i===0?'active':''}"></span>`).join('')}</div>`+overlayHtml;
  currentFolder=data.folder;
  currentMotion=data.prediction.motion;
  currentFrame=0;
  updateMotionBtns();
  document.getElementById('progress').textContent=`${data.idx+1}/${data.total}`;
  
  // Highlight matching categories
  console.log('AI prediction category:',data.prediction.category);
  document.querySelectorAll('.cat').forEach(b=>{
    const isMatch=b.dataset.cat===data.prediction.category;
    console.log('Button:',b.dataset.cat,'isMatch:',isMatch);
    b.classList.remove('match','gt-match');
    if(isMatch)b.classList.add('match');
    if(data.ground_truth&&b.dataset.cat===data.ground_truth.category)b.classList.add('gt-match');
  });
  
  // Disable accept button if no AI prediction
  document.getElementById('btn-accept').disabled=!data.prediction.category;
  
  startLoop();
}

async function load(){
  const data=await(await fetch('/next')).json();
  render(data);
}
async function label(cat){await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder:currentFolder,category:cat,motion:currentMotion})});await load()}
async function accept(){if(!currentPrediction||!currentPrediction.category)return;await fetch('/accept',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder:currentFolder,prediction:currentPrediction})});await load()}
async function skip(){await fetch('/skip',{method:'POST'});await load()}
async function back(){await fetch('/back',{method:'POST'});await load()}

document.getElementById('cats').innerHTML=CATS.map((c,i)=>`<button class="btn cat" data-cat="${c}" onclick="label('${c}')"><kbd>${i+1}</kbd>${c}</button>`).join('');
document.addEventListener('keydown',e=>{
  if(e.key>='1'&&e.key<='9'&&CATS[+e.key-1])label(CATS[+e.key-1]);
  else if(e.key.toLowerCase()==='a')accept();
  else if(e.key.toLowerCase()==='s')skip();
  else if(e.key.toLowerCase()==='b')back();
  else if(e.key.toLowerCase()==='m')setMotion(true);
  else if(e.key.toLowerCase()==='n')setMotion(false);
});
load();
</script></body></html>"""

if __name__ == "__main__":
    import sys
    review_mode = "--review" in sys.argv or "-r" in sys.argv
    run_ai = "--run-ai" in sys.argv
    
    # Parse datetime range arguments
    datetime_from = None
    datetime_to = None
    for i, arg in enumerate(sys.argv):
        if arg == "--from" and i + 1 < len(sys.argv):
            datetime_from = sys.argv[i + 1]
        elif arg == "--to" and i + 1 < len(sys.argv):
            datetime_to = sys.argv[i + 1]
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python verdis_belt.py [OPTIONS]")
        print("\nOptions:")
        print("  --review, -r              Review existing ground truths")
        print("  --run-ai                  Run AI on folders missing results")
        print("  --from DATETIME           Start of date range (format: YYYYMMDD or YYYYMMDD_HHMMSS)")
        print("  --to DATETIME             End of date range (format: YYYYMMDD or YYYYMMDD_HHMMSS)")
        print("  --help, -h                Show this help message")
        print("\nExamples:")
        print("  python verdis_belt.py --from 20260128 --to 20260128")
        print("  python verdis_belt.py --from 20260128_100000 --to 20260128_120000")
        print("\nNote: When using --from/--to, ALL folders in the range are shown (no 15-min filtering)")
        sys.exit(0)
    
    registry = RegistryBase(RegistryConnector(REGISTRY_LOCAL_IP))
    registry.add_id(VerdisBeltGroundTruth, "5adf5d6f-539a-4533-9461-ff8b390fd9cf")
    validator = VerdisBeltValidator(registry, review_mode=review_mode, run_ai=run_ai, 
                                    datetime_from=datetime_from, datetime_to=datetime_to)
    validator.run()