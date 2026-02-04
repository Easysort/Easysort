"""Verdis Belt Trainer - category and motion classification."""
import numpy as np
import random
import sys
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from easysort.registry import RegistryBase, RegistryConnector
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.validators.verdis_belt import VerdisBeltGroundTruth
from easysort.gpt_trainer import YoloTrainer
from easyprod.scripts.verdis.belt import ALLOWED_CATEGORIES, POLY_POINTS

MAX_SAMPLES_PER_CLASS = 350  # Max samples per category to balance dataset and reduce memory
EPOCHS = 30  # Max epochs (early stopping will likely trigger before this)
PATIENCE = 5  # Early stopping patience (YOLO stops if no val improvement)
BBOX = (min(p[0] for p in POLY_POINTS), min(p[1] for p in POLY_POINTS), max(p[0] for p in POLY_POINTS), max(p[1] for p in POLY_POINTS))

def crop(img, pad=20):
    x1, y1, x2, y2 = BBOX
    h, w = img.shape[:2]
    return img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

def load_data(registry):
    """Load labeled data from registry with balanced sampling.
    
    Returns cat_imgs, cat_labels, motion_labels (NO motion_imgs to save memory).
    Motion images can be created later by stacking every 3 cat_imgs.
    """
    # List all files ONCE upfront
    print("Listing all files from registry...")
    all_files = list(registry.backend.LIST("verdis/gadstrup/5"))
    print(f"Found {len(all_files)} total files")
    
    # Build lookup: folder -> list of image files
    folder_images: dict[Path, list[Path]] = defaultdict(list)
    for f in all_files:
        if f.suffix.lower() in (".jpg", ".png", ".jpeg"):
            folder = f.parent
            folder_images[folder].append(f)
    
    for folder in folder_images:
        folder_images[folder] = sorted(folder_images[folder])[:3]
    
    # Find ground truth files
    gt_hash = registry._get_hash_lookup().get("5adf5d6f-539a-4533-9461-ff8b390fd9cf", "")
    gt_files = [f for f in all_files if gt_hash in f.name]
    print(f"Found {len(gt_files)} ground truth files")
    
    # First pass: read labels and group by category (no image loading yet)
    print("Reading labels and grouping by category...")
    category_samples: dict[str, list[tuple]] = defaultdict(list)  # cat -> [(gt_file, gt, folder)]
    
    for gt_file in tqdm(gt_files, desc="Reading labels"):
        folder = gt_file.parent.parent
        imgs = folder_images.get(folder, [])
        if not imgs: continue
        try:
            gt = registry.GET(imgs[0], VerdisBeltGroundTruth, throw_error=False)
            if not gt: continue
            cat = gt.category.lower().replace(' ', '_')
            category_samples[cat].append((gt_file, gt, folder))
        except: pass
    
    # Print distribution before balancing
    print("\nCategory distribution (before balancing):")
    for cat, samples in sorted(category_samples.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(samples)}")
    
    # Balance: sample up to MAX_SAMPLES_PER_CLASS from each category
    balanced_samples = []
    for cat, samples in category_samples.items():
        if len(samples) > MAX_SAMPLES_PER_CLASS:
            samples = random.sample(samples, MAX_SAMPLES_PER_CLASS)
        balanced_samples.extend(samples)
    
    print(f"\nAfter balancing: {len(balanced_samples)} samples (max {MAX_SAMPLES_PER_CLASS}/class)")
    random.shuffle(balanced_samples)
    
    # Second pass: load images (only cat_imgs, motion_imgs created later to save memory)
    cat_imgs, cat_labels, motion_labels = [], [], []
    errors = 0
    
    for i, (gt_file, gt, folder) in enumerate(tqdm(balanced_samples, desc="Loading images")):
        img_paths = folder_images.get(folder, [])
        if len(img_paths) < 3: continue  # Need all 3 frames
        try:
            cat = gt.category.lower().replace(' ', '_')
            motion = "motion" if gt.motion else "no_motion"
            
            # Load all 3 images for category (each is a separate sample)
            for p in img_paths:
                img = crop(np.array(registry.GET(p, registry.DefaultMarkers.ORIGINAL_MARKER)))
                cat_imgs.append(img)
                cat_labels.append(cat)
            
            # Store motion label (1 per 3 images)
            motion_labels.append(motion)
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError at {i}: {gt_file}: {e}")
        
        if i > 0 and i % 200 == 0:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024**3
            print(f"\n[{i}/{len(balanced_samples)}] Memory: {mem:.2f}GB, {len(cat_imgs)} imgs", flush=True)
    
    print(f"\nLoaded {len(cat_imgs)} category images, {len(motion_labels)} motion labels ({errors} errors)")
    print(f"Categories: {len(set(cat_labels))} classes")
    print(f"Motion: { {l: motion_labels.count(l) for l in set(motion_labels)} }")
    
    return cat_imgs, cat_labels, motion_labels


def convert_to_motion_imgs(cat_imgs: list) -> list:
    """Convert category images to motion images by stacking every 3 horizontally."""
    motion_imgs = []
    for i in range(0, len(cat_imgs), 3):
        if i + 2 < len(cat_imgs):
            stacked = np.concatenate([cat_imgs[i], cat_imgs[i+1], cat_imgs[i+2]], axis=1)
            motion_imgs.append(stacked)
    return motion_imgs

def copy_models_to_prod():
    """Copy trained models to easyprod/models for production use.
    
    Prioritizes the best_balanced.pt model (selected by balanced accuracy)
    over the standard best.pt (selected by YOLO's default metric).
    """
    import shutil
    prod_models = Path("easyprod/models")
    prod_models.mkdir(parents=True, exist_ok=True)
    
    # Prioritize best_balanced.pt (our custom best model based on balanced accuracy)
    model_sources = [
        ("verdis_category_model/best_balanced.pt", "verdis_category_model/train/weights/best.pt", "verdis_category.pt"),
        ("verdis_motion_model/best_balanced.pt", "verdis_motion_model/train/weights/best.pt", "verdis_motion.pt"),
    ]
    
    for primary_src, fallback_src, dst in model_sources:
        src = Path(primary_src) if Path(primary_src).exists() else Path(fallback_src)
        if src.exists():
            shutil.copy(src, prod_models / dst)
            print(f"Copied {src} -> {prod_models / dst}")
        else:
            print(f"Warning: Neither {primary_src} nor {fallback_src} found")

if __name__ == "__main__":
    import gc
    
    registry = RegistryBase(RegistryConnector(REGISTRY_LOCAL_IP))
    if len(sys.argv) < 2 or sys.argv[1] == "copy":
        copy_models_to_prod()
        sys.exit(0)

    # Load data (no motion_imgs yet - saves memory)
    cat_imgs, cat_labels, motion_labels = load_data(registry)
    
    # Only use classes that actually have samples (fixes class mismatch errors)
    actual_cat_classes = sorted(set(cat_labels))
    actual_motion_classes = sorted(set(motion_labels))
    
    print(f"\nCategory classes with data: {actual_cat_classes}")
    print(f"Motion classes with data: {actual_motion_classes}")
    
    cat_trainer = YoloTrainer("verdis_category", actual_cat_classes)
    motion_trainer = YoloTrainer("verdis_motion", actual_motion_classes)
    
    if len(sys.argv) < 2 or sys.argv[1] == "train":
        # === CATEGORY TRAINING ===
        print("\n" + "="*70)
        print("=== CATEGORY CLASSIFIER ===")
        print("="*70)
        cat_trainer.train(cat_imgs, cat_labels, epochs=EPOCHS, patience=PATIENCE)
        
        # Convert cat_imgs to motion_imgs (stack every 3)
        print("\nConverting to motion images...")
        motion_imgs = convert_to_motion_imgs(cat_imgs)
        print(f"Created {len(motion_imgs)} motion images from {len(cat_imgs)} category images")
        
        # Free category data - motion_imgs now holds the stacked versions
        del cat_imgs, cat_labels
        gc.collect()
        print("Freed category data from memory")
        
        # === MOTION TRAINING ===
        print("\n" + "="*70)
        print("=== MOTION CLASSIFIER ===")
        print("="*70)
        motion_trainer.train(motion_imgs, motion_labels, epochs=EPOCHS, patience=PATIENCE)
        
        # Free motion data
        del motion_imgs, motion_labels
        gc.collect()
        
        # Copy to production
        copy_models_to_prod()
        
    elif sys.argv[1] == "eval":
        # For eval, we need motion_imgs
        motion_imgs = convert_to_motion_imgs(cat_imgs)
        
        print("\n" + "="*70)
        print("=== CATEGORY EVALUATION ===")
        print("="*70)
        cat_trainer.evaluate(cat_imgs, cat_labels)
        
        print("\n" + "="*70)
        print("=== MOTION EVALUATION ===")
        print("="*70)
        motion_trainer.evaluate(motion_imgs, motion_labels)
    elif sys.argv[1] == "copy":
        copy_models_to_prod()
    else:
        print("Usage: python verdis_belt.py [train|eval|copy]")
        sys.exit(1)