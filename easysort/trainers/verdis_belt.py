"""Verdis Belt Trainer - category and motion classification."""
import numpy as np
import random
import sys
from tqdm import tqdm
from pathlib import Path

from easysort.registry import RegistryBase, RegistryConnector
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.validators.verdis_belt import VerdisBeltGroundTruth
from easysort.gpt_trainer import YoloTrainer
from easyprod.scripts.verdis.belt import ALLOWED_CATEGORIES, POLY_POINTS

MAX_SAMPLES = 10000
EPOCHS = 30  # Max epochs (early stopping will likely trigger before this)
PATIENCE = 5  # Early stopping patience
MIN_EPOCHS = 5  # Minimum epochs before early stopping can trigger
BBOX = (min(p[0] for p in POLY_POINTS), min(p[1] for p in POLY_POINTS), max(p[0] for p in POLY_POINTS), max(p[1] for p in POLY_POINTS))

def crop(img, pad=20):
    x1, y1, x2, y2 = BBOX
    h, w = img.shape[:2]
    return img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

def load_data(registry):
    """Load labeled data from registry."""
    gt_hash = registry._get_hash_lookup().get("5adf5d6f-539a-4533-9461-ff8b390fd9cf", "")
    gt_files = [f for f in registry.backend.LIST("verdis/gadstrup/5") if gt_hash in f.name]
    if len(gt_files) > MAX_SAMPLES: gt_files = random.sample(gt_files, MAX_SAMPLES)
    
    cat_imgs, cat_labels, motion_imgs, motion_labels = [], [], [], []
    
    for gt_file in tqdm(gt_files, desc="Loading"):
        folder = gt_file.parent.parent
        imgs = sorted([f for f in registry.backend.LIST(str(folder)) if f.suffix.lower() in (".jpg", ".png", ".jpeg")])[:3]
        if not imgs: continue
        try:
            gt = registry.GET(imgs[0], VerdisBeltGroundTruth, throw_error=False)
            if not gt: continue
            for p in imgs:
                cat_imgs.append(crop(np.array(registry.GET(p, registry.DefaultMarkers.ORIGINAL_MARKER))))
                cat_labels.append(gt.category.lower().replace(' ', '_'))
            motion_imgs.append(crop(np.array(registry.GET(imgs[len(imgs)//2], registry.DefaultMarkers.ORIGINAL_MARKER))))
            motion_labels.append("motion" if gt.motion else "no_motion")
        except: pass
    
    print(f"Loaded {len(cat_imgs)} category, {len(motion_imgs)} motion samples")
    return cat_imgs, cat_labels, motion_imgs, motion_labels

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
    registry = RegistryBase(RegistryConnector(REGISTRY_LOCAL_IP))
    cat_imgs, cat_labels, motion_imgs, motion_labels = load_data(registry)
    
    cat_trainer = YoloTrainer("verdis_category", [c.lower().replace(' ', '_') for c in ALLOWED_CATEGORIES])
    motion_trainer = YoloTrainer("verdis_motion", ["motion", "no_motion"])
    
    if len(sys.argv) < 2 or sys.argv[1] == "train":
        print("\n" + "="*70)
        print("=== CATEGORY CLASSIFIER ===")
        print("="*70)
        cat_trainer.train(cat_imgs, cat_labels, epochs=EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS)
        
        print("\n" + "="*70)
        print("=== MOTION CLASSIFIER ===")
        print("="*70)
        motion_trainer.train(motion_imgs, motion_labels, epochs=EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS)
    elif sys.argv[1] == "eval":
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