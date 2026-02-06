"""GPT and YOLO trainers."""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import base64
import random
import shutil
import gc

import openai
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from easysort.helpers import OPENAI_API_KEY


class GPTTrainer:
    def __init__(self, model: str = "gpt-5-2025-08-07"):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.openai_client.models.list() # breaks if api key is invalid
        self.model = model

    def _openai_call(self, model: str, prompt: str, image_paths: List[List[np.ndarray]], output_schema: dataclass, max_workers: int = 10) -> List[dataclass]:
        def process_single(image_arrays):
            images_b64 = [base64.b64encode(cv2.imencode('.jpg', img_array)[1].tobytes()).decode("utf-8") for img_array in image_arrays]
            full_prompt = f"{prompt} Return only a json with the following keys and types: {output_schema.__annotations__}"
            content = [{"type": "text", "text": full_prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} for img_b64 in images_b64]
            response = self.openai_client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], response_format={"type": "json_object"}, timeout=90,)
            return output_schema(**json.loads(response.choices[0].message.content))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_single, image_paths), total=len(image_paths), desc="OpenAI calls"))
        return results

class YoloTrainer:
    # FIXED 
    def __init__(self, name: str, classes: List[str], model_path: str = "yolo11s-cls.pt"):
        self.name = name
        self.model = YOLO(model_path)
        self.classes = classes

    def train(self, images: List[np.ndarray], labels: List[str], epochs: int = 30, patience: int = 5):
        self.model.train(
            data=str(self.dataset),
            epochs=epochs,
            patience=patience,
            imgsz=224,
            batch=32,
            project=f"{self.name}_model",
            name="train",
            exist_ok=True,
            verbose=True,
            plots=True,
        )

    
class YoloTrainer2:
    def __init__(self, name: str, classes: List[str]):
        self.name = name
        self.classes = classes
        self.dataset = Path(f"{name}_dataset")
        self.model_path = Path(f"{name}_model/train/weights/best.pt")
        self.best_model_path = Path(f"{name}_model/best_balanced.pt")
    
    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    
    def _stratified_split(self, images: List[np.ndarray], labels: List[str], 
                          val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List, List, List, List, List, List]:
        """Split data into train/val/test maintaining class distribution.
        
        Returns: train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels
        """
        # Group indices by class
        class_indices: Dict[str, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            if label in self.classes:
                class_indices[label].append(i)
        
        train_indices, val_indices, test_indices = [], [], []
        
        # For each class, split proportionally
        for cls, indices in class_indices.items():
            random.shuffle(indices)
            n = len(indices)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            
            test_indices.extend(indices[:n_test])
            val_indices.extend(indices[n_test:n_test + n_val])
            train_indices.extend(indices[n_test + n_val:])
        
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)
        
        return (
            [images[i] for i in train_indices], [labels[i] for i in train_indices],
            [images[i] for i in val_indices], [labels[i] for i in val_indices],
            [images[i] for i in test_indices], [labels[i] for i in test_indices],
        )
    
    def _compute_metrics(self, model: YOLO, images: List[np.ndarray], labels: List[str]) -> Tuple[Dict[str, Dict], float]:
        """Compute per-class metrics and balanced accuracy using batch prediction."""
        # Batch predict for speed
        bgr_images = [self._to_bgr(img) for img in images]
        results = model.predict(bgr_images, verbose=False, batch=32)
        preds = [model.names[int(r.probs.top1)] for r in results]
        
        class_metrics = {}
        class_accuracies = []
        
        for cls in self.classes:
            n_total = sum(1 for l in labels if l == cls)
            if n_total == 0:
                continue
            n_correct = sum(1 for l, p in zip(labels, preds) if l == cls and p == cls)
            n_predicted = sum(1 for p in preds if p == cls)
            
            precision = n_correct / n_predicted if n_predicted > 0 else 0
            recall = n_correct / n_total
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {
                'accuracy': recall,  # Per-class accuracy is recall
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_samples': n_total,
                'n_correct': n_correct
            }
            class_accuracies.append(recall)
        
        # Balanced accuracy: mean of per-class accuracies
        balanced_acc = np.mean(class_accuracies) if class_accuracies else 0
        
        return class_metrics, balanced_acc
    
    def _print_epoch_metrics(self, epoch: int, val_metrics: Dict, test_metrics: Dict, 
                             val_bal_acc: float, test_bal_acc: float, is_best: bool):
        """Print per-category metrics for an epoch (val + held-out test)."""
        best_marker = " *** BEST ***" if is_best else ""
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}{best_marker}")
        print(f"{'='*80}")
        print(f"\n{'Class':<16} {'Val Acc':<10} {'Test Acc':<10} {'Test Prec':<11} {'Test Rec':<10} {'Test F1':<9} {'N(val)':<8} {'N(test)':<8}")
        print("-" * 92)
        
        for cls in self.classes:
            v = val_metrics.get(cls, {})
            t = test_metrics.get(cls, {})
            val_acc = f"{v.get('accuracy', 0):.1%}" if v else "N/A"
            test_acc = f"{t.get('accuracy', 0):.1%}" if t else "N/A"
            test_prec = f"{t.get('precision', 0):.1%}" if t else "N/A"
            test_rec = f"{t.get('recall', 0):.1%}" if t else "N/A"
            test_f1 = f"{t.get('f1', 0):.2f}" if t else "N/A"
            n_val = v.get('n_samples', 0)
            n_test = t.get('n_samples', 0)
            print(f"{cls:<16} {val_acc:<10} {test_acc:<10} {test_prec:<11} {test_rec:<10} {test_f1:<9} {n_val:<8} {n_test:<8}")
        
        print("-" * 92)
        print(f"{'BALANCED ACC':<16} {val_bal_acc:<10.1%} {test_bal_acc:<10.1%}")
        
        # Overfitting indicator: val much better than test
        if val_bal_acc - test_bal_acc > 0.10:
            print(f"⚠️  Val-Test gap: {val_bal_acc - test_bal_acc:.1%} (possible overfitting to val set)")
        print()
    
    def train(self, images: List[np.ndarray], labels: List[str], epochs: int = 30, 
              patience: int = 5):
        """
        Train with stratified train/val split. Uses YOLO's built-in early stopping.
        
        Args:
            images: Training images
            labels: Training labels  
            epochs: Maximum number of epochs
            patience: Early stopping patience (YOLO stops if no val improvement)
        """
        # Stratified split: train (85%), val (15%)
        train_imgs, train_labels, val_imgs, val_labels, _, _ = self._stratified_split(
            images, labels, val_ratio=0.15, test_ratio=0.0
        )
        
        # Print split distribution
        print(f"\n{'='*70}")
        print(f"DATASET SPLIT - {self.name}")
        print(f"{'='*70}")
        print(f"\n{'Class':<20} {'Train':<10} {'Val':<10} {'Total':<10}")
        print("-" * 50)
        for cls in self.classes:
            n_train = sum(1 for l in train_labels if l == cls)
            n_val = sum(1 for l in val_labels if l == cls)
            print(f"{cls:<20} {n_train:<10} {n_val:<10} {n_train + n_val:<10}")
        print("-" * 50)
        print(f"{'TOTAL':<20} {len(train_labels):<10} {len(val_labels):<10} {len(train_labels) + len(val_labels):<10}")
        print()
        
        # Create dataset directories
        for split in ['train', 'val']:
            for cls in self.classes:
                (self.dataset / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Save training images
        for i, (img, label) in enumerate(tqdm(zip(train_imgs, train_labels), total=len(train_imgs), desc="Saving train")):
            path = self.dataset / 'train' / label / f"{i:06d}.jpg"
            cv2.imwrite(str(path), self._to_bgr(img))
        
        # Save validation images
        for i, (img, label) in enumerate(tqdm(zip(val_imgs, val_labels), total=len(val_imgs), desc="Saving val")):
            path = self.dataset / 'val' / label / f"{i:06d}.jpg"
            cv2.imwrite(str(path), self._to_bgr(img))
        
        # Free train images only (keep val for final evaluation)
        del train_imgs
        gc.collect()
        print("Freed train image memory before training")
        
        # Remove previous run weights so class count matches current dataset (avoids "requires 7 classes, not 6")
        weights_dir = Path(f"{self.name}_model/train/weights")
        if weights_dir.exists():
            for f in weights_dir.glob("*.pt"):
                f.unlink()
                print(f"Removed old weights: {f}")
        
        # Train with YOLO's built-in early stopping
        print(f"\n{'='*70}")
        print("TRAINING START")
        print(f"Max epochs: {epochs}, Early stopping patience: {patience}")
        print(f"{'='*70}\n")
        
        model = YOLO("yolo11s-cls.pt")
        model.train(
            data=str(self.dataset),
            epochs=epochs,
            patience=patience,  # YOLO's early stopping based on val metrics
            imgsz=224,
            batch=32,
            project=f"{self.name}_model",
            name="train",
            exist_ok=True,
            verbose=True,
            plots=True,
        )
        
        # Copy best model
        if self.model_path.exists():
            shutil.copy(self.model_path, self.best_model_path)
            print(f"\nBest model saved to: {self.best_model_path}")
        
        # Final evaluation on validation set
        print(f"\n{'='*70}")
        print("FINAL EVALUATION ON VALIDATION SET")
        print(f"{'='*70}")
        self.evaluate(val_imgs, val_labels)
    
    def evaluate(self, images: List[np.ndarray], labels: List[str], model_path: Optional[str] = None):
        """Evaluate model on given images."""
        path = Path(model_path) if model_path else self.best_model_path
        if not path.exists():
            path = self.model_path
        if not path.exists():
            return print(f"Model not found: {path}")
        
        print(f"Evaluating model: {path}")
        model = YOLO(str(path))
        
        class_metrics, balanced_acc = self._compute_metrics(model, images, labels)
        
        # Print results
        print(f"\n{'Class':<20} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<8} {'N':<6}")
        print("-" * 70)
        for cls in self.classes:
            m = class_metrics.get(cls, {})
            if m:
                print(f"{cls:<20} {m['accuracy']:<10.1%} {m['precision']:<12.1%} {m['recall']:<10.1%} {m['f1']:<8.2f} {m['n_samples']:<6}")
        
        print("-" * 70)
        print(f"{'BALANCED ACCURACY':<20} {balanced_acc:<10.1%}")
