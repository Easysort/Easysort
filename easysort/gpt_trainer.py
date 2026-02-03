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
    def __init__(self, name: str, classes: List[str]):
        self.name = name
        self.classes = classes
        self.dataset = Path(f"{name}_dataset")
        self.model_path = Path(f"{name}_model/train/weights/best.pt")
        self.best_model_path = Path(f"{name}_model/best_balanced.pt")
    
    def _to_bgr(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    
    def _stratified_split(self, images: List[np.ndarray], labels: List[str], val_ratio: float = 0.2) -> Tuple[List, List, List, List]:
        """Split data maintaining class distribution in both train and val sets."""
        # Group indices by class
        class_indices: Dict[str, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            if label in self.classes:
                class_indices[label].append(i)
        
        train_indices, val_indices = [], []
        
        # For each class, take val_ratio for validation
        for cls, indices in class_indices.items():
            random.shuffle(indices)
            n_val = max(1, int(len(indices) * val_ratio))  # At least 1 sample for val
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
        
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        train_imgs = [images[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_imgs = [images[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        return train_imgs, train_labels, val_imgs, val_labels
    
    def _compute_metrics(self, model: YOLO, images: List[np.ndarray], labels: List[str]) -> Tuple[Dict[str, Dict], float]:
        """Compute per-class metrics and balanced accuracy."""
        preds = []
        for img in images:
            result = model.predict(self._to_bgr(img), verbose=False)[0]
            preds.append(model.names[int(result.probs.top1)])
        
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
    
    def _print_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, train_bal_acc: float, val_bal_acc: float, is_best: bool):
        """Print per-category metrics for an epoch."""
        best_marker = " *** BEST ***" if is_best else ""
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}{best_marker}")
        print(f"{'='*60}")
        print(f"\n{'Class':<18} {'Train Acc':<12} {'Val Acc':<12} {'Val Prec':<12} {'Val Rec':<12} {'Val F1':<10} {'N(val)':<8}")
        print("-" * 90)
        
        for cls in self.classes:
            t = train_metrics.get(cls, {})
            v = val_metrics.get(cls, {})
            train_acc = f"{t.get('accuracy', 0):.1%}" if t else "N/A"
            val_acc = f"{v.get('accuracy', 0):.1%}" if v else "N/A"
            val_prec = f"{v.get('precision', 0):.1%}" if v else "N/A"
            val_rec = f"{v.get('recall', 0):.1%}" if v else "N/A"
            val_f1 = f"{v.get('f1', 0):.2f}" if v else "N/A"
            n_val = v.get('n_samples', 0)
            print(f"{cls:<18} {train_acc:<12} {val_acc:<12} {val_prec:<12} {val_rec:<12} {val_f1:<10} {n_val:<8}")
        
        print("-" * 90)
        print(f"{'BALANCED ACC':<18} {train_bal_acc:<12.1%} {val_bal_acc:<12.1%}")
        print()
    
    def train(self, images: List[np.ndarray], labels: List[str], epochs: int = 20, 
              patience: int = 5, min_epochs: int = 3):
        """
        Train with stratified validation, per-epoch evaluation, and early stopping.
        
        Args:
            images: Training images
            labels: Training labels
            epochs: Maximum number of epochs
            patience: Stop training if no improvement for this many epochs
            min_epochs: Minimum epochs before early stopping can trigger
        """
        # Stratified split
        train_imgs, train_labels, val_imgs, val_labels = self._stratified_split(images, labels)
        
        # Print split distribution
        print(f"\n{'='*60}")
        print(f"DATASET SPLIT - {self.name}")
        print(f"{'='*60}")
        print(f"\n{'Class':<20} {'Train':<10} {'Val':<10} {'Total':<10}")
        print("-" * 50)
        for cls in self.classes:
            n_train = sum(1 for l in train_labels if l == cls)
            n_val = sum(1 for l in val_labels if l == cls)
            print(f"{cls:<20} {n_train:<10} {n_val:<10} {n_train + n_val:<10}")
        print("-" * 50)
        print(f"{'TOTAL':<20} {len(train_labels):<10} {len(val_labels):<10} {len(train_labels) + len(val_labels):<10}")
        print()
        
        # Create dataset directories and save images
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
        
        # Training with per-epoch validation
        best_val_balanced_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        
        print(f"\n{'='*60}")
        print("TRAINING START")
        print(f"Max epochs: {epochs}, Early stopping patience: {patience}, Min epochs: {min_epochs}")
        print(f"{'='*60}")
        
        for epoch in range(1, epochs + 1):
            print(f"\n>>> Training epoch {epoch}/{epochs}...")
            
            # Train for 1 epoch
            model = YOLO("yolo11s-cls.pt" if epoch == 1 else str(self.model_path.parent / "last.pt"))
            model.train(
                data=str(self.dataset),
                epochs=1,
                imgsz=224,
                batch=32,
                project=f"{self.name}_model",
                name="train",
                exist_ok=True,
                verbose=False,
                plots=False,
            )
            
            # Load the latest model for evaluation
            current_model_path = Path(f"{self.name}_model/train/weights/last.pt")
            if not current_model_path.exists():
                print(f"Warning: Model not found at {current_model_path}")
                continue
            
            eval_model = YOLO(str(current_model_path))
            
            # Evaluate on train and val
            train_metrics, train_bal_acc = self._compute_metrics(eval_model, train_imgs, train_labels)
            val_metrics, val_bal_acc = self._compute_metrics(eval_model, val_imgs, val_labels)
            
            # Check if this is the best model
            is_best = val_bal_acc > best_val_balanced_acc
            if is_best:
                best_val_balanced_acc = val_bal_acc
                best_epoch = epoch
                epochs_without_improvement = 0
                # Save as best model
                shutil.copy(current_model_path, self.best_model_path)
                shutil.copy(current_model_path, self.model_path)  # Also update the standard best.pt
            else:
                epochs_without_improvement += 1
            
            # Print metrics
            self._print_epoch_metrics(epoch, train_metrics, val_metrics, train_bal_acc, val_bal_acc, is_best)
            
            # Check for overfitting warning
            if train_bal_acc - val_bal_acc > 0.15:
                print(f"⚠️  OVERFITTING WARNING: Train-Val gap = {train_bal_acc - val_bal_acc:.1%}")
            
            # Early stopping check
            if epoch >= min_epochs and epochs_without_improvement >= patience:
                print(f"\n🛑 EARLY STOPPING: No improvement for {patience} epochs")
                print(f"   Best epoch: {best_epoch} with balanced accuracy: {best_val_balanced_acc:.1%}")
                break
        
        # Final summary
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best model saved at: {self.best_model_path}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation balanced accuracy: {best_val_balanced_acc:.1%}")
        
        # Final evaluation with best model
        print("\n>>> Final evaluation with best model:")
        best_model = YOLO(str(self.best_model_path))
        val_metrics, val_bal_acc = self._compute_metrics(best_model, val_imgs, val_labels)
        self._print_epoch_metrics(best_epoch, {}, val_metrics, 0, val_bal_acc, True)
    
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
