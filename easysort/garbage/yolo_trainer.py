from ultralytics import YOLO
from pathlib import Path
import numpy as np
from typing import List
import cv2

def create_dataset(images: List[np.ndarray], labels: List[bool], dataset_path: str, train_split: float = 0.8):
    """Create YOLO classification dataset from images and boolean labels."""
    dataset_path = Path(dataset_path)
    
    # Create directory structure
    for split in ['train', 'val']:
        for cls in ['class0', 'class1']:
            (dataset_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    indices = np.random.permutation(len(images))
    split_idx = int(len(images) * train_split)
    
    # Save images
    for i, orig_idx in enumerate(indices):
        img = images[orig_idx]
        label = labels[orig_idx]
        cls = 'class1' if label else 'class0'
        split = 'train' if i < split_idx else 'val'
        
        img_path = dataset_path / split / cls / f"{i:06d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    return str(dataset_path)

def train_classifier(images: List[np.ndarray], labels: List[bool], 
                     dataset_path: str = "yolo_dataset",
                     epochs: int = 50,
                     imgsz: int = 224,
                     batch: int = 32,
                     train_split: float = 0.8):
    """Train YOLOv8n classification model on images with boolean labels."""
    # Create dataset
    dataset_path = create_dataset(images, labels, dataset_path, train_split)
    
    # Load and train model
    model = YOLO('yolov8n-cls.pt')
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch
    )
    
    # Validate
    metrics = model.val()
    
    return model, metrics

