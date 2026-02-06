from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.dataloader import DataLoader
from easysort.trainer import YoloTrainer
from pathlib import Path
import numpy as np
import cv2


def convert_to_motion_imgs(img: np.ndarray) -> np.ndarray:
    print("Input size: ", img.shape)
    num_imgs = img.shape[0] / 545
    print("Number of images: ", num_imgs)
    img_1 = img[0:545, :]
    img_2 = img[545:1090, :]
    img_3 = img[1090:1635, :]
    print("Image 1 shape: ", img_1.shape)
    print("Image 2 shape: ", img_2.shape)
    print("Image 3 shape: ", img_3.shape)
    # save img 1 and 3
    cv2.imwrite("img_1.jpg", img_1)
    cv2.imwrite("img_3.jpg", img_3)
    assert False, "stop here"
    

if __name__ == "__main__":
    # Train motion model
    registry = RegistryBase(base=REGISTRY_LOCAL_IP)
    MOTION_CATEGORIES = ["motion", "no_motion"]

    dataloader = DataLoader(registry, classes=MOTION_CATEGORIES, destination=Path("verdis_motion_dataset_reworked"), force_recreate=True)
    dataloader.from_yolo_dataset(Path("verdis_motion_dataset"), convert_function=convert_to_motion_imgs)
    image = dataloader.sample_image()
    print(image.shape)

    trainer = YoloTrainer("verdis_motion_reworked", MOTION_CATEGORIES, dataloader=dataloader)
    trainer.train(epochs=10, patience=5, imgsz=224, batch=32)

    trainer.eval()