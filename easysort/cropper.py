from easysort.sampler import Sampler, Crop
from easysort.registry import DataRegistry
import cv2

if __name__ == "__main__":

    # Adjust the crop to work well for the newly installed camera
    # Then add to sampler.DEVICE_TO_CROP
    DEVICE = "Argo-roskilde-03-01"
    crop = Crop(x=400, y=80, w=400, h=600)


    path = DataRegistry.LIST(f"argo/{DEVICE}")[0]
    frames = Sampler.unpack(path, crop=crop)
    cv2.imwrite(f"cropped_{DEVICE}_0.jpg", frames[0])
    print(f"saved to cropped_{DEVICE}_0.jpg")
    print("image size: ", frames[0].shape)
