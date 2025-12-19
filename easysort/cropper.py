from easysort.sampler import Sampler, Crop
from easysort.registry import Registry
import cv2

if __name__ == "__main__":

    # Adjust the crop to work well for the newly installed camera
    # Then add to sampler.DEVICE_TO_CROP
    DEVICE = "Argo-roskilde-03-01"
    crop = Crop(x=0, y=0, w=1000, h=1000)


    path = Registry._registry_path("argo/Argo-roskilde-03-01/2025/12/10/08/photo_20251210T082225Z.jpg")
    frames = Sampler.unpack(path, crop=crop)
    cv2.imwrite(f"cropped_{DEVICE}_0.jpg", frames[0])
    print(f"saved to cropped_{DEVICE}_0.jpg")
    print("image size: ", frames[0].shape)
