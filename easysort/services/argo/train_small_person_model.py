import neptune
from easysort.common.environment import Env
from pathlib import Path
from PIL import Image
from typing import Optional
import cv2
from tinygrad import Device
from easysort.services.argo.models.u2net import U2NET, U2NETP, U2NETPClassifier
from tinygrad import nn
from tinygrad.helpers import get_child
from tinygrad.engine.jit import TinyJit
import numpy as np
from tinygrad import Tensor
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from easysort.services.argo.models.header import TinyMaskPresenceNet
from tinygrad.nn.optim import Adam


def normPRED(d):
    ma, mi = d.max(), d.min()
    return (d-mi)/(ma-mi)

class Runner:
    def __init__(self, data_path: Path) -> None:
        assert data_path.exists(), f"Data path {data_path} does not exist"
        assert data_path.is_dir(), f"Data path {data_path} is not a directory"
        self.data_path = data_path
        self.logger: Optional[neptune.Run] = None
        self.image_paths = self.get_image_paths()
        print(f"Found {len(self.image_paths)} image paths")
        self.unit = None
    
    def load_u2net_model(self, path: Path) -> U2NET:
        assert path.exists(), f"Path {path} does not exist"
        assert path.is_file(), f"Path {path} is not a file"
        unet = U2NET(3,1)

        @TinyJit
        def jit_unet(x): return unet(x)

        loaded = nn.state.torch_load(path)
        for k, v in loaded.items():
            get_child(unet, k).assign(v.numpy()).realize()

        return jit_unet

    def run_unet(self) -> None:
        if self.unit is None: self.unit = self.load_u2net_model(Path("/Users/lucasvilsen/Documents/fun/easysort/easysort/services/argo/models/u2net_human_seg.pth"))
        print(f"Running U^2 Net on device: {Device.DEFAULT}.")
        for file in tqdm(self.image_paths):
            self.run_unet_single_im(file)

    def run_unet_single_im(self, file: Path) -> None:
        image = cv2.imread(str(file))
        if image is None: print("Image not found"); return
        image = cv2.resize(image, (320,320))
        image_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        pred = self._unit_inf(image_rotated)
        np.save(file.with_suffix('.fg_mask.npy'), pred)

    def _unit_inf(self, image: np.ndarray) -> np.ndarray:
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        image = image/np.max(image)

        tmpImg[:,:,0] = (image[:,:,2]-0.406)/0.225
        tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
        tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229

        # convert BGR to RGB
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpImg = tmpImg[np.newaxis,:,:,:]
        tmpTensor = Tensor(tmpImg.astype(np.float32))

        # inference
        d1,d2,d3,d4,d5,d6,d7= self.unit(tmpTensor)

        # normalization
        pred = 1.0 - d1[:,0,:,:]
        pred = normPRED(pred)

        # convert tinygrad tensor to numpy array
        pred = pred.squeeze()
        pred = pred.numpy()

        del d1,d2,d3,d4,d5,d6,d7

        return pred

    def get_image_paths(self) -> list[Path]:
        return [f for f in self.data_path.glob("**/*.jpg") if f.suffix != '.fg_mask.jpg' and not f.name.startswith(".")]

    def get_unet_image_paths(self) -> list[Path]:
        return [f for f in self.data_path.glob("**/*.fg_mask.npy") if not f.name.startswith(".")]

    def get_json_paths(self) -> list[Path]:
        return [f for f in self.data_path.glob("**/*.json") if not f.name.startswith(".")]

    def create_dataset(self) -> None:
        xs = []
        ys = []
        for image_path in tqdm(self.get_image_paths()):
            json_path = image_path.with_suffix('.json')
            if not json_path.exists(): continue
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, str):
                        try:
                            json_data = json.loads(json_data)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error loading JSON file {json_path}: {e}")
                continue
            try:
                num_people = None
                if isinstance(json_data, dict):
                    num_people = json_data.get("number_of_people")
                    if num_people is None:
                        num_people = json_data.get("total_people", 0)
                if num_people is None:
                    num_people = 0
                y = 1 if int(num_people) > 0 else 0
            except Exception:
                y = 0
            # load image as grayscale and resize to 256x256 for efficiency
            im = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
            x = (im.astype(np.float32) / 255.0)
            xs.append(x)
            ys.append(y)

        print(f"Created dataset with {len(xs)} images and {len(ys)} labels out of {len(self.get_image_paths())} images")

        train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.1, random_state=42)
        train_x = np.array(train_x, dtype=np.float32)
        test_x = np.array(test_x, dtype=np.float32)
        train_y = np.array(train_y, dtype=np.float32).reshape(-1, 1)
        test_y = np.array(test_y, dtype=np.float32).reshape(-1, 1)
        return train_x, train_y, test_x, test_y

    def train(self) -> None:
        train_x, train_y, test_x, test_y = self.create_dataset()
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        model = U2NETPClassifier(in_ch=1)
        optim = Adam(model.params)
        # self.logger = neptune.init_run(project='apros7/easysort', api_token=Env.NEPTUNE_API_TOKEN)
        
        def evaluate(eval_bs: int = 32) -> tuple[float, float]:
            losses = []
            correct = 0
            total = 0
            with Tensor.train(mode=False):
                for start in range(0, test_x.shape[0], eval_bs):
                    end = min(start + eval_bs, test_x.shape[0])
                    xb = Tensor(test_x[start:end])
                    yb = Tensor(test_y[start:end])
                    # use non-jitted forward during eval to avoid fusing large graphs
                    logits = model(xb)
                    loss = -(yb * logits.logsigmoid() + (1 - yb) * (-logits).logsigmoid()).mean()
                    probs = logits.sigmoid()
                    preds = (probs > 0.5)
                    correct += (preds == (yb > 0.5)).mean().numpy() * (end - start)
                    losses.append(loss.numpy())
                    total += (end - start)
            test_loss = float(np.mean(losses)) if losses else float('nan')
            acc = float(correct / max(total, 1))
            return test_loss, acc
        with Tensor.train():
            bs = 16
            steps = 1000
            for i in tqdm(range(steps), desc="Training"):
                samp = np.random.randint(0, train_x.shape[0], size=(bs))
                x = Tensor(train_x[samp], requires_grad=False)
                y = Tensor(train_y[samp])
                optim.zero_grad()
                # non-jitted forward to preserve autograd
                logits = model(x)
                loss = -(y * logits.logsigmoid() + (1 - y) * (-logits).logsigmoid()).mean()
                loss.backward()
                optim.step()
                if (i+1) % 50 == 0:
                    test_loss, acc = evaluate(eval_bs=bs)
                    print(f"Step {i+1} | TrainLoss: {loss.numpy():.4f} | TestLoss: {test_loss:.4f} | TestAcc: {acc:.4f}")
                else:
                    print(f"Step {i+1} | TrainLoss: {loss.numpy():.4f}")

if __name__ == "__main__":
    runner = Runner(data_path=Path("/Volumes/Easysort128/tmpddtx4_r6"))
    runner.train()