from tinygrad.tinygrad import tensor
from tinygrad.examples.yolov8 import YOLOv8
from tinygrad.tinygrad.nn.state import safe_save, get_state_dict

## Load Dataset
## Setup training
## Run validation and save model

def save(net, save_path): state_dict = get_state_dict(net); safe_save(state_dict, save_path)