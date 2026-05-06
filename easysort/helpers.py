from typing import ClassVar, overload, TypeVar, Any
import functools, os, datetime
from dotenv import load_dotenv
from typing import Generator
from pathlib import Path
import json
import io
import numpy as np
from PIL import Image

T = TypeVar("T")
load_dotenv()

@overload
def getenv(key:str) -> int: ...
@overload
def getenv(key:str, default:T) -> T: ...
@functools.cache
def getenv(key:str, default:Any=0): return type(default)(os.getenv(key, default))

class ContextVar:
    _cache: ClassVar[dict[str, "ContextVar"]] = {}
    value: int
    key: str
    def __init__(self, key, default_value):
      if key in ContextVar._cache: raise RuntimeError(f"attempt to recreate ContextVar {key}")
      ContextVar._cache[key] = self
      self.value, self.key = getenv(key, default_value), key
    def __bool__(self): return bool(self.value)
    def __ge__(self, x): return self.value >= x
    def __gt__(self, x): return self.value > x
    def __lt__(self, x): return self.value < x

class Sort: # Expects paths like: ../2025/12/10/08/photo_20251210T082225Z.jpg
    @staticmethod
    def after(data: list[Path], date: datetime.datetime) -> Generator[Path, None, None]:
      for it in data:
        year, month, day, hour, minute, second = it.parts[-5], it.parts[-4], it.parts[-3], it.parts[-1][:2], it.parts[-1][2:4], it.parts[-1][4:6]
        if datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)) > date: yield it

    @staticmethod
    def before(data: list[Path], date: datetime.datetime) -> Generator[Path, None, None]:
      for it in data:
        year, month, day, hour, minute, second = it.parts[-5], it.parts[-4], it.parts[-3], it.parts[-1][:2], it.parts[-1][2:4], it.parts[-1][4:6]
        if datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)) < date: yield it


DEBUG, ON_LOCAL_CLOUD = ContextVar("DEBUG", 0), ContextVar("ON_LOCAL_CLOUD", 0) # ON_LOCAL_CLOUD: Everything must be run locally, on this device.
REGISTRY_PATH = getenv("REGISTRY_PATH", Path(""))
REGISTRY_LOCAL_IP, REGISTRY_TAILSCALE_IP = getenv("REGISTRY_LOCAL_IP", ""), getenv("REGISTRY_TAILSCALE_IP", "")
SUPABASE_URL, SUPABASE_KEY = getenv("SUPABASE_URL", ""), getenv("SUPABASE_KEY", "")
SUPABASE_DATA_BUCKETS: list[str] = [b for b in os.getenv("SUPABASE_DATA_BUCKETS", "").split(",") if b]
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY", "")

REGISTRY_REFERENCE_SUFFIXES = (".json", ".npy", ".png", ".jpg", ".jpeg")
RefData = dict | np.ndarray | Image.Image
REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE: dict[str, type[RefData]] = {
  ".json": dict, ".npy": np.ndarray, ".png": Image.Image, ".jpg": Image.Image, ".jpeg": Image.Image}

def _npy_to_bytes(array: np.ndarray) -> bytes:
  buf = io.BytesIO()
  np.save(buf, array)
  return buf.getvalue()
def _image_to_bytes(image: Image.Image, fmt: str) -> bytes:
  buf = io.BytesIO()
  image.save(buf, format=fmt)
  return buf.getvalue()

REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES = {
  ".json": lambda obj: json.dumps(obj, ensure_ascii=False).encode("utf-8"),
  ".npy": _npy_to_bytes,
  ".png": lambda image: _image_to_bytes(image, "PNG"),
  ".jpg": lambda image: _image_to_bytes(image, "JPEG"),
  ".jpeg": lambda image: _image_to_bytes(image, "JPEG"),
}
REGISTRY_REFERENCE_TYPES_MAPPING_FROM_BYTES = {
  ".json": lambda data: json.loads(data.decode("utf-8")),
  ".npy": lambda data: np.load(io.BytesIO(data)),
  ".png": lambda data: Image.open(io.BytesIO(data)),
  ".jpg": lambda data: Image.open(io.BytesIO(data)),
  ".jpeg": lambda data: Image.open(io.BytesIO(data)),
}

def current_timestamp() -> str: return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")