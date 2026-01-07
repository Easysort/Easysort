from typing import ClassVar, overload, TypeVar, Any
import functools, os
from dotenv import load_dotenv
import datetime
from typing import Callable, Generator
from pathlib import Path
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
      for item in data:
        year, month, day, hour, minute, second = item.parts[-5], item.parts[-4], item.parts[-3], item.parts[-1][:2], item.parts[-1][2:4], item.parts[-1][4:6]
        if datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)) > date: yield item

    @staticmethod
    def before(data: list[Path], date: datetime.datetime) -> Generator[Path, None, None]:
      for item in data:
        year, month, day, hour, minute, second = item.parts[-5], item.parts[-4], item.parts[-3], item.parts[-1][:2], item.parts[-1][2:4], item.parts[-1][4:6]
        if datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)) < date: yield item


DEBUG, TESTING = ContextVar("DEBUG", 0), ContextVar("TESTING", 0)
DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH, REGISTRY_PATH = getenv("DATA_REGISTRY_PATH", Path("")), getenv("RESULTS_REGISTRY_PATH", Path("")), getenv("REGISTRY_PATH", Path(""))
SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET = getenv("SUPABASE_URL", ""), getenv("SUPABASE_KEY", ""), getenv("SUPABASE_DATA_REGISTRY_BUCKET", "")
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_REGION = getenv("AWS_ACCESS_KEY_ID", ""), getenv("AWS_SECRET_ACCESS_KEY", ""), getenv("AWS_S3_BUCKET", ""), getenv("AWS_REGION", "eu-north-1")
OPENAI_API_KEY = getenv("OPENAI_API_KEY", "")

REGISTRY_REFERENCE_TYPES = (np.ndarray, dict, Image.Image)

class Concat:
  @staticmethod
  def weekly(videos: list[str], class_sorting_func: Callable[[str], str]) -> list[str]:
    pass


# weekly_results = Concat.weekly(all_videos, class_sorting_func)