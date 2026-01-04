from typing import ClassVar, overload, TypeVar, Any
import functools, os
from dotenv import load_dotenv
import datetime
from typing import Callable
from pathlib import Path

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

class Sort:
    @staticmethod
    def since(data: list[str], date: datetime.datetime) -> list[str]:
      for item in data:
        elements = item.split("/")
        year, month, day = elements[-5], elements[-4], elements[-3]
        if datetime.datetime(int(year), int(month), int(day)) >= date:
          yield item

    @staticmethod
    def before(data: list[str], date: datetime.datetime) -> list[str]:
      for item in data:
        elements = item.split("/")
        year, month, day = elements[-5], elements[-4], elements[-3]
        if datetime.datetime(int(year), int(month), int(day)) < date:
          yield item

    @staticmethod
    def unique_frames(frames): pass


DEBUG = ContextVar("DEBUG", 0)
DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH, REGISTRY_PATH = getenv("DATA_REGISTRY_PATH", Path("")), getenv("RESULTS_REGISTRY_PATH", Path("")), getenv("REGISTRY_PATH", Path(""))
SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET = getenv("SUPABASE_URL", ""), getenv("SUPABASE_KEY", ""), getenv("SUPABASE_DATA_REGISTRY_BUCKET", "")
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_REGION = getenv("AWS_ACCESS_KEY_ID", ""), getenv("AWS_SECRET_ACCESS_KEY", ""), getenv("AWS_S3_BUCKET", ""), getenv("AWS_REGION", "eu-north-1")
OPENAI_API_KEY = getenv("OPENAI_API_KEY", "")

class Concat:
  @staticmethod
  def weekly(videos: list[str], class_sorting_func: Callable[[str], str]) -> list[str]:
    pass