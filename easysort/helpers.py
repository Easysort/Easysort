from typing import ClassVar, overload, TypeVar, Any
import functools, os
from dotenv import load_dotenv

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

DEBUG = ContextVar("DEBUG", 0)
DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH = getenv("DATA_REGISTRY_PATH", ""), getenv("RESULTS_REGISTRY_PATH", "")
SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET = getenv("SUPABASE_URL", ""), getenv("SUPABASE_KEY", ""), getenv("SUPABASE_DATA_REGISTRY_BUCKET", "")
OPENAI_API_KEY = getenv("OPENAI_API_KEY", "")