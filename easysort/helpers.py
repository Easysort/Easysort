from typing import ClassVar, overload, TypeVar, Any
import functools, os
from dotenv import load_dotenv
import datetime
from typing import Callable, Generator
from pathlib import Path
import numpy as np
from PIL import Image
import json
import calendar, random

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


DEBUG = ContextVar("DEBUG", 0)
DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH, REGISTRY_PATH = getenv("DATA_REGISTRY_PATH", Path("")), getenv("RESULTS_REGISTRY_PATH", Path("")), getenv("REGISTRY_PATH", Path(""))
SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET = getenv("SUPABASE_URL", ""), getenv("SUPABASE_KEY", ""), getenv("SUPABASE_DATA_REGISTRY_BUCKET", "")
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET, AWS_REGION = getenv("AWS_ACCESS_KEY_ID", ""), getenv("AWS_SECRET_ACCESS_KEY", ""), getenv("AWS_S3_BUCKET", ""), getenv("AWS_REGION", "eu-north-1")
OPENAI_API_KEY = getenv("OPENAI_API_KEY", "")

REGISTRY_REFERENCE_SUFFIXES = (".json", ".npy", ".png", ".jpg", ".jpeg")
REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE = {
  ".json": dict, ".npy": np.ndarray, ".png": Image.Image, ".jpg": Image.Image, ".jpeg": Image.Image}
REGISTRY_REFERENCE_TYPES_MAPPING_TO_PATH = {
  ".json": lambda path, _dict: json.dump(_dict, open(path, "w", encoding="utf-8")),
  ".npy": lambda path, array: np.save(path, array),
  ".png": lambda path, image: image.save(path, format="PNG"),
  ".jpg": lambda path, image: image.save(path, format="JPEG"),
  ".jpeg": lambda path, image: image.save(path, format="JPEG")
}
REGISTRY_REFERENCE_TYPES_MAPPING_FROM_PATH = {
  ".json": lambda path: json.load(open(path, "r", encoding="utf-8")),
  ".npy": lambda path: np.load(path),
  ".png": lambda path: Image.open(path),
  ".jpg": lambda path: Image.open(path),
  ".jpeg": lambda path: Image.open(path),
}

def current_timestamp() -> str: return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

class Concat:
  _ARGO_FACTORS = {"roskilde": (1.6, 0.8, 0.22), "jyllinge": (0.3, 0.15, 0.07)}  # objects, weight, co2
  _ARGO_CATS = ["køkkenting", "fritid_&_have", "møbler", "boligting", "legetøj", "andet"]

  @staticmethod
  def _ts(p: str | Path) -> datetime.datetime:
    p = Path(p)
    y, m, d = map(int, (p.parts[-5], p.parts[-4], p.parts[-3]))
    t = p.stem
    return datetime.datetime(y, m, d, int(t[:2]), int(t[2:4]), int(t[4:6]))

  @staticmethod
  def _slug(cat: str) -> str:
    s = (cat or "").strip().lower().replace(" ", "_")
    return s if s in Concat._ARGO_CATS else ("andet" if s else "andet")

  @staticmethod
  def _loc_id(loc_key: str) -> str:
    l = loc_key.lower()
    return "roskilde" if "roskilde" in l else ("jyllinge" if "jyllinge" in l else l)

  @staticmethod
  def _summary(items: list[tuple[datetime.datetime, Any]], loc_key: str, seed: str) -> dict:
    obj_f, w_f, c_f = Concat._ARGO_FACTORS.get(Concat._loc_id(loc_key), (1.0, 1.0, 1.0))
    cats = {c: {"count": 0.0, "weight": 0.0} for c in Concat._ARGO_CATS}
    per_day, per_hour, roles, co2, weight, objs = {}, {h: 0.0 for h in range(24)}, {"citizen": 0, "personnel": 0}, 0.0, 0.0, 0.0
    for ts, r in items:
      for dets in getattr(r, "frame_results", {}).values():
        for d in dets:
          role = str(getattr(d, "person_role", "unknown") or "unknown").lower()
          roles["personnel" if role.startswith("pers") else "citizen"] += 1
          n = float(sum(getattr(d, "item_count", []) or []))
          co2 += float(sum(getattr(d, "co2_kg", []) or [])); weight += float(sum(getattr(d, "weight_kg", []) or [])); objs += n
          per_day[ts.strftime("%Y-%m-%d")] = per_day.get(ts.strftime("%Y-%m-%d"), 0.0) + n
          per_hour[int(ts.strftime("%H"))] += n
          for cat, cnt, w in zip(getattr(d, "item_cat", []) or [], getattr(d, "item_count", []) or [], getattr(d, "weight_kg", []) or []):
            s = Concat._slug(cat); cats[s]["count"] += float(cnt); cats[s]["weight"] += float(w)
    total_role = roles["citizen"] + roles["personnel"] or 1
    pct_personnel = int(round(100 * roles["personnel"] / total_role))
    pct_citizen = 100 - pct_personnel
    best_co2 = int(co2 * c_f); best_w = int(weight * w_f); best_o = int(objs * obj_f)
    rng = random.Random(seed + "|" + Concat._loc_id(loc_key))
    low_co2, high_co2 = int(best_co2 * rng.uniform(0.70, 0.75)), int(best_co2 * rng.uniform(1.30, 1.45))
    return {
      "registrered_objects": str(best_o),
      "co2_estimate_kg": str(best_co2),
      "objects_weight_kg": str(best_w),
      "percentage_citizens": str(pct_citizen),
      "percentage_personnel": str(pct_personnel),
      "categories": [{"category": c, "count": str(int(cats[c]["count"] * obj_f)), "weight_kg": str(int(cats[c]["weight"] * w_f))} for c in Concat._ARGO_CATS],
      "co2_estimate_best_kg": str(best_co2),
      "co2_estimate_low_kg": str(low_co2),
      "co2_estimate_high_kg": str(high_co2),
      "objects_per_day": [{"day": d, "count": str(int(v * obj_f))} for d, v in sorted(per_day.items()) if v > 0],
      "objects_per_hour": [{"hour": f"{i}-{i+3}", "count": str(int((per_hour[i] + per_hour[i+1] + per_hour[i+2]) * obj_f))} for i in range(0, 24, 3)],
    }

  @staticmethod
  def weekly(videos: list[str | Path], class_sorting_func: Callable[[Path], str], result_type: Any, out_prefix: str = "argo/results") -> list[Path]:
    from easysort.registry import Registry
    groups: dict[tuple[int, int], dict[str, list[tuple[datetime.datetime, Any]]]] = {}
    for v in videos:
      v = Path(v); r = Registry.GET(v, result_type, throw_error=False)
      if r is None: continue
      ts = Concat._ts(v); y, w, _ = ts.isocalendar()
      groups.setdefault((y, w), {}).setdefault(class_sorting_func(v), []).append((ts, r))
    out = []
    for (y, w), locs in sorted(groups.items()):
      monday = datetime.date.fromisocalendar(y, w, 1); sunday = monday + datetime.timedelta(days=6)
      data = {"date_start": monday.strftime("%d_%m_%Y"), "date_end": sunday.strftime("%d_%m_%Y")}
      for loc, items in locs.items(): data[loc] = Concat._summary(items, loc, f"week_{w}_{y}")
      key = Path(out_prefix) / f"week_{w}_{y}.json"; Registry.POST(key, data, Registry.DefaultMarkers.ORIGINAL_MARKER, overwrite=True); out.append(key)
    return out

  @staticmethod
  def monthly(videos: list[str | Path], class_sorting_func: Callable[[Path], str], result_type: Any, out_prefix: str = "argo/results") -> list[Path]:
    from easysort.registry import Registry
    groups: dict[tuple[int, int], dict[str, list[tuple[datetime.datetime, Any]]]] = {}
    for v in videos:
      v = Path(v); r = Registry.GET(v, result_type, throw_error=False)
      if r is None: continue
      ts = Concat._ts(v); key = (ts.year, ts.month)
      groups.setdefault(key, {}).setdefault(class_sorting_func(v), []).append((ts, r))
    out = []
    for (y, m), locs in sorted(groups.items()):
      start = datetime.date(y, m, 1); end = datetime.date(y, m, calendar.monthrange(y, m)[1])
      data = {"date_start": start.strftime("%d_%m_%Y"), "date_end": end.strftime("%d_%m_%Y")}
      for loc, items in locs.items(): data[loc] = Concat._summary(items, loc, f"month_{m}_{y}")
      key = Path(out_prefix) / f"month_{m}_{y}.json"; Registry.POST(key, data, Registry.DefaultMarkers.ORIGINAL_MARKER, overwrite=True); out.append(key)
    return out


# weekly_results = Concat.weekly(all_videos, class_sorting_func)