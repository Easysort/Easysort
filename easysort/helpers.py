from typing import ClassVar, overload, TypeVar, Any
import functools, os
from dotenv import load_dotenv
import datetime
from typing import Callable, Generator
from pathlib import Path
import numpy as np
from PIL import Image
import json
import calendar, random, re

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


DEBUG, ON_LOCAL_CLOUD = ContextVar("DEBUG", 0), ContextVar("ON_LOCAL_CLOUD", 0) # ON_LOCAL_CLOUD: Everything must be run locally, on this device. 
DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH, REGISTRY_PATH = getenv("DATA_REGISTRY_PATH", Path("")), getenv("RESULTS_REGISTRY_PATH", Path("")), getenv("REGISTRY_PATH", Path(""))
REGISTRY_LOCAL_IP, REGISTRY_TAILSCALE_IP = getenv("REGISTRY_LOCAL_IP", ""), getenv("REGISTRY_TAILSCALE_IP", "")
SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET = getenv("SUPABASE_URL", ""), getenv("SUPABASE_KEY", ""), getenv("SUPABASE_DATA_REGISTRY_BUCKET", "")
OPENAI_API_KEY = getenv("OPENAI_API_KEY", "")

REGISTRY_REFERENCE_SUFFIXES = (".json", ".npy", ".png", ".jpg", ".jpeg")
REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE = {
  ".json": dict, ".npy": np.ndarray, ".png": Image.Image, ".jpg": Image.Image, ".jpeg": Image.Image}
import io

REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES = {
  ".json": lambda obj: json.dumps(obj, ensure_ascii=False).encode("utf-8"),
  ".npy": lambda array: (lambda buf=io.BytesIO(): (np.save(buf, array), buf.seek(0), buf.read())[2])(),
  ".png": lambda image: (lambda buf=io.BytesIO(): (image.save(buf, format="PNG"), buf.getvalue())[1])(),
  ".jpg": lambda image: (lambda buf=io.BytesIO(): (image.save(buf, format="JPEG"), buf.getvalue())[1])(),
  ".jpeg": lambda image: (lambda buf=io.BytesIO(): (image.save(buf, format="JPEG"), buf.getvalue())[1])(),
}
REGISTRY_REFERENCE_TYPES_MAPPING_FROM_BYTES = {
  ".json": lambda data: json.loads(data.decode("utf-8")),
  ".npy": lambda data: np.load(io.BytesIO(data)),
  ".png": lambda data: Image.open(io.BytesIO(data)),
  ".jpg": lambda data: Image.open(io.BytesIO(data)),
  ".jpeg": lambda data: Image.open(io.BytesIO(data)),
}

def current_timestamp() -> str: return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
def registry_file_to_local_file_path(registry_file: Path) -> Path: return registry_file.replace("/", "_")



class Concat:
  _ARGO_FACTORS = {"roskilde": (1.0, 0.6, 0.2), "jyllinge": (1.0, 0.6, 0.2)}  # objects, weight, co2
  _ARGO_CATS = ["køkkenting", "fritid_&_have", "møbler", "boligting", "legetøj", "andet"]
  _DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
  _HOURS = tuple(f"{i}-{i+3}" for i in range(0, 24, 3))

  @staticmethod
  def _ts(p: str | Path) -> datetime.datetime | None:
    p = Path(p)
    try:
      y, m, d = map(int, (p.parts[-5], p.parts[-4], p.parts[-3]))
      t = p.stem
      if len(t) != 6 or not t.isdigit(): return None
      return datetime.datetime(y, m, d, int(t[:2]), int(t[2:4]), int(t[4:6]))
    except Exception:
      return None

  @staticmethod
  def _slug(cat: str) -> str:
    s = (cat or "").strip().lower().replace(" ", "_")
    return s if s in Concat._ARGO_CATS else ("andet" if s else "andet")

  @staticmethod
  def _loc_id(loc_key: str) -> str:
    l = loc_key.lower()
    return "roskilde" if "roskilde" in l else ("jyllinge" if "jyllinge" in l else l)

  @staticmethod
  def _summary(items: list[tuple[datetime.datetime, Any]], loc_key: str, seed: str, filter_personal: bool | None = None) -> dict:
    """Generate summary. filter_personal: None=all items, True=only personal, False=only recycling."""
    obj_f, w_f, c_f = Concat._ARGO_FACTORS.get(Concat._loc_id(loc_key), (1.0, 1.0, 1.0))
    print(f"Using the following factors for {loc_key}: {obj_f}, {w_f}, {c_f}")
    cats = {c: {"count": 0.0, "weight": 0.0} for c in Concat._ARGO_CATS}
    per_day, per_hour, roles, co2, weight, objs = [0.0] * 7, {h: 0.0 for h in range(24)}, {"citizen": 0, "personnel": 0}, 0.0, 0.0, 0.0
    for ts, r in items:
      for dets in getattr(r, "frame_results", {}).values():
        for d in dets:
          role = str(getattr(d, "person_role", "unknown") or "unknown").lower()
          roles["personnel" if role.startswith("pers") else "citizen"] += 1
          personal_flags = getattr(d, "personal_item", []) or []
          item_cats = getattr(d, "item_cat", []) or []
          item_counts = getattr(d, "item_count", []) or []
          item_weights = getattr(d, "weight_kg", []) or []
          item_co2s = getattr(d, "co2_kg", []) or []
          for i, (cat, cnt, w, c) in enumerate(zip(item_cats, item_counts, item_weights, item_co2s)):
            if filter_personal is not None:
              is_personal = personal_flags[i] if i < len(personal_flags) else False
              if filter_personal and not is_personal: continue
              if not filter_personal and is_personal: continue
            co2 += float(c); weight += float(w); objs += float(cnt)
            per_day[int(ts.strftime("%w")) - 1] += float(cnt)
            per_hour[int(ts.strftime("%H"))] += float(cnt)
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
      "objects_per_day": [{"day": Concat._DAYS[i], "count": str(int(per_day[i] * obj_f))} for i in range(7)],
      "objects_per_hour": [{"hour": f"{i}-{i+3}", "count": str(int((per_hour[i] + per_hour[i+1] + per_hour[i+2]) * obj_f))} for i in range(0, 24, 3)],
    }

  @staticmethod
  def weekly(videos: list[str | Path], class_sorting_func: Callable[[Path], str], result_type: Any, out_prefix: str = "argo/results", filter_personal: bool | None = None) -> list[Path]:
    from easysort.registry import Registry
    groups: dict[tuple[int, int], dict[str, list[tuple[datetime.datetime, Any]]]] = {}
    for v in videos:
      v = Path(v); r = Registry.GET(v, result_type, throw_error=False)
      if r is None: continue
      ts = Concat._ts(v)
      if ts is None: continue
      y, w, _ = ts.isocalendar()
      groups.setdefault((y, w), {}).setdefault(class_sorting_func(v), []).append((ts, r))
    out = []
    suffix = "_personal" if filter_personal else ""
    for (y, w), locs in sorted(groups.items()):
      monday = datetime.date.fromisocalendar(y, w, 1); sunday = monday + datetime.timedelta(days=6)
      data = {"date_start": monday.strftime("%d_%m_%Y"), "date_end": sunday.strftime("%d_%m_%Y")}
      for loc, items in locs.items(): data[loc] = Concat._summary(items, loc, f"week_{w}_{y}", filter_personal)
      key = Path(out_prefix) / f"week_{w}_{y}{suffix}.json"; Registry.POST(key, data, Registry.DefaultMarkers.ORIGINAL_MARKER, overwrite=True); out.append(key)
    return out

  @staticmethod
  def monthly(videos: list[str | Path], class_sorting_func: Callable[[Path], str], result_type: Any, out_prefix: str = "argo/results", filter_personal: bool | None = None) -> list[Path]:
    from easysort.registry import Registry
    suffix = "_personal" if filter_personal else ""
    rx = re.compile(rf"^week_(\d+)_(\d+){suffix}(?:_force)?\.json$", re.I)
    best: dict[tuple[int, int], Path] = {}
    for f in Registry.LIST(out_prefix, suffix=[".json"]):
      m = rx.match(f.name)
      if not m: continue
      w, y = int(m.group(1)), int(m.group(2))
      if (y, w) not in best or f.stem.endswith("_force"): best[(y, w)] = f

    def _i(x) -> int:
      try: return int(float(x))
      except Exception: return 0

    day_idx = {d.lower(): i for i, d in enumerate(Concat._DAYS)}
    months: dict[tuple[int, int], list[tuple[datetime.date, datetime.date, dict]]] = {}
    for (y, w), f in best.items():
      try:
        mon = datetime.date.fromisocalendar(y, w, 1); sun = mon + datetime.timedelta(days=6)
      except Exception:
        continue
      try:
        wk = Registry.GET(f, Registry.DefaultMarkers.ORIGINAL_MARKER)
      except Exception:
        continue
      months.setdefault((sun.year, sun.month), []).append((mon, sun, wk))

    out: list[Path] = []
    for (y, m), weeks in sorted(months.items()):
      weeks.sort(key=lambda x: x[0])
      start, end = weeks[0][0], weeks[-1][1]
      data: dict[str, Any] = {"date_start": start.strftime("%d_%m_%Y"), "date_end": end.strftime("%d_%m_%Y")}
      acc: dict[str, Any] = {}
      for _, _, wk in weeks:
        for loc, locd in wk.items():
          if loc in ("date_start", "date_end") or not isinstance(locd, dict): continue
          a = acc.setdefault(loc, {"o": 0, "w": 0, "c": 0, "lo": 0, "hi": 0, "ppn": 0, "ppd": 0,
                                   "cats": {c: [0, 0] for c in Concat._ARGO_CATS}, "d": [0] * 7, "h": {k: 0 for k in Concat._HOURS}})
          o = _i(locd.get("registrered_objects")); a["o"] += o
          a["w"] += _i(locd.get("objects_weight_kg")); a["c"] += _i(locd.get("co2_estimate_best_kg") or locd.get("co2_estimate_kg"))
          a["lo"] += _i(locd.get("co2_estimate_low_kg")); a["hi"] += _i(locd.get("co2_estimate_high_kg"))
          a["ppn"] += _i(locd.get("percentage_personnel")) * o; a["ppd"] += o
          for row in (locd.get("categories") or []):
            if not isinstance(row, dict): continue
            cat = str(row.get("category", "")).strip().lower()
            if cat not in a["cats"]: continue
            a["cats"][cat][0] += _i(row.get("count")); a["cats"][cat][1] += _i(row.get("weight_kg"))
          for row in (locd.get("objects_per_day") or []):
            if not isinstance(row, dict): continue
            day = str(row.get("day", "")).strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
              try: idx = datetime.date.fromisoformat(day).weekday()
              except Exception: continue
            else:
              idx = day_idx.get(day.lower())
            if idx is not None: a["d"][idx] += _i(row.get("count"))
          for row in (locd.get("objects_per_hour") or []):
            if not isinstance(row, dict): continue
            hr = str(row.get("hour", "")).strip()
            if hr in a["h"]: a["h"][hr] += _i(row.get("count"))

      for loc, a in acc.items():
        pp = int(round(a["ppn"] / (a["ppd"] or 1)))
        data[loc] = {
          "registrered_objects": str(a["o"]),
          "co2_estimate_kg": str(a["c"]),
          "objects_weight_kg": str(a["w"]),
          "percentage_citizens": str(100 - pp),
          "percentage_personnel": str(pp),
          "categories": [{"category": c, "count": str(a["cats"][c][0]), "weight_kg": str(a["cats"][c][1])} for c in Concat._ARGO_CATS],
          "co2_estimate_best_kg": str(a["c"]),
          "co2_estimate_low_kg": str(a["lo"]),
          "co2_estimate_high_kg": str(a["hi"]),
          "objects_per_day": [{"day": Concat._DAYS[i], "count": str(a["d"][i])} for i in range(7)],
          "objects_per_hour": [{"hour": k, "count": str(a["h"][k])} for k in Concat._HOURS],
        }

      key = Path(out_prefix) / f"month_{m}_{y}{suffix}.json"
      Registry.POST(key, data, Registry.DefaultMarkers.ORIGINAL_MARKER, overwrite=True)
      out.append(key)
    return out


# weekly_results = Concat.weekly(all_videos, class_sorting_func)