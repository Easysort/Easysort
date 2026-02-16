"""Verdis Bales Validator - annotate center points around bales."""

from easysort.registry import RegistryBase, RegistryConnector
from easysort.helpers import REGISTRY_LOCAL_IP, current_timestamp
from flask import Flask, request, jsonify
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
import base64, io, random, json

CACHE_FILE = Path("verdis_bales_gt_cache.json")


@dataclass
class VerdisBalesGroundTruth:
  points: List[List[float]]
  id: str = field(default_factory=lambda: "a5f6a27e-fc7c-41e3-b335-07e90f8f1320")
  metadata: Any = field(
    default_factory=lambda: RegistryBase.BaseDefaultTypes.BASEMETADATA(
      model="human",
      created_at=current_timestamp(),
    )
  )


class VerdisBalesValidator:
  prefix = "verdis/gadstrup/4"
  port = 8080

  def __init__(self, registry: RegistryBase, review_mode: bool = False, force_reload: bool = False, all_images: bool = False):
    self.registry, self.app = registry, Flask(__name__)
    self._idx = 0
    self._review_mode = review_mode
    self._force_reload = force_reload
    self._all_images = all_images
    self._setup_routes()

    self._gt_cache: Dict[str, List[List[float]]] = {}
    self._load_and_display_distributions()

    if review_mode:
      self._folders = self._get_review_folders()
      print(f"\n*** REVIEW MODE: Reviewing {len(self._folders)} existing ground truths ***\n")
    elif all_images:
      self._folders = sorted(self._all_folders, key=lambda f: self._folder_datetime.get(f, ("", "")))
      print(f"\n*** ALL IMAGES MODE: Showing {len(self._folders)} folders (no 15-min filtering) ***\n")
    else:
      self._folders = self._get_unlabeled_folders_sorted()

  def _load_gt_cache(self) -> Dict[str, List[List[float]]]:
    if not self._force_reload and CACHE_FILE.exists():
      try:
        return json.loads(CACHE_FILE.read_text())
      except:
        pass
    return {}

  def _save_gt_cache(self):
    CACHE_FILE.write_text(json.dumps(self._gt_cache, indent=2))

  def _extract_datetime_from_image(self, img_path: Path) -> Optional[Tuple[str, str]]:
    name = img_path.stem
    parts = name.split("_")
    for i, part in enumerate(parts):
      if len(part) == 8 and part.isdigit():
        if i + 1 < len(parts) and len(parts[i + 1]) >= 6 and parts[i + 1][:6].isdigit():
          return part, parts[i + 1][:6]
    return None

  def _load_and_display_distributions(self):
    files = self.registry.backend.LIST(self.prefix)
    self._files = files

    hash_lookup = self.registry._get_hash_lookup()
    gt_hash = hash_lookup.get(self.registry.get_id(VerdisBalesGroundTruth), "")
    gt_files = [f for f in files if gt_hash and gt_hash in f.name]

    print(f"\nFound {len(gt_files)} ground truth files")

    all_folder_imgs: Dict[Path, List[Path]] = {}
    for f in files:
      if f.suffix.lower() in (".jpg", ".png", ".jpeg"):
        all_folder_imgs.setdefault(f.parent, []).append(f)
    for folder in all_folder_imgs:
      all_folder_imgs[folder] = sorted(all_folder_imgs[folder])

    folder_datetime_all: Dict[Path, Tuple[str, str]] = {}
    folder_first_img: Dict[Path, Path] = {}
    for folder, imgs in all_folder_imgs.items():
      if not imgs:
        continue
      dt = self._extract_datetime_from_image(imgs[0])
      if dt is None:
        for img in imgs[1:3]:
          dt = self._extract_datetime_from_image(img)
          if dt:
            break
      if dt:
        folder_datetime_all[folder] = dt
        folder_first_img[folder] = imgs[0]

    self._folder_to_img = {}
    self._folder_to_all_imgs = {}
    self._folder_datetime: Dict[Path, Tuple[str, str]] = {}

    if self._all_images:
      for folder, imgs in all_folder_imgs.items():
        if folder in folder_first_img:
          self._folder_to_img[folder] = folder_first_img[folder]
          self._folder_to_all_imgs[folder] = imgs
          if folder in folder_datetime_all:
            self._folder_datetime[folder] = folder_datetime_all[folder]
      self._all_folders = sorted(self._folder_to_img.keys(), key=lambda f: self._folder_datetime.get(f, ("", "")))
      print(f"All images mode: {len(self._all_folders)} folders (no 15-min filtering)")
    else:

      def get_quarter_slot(time_str: str) -> int:
        minutes = int(time_str[2:4])
        if minutes < 15:
          return 0
        if minutes < 30:
          return 1
        if minutes < 45:
          return 2
        return 3

      slots: Dict[str, List[Tuple[Path, str]]] = {}
      for folder, (date_str, time_str) in folder_datetime_all.items():
        hour = time_str[:2]
        quarter = get_quarter_slot(time_str)
        slot_key = f"{date_str}-{hour}-{quarter}"
        slots.setdefault(slot_key, []).append((folder, time_str))

      for slot_key, folder_times in slots.items():
        folder_times.sort(key=lambda x: x[1])
        chosen_folder, chosen_time = folder_times[0]

        self._folder_to_img[chosen_folder] = folder_first_img[chosen_folder]
        self._folder_to_all_imgs[chosen_folder] = all_folder_imgs[chosen_folder]
        self._folder_datetime[chosen_folder] = folder_datetime_all[chosen_folder]

      skipped_count = len(all_folder_imgs) - len(self._folder_to_img)
      self._all_folders = sorted(self._folder_to_img.keys(), key=lambda f: self._folder_datetime.get(f, ("", "")))
      print(f"Filtered to {len(self._all_folders)} folders (1 per 15-min slot, skipped {skipped_count}, oldest first)")

    cached_gt = self._load_gt_cache()
    for folder_key, points in list(cached_gt.items()):
      if points and len(points[0]) == 4:
        cached_gt[folder_key] = [[(p[0] + p[2]) / 2, (p[1] + p[3]) / 2] for p in points]
    cached_folders = set(cached_gt.keys())

    gt_folder_to_file: Dict[str, Path] = {}
    for gt_file in gt_files:
      img_folder = str(gt_file.parent.parent)
      if img_folder not in gt_folder_to_file:
        gt_folder_to_file[img_folder] = gt_file

    new_gt_folders = [f for f in gt_folder_to_file.keys() if f not in cached_folders]
    print(f"GT Cache: {len(cached_folders)} cached, {len(new_gt_folders)} new to fetch")

    all_gt_cache = dict(cached_gt)
    for folder_str in tqdm(new_gt_folders, desc="Fetching new ground truth"):
      try:
        img_folder = Path(folder_str)
        imgs = sorted([f for f in self._files if f.parent == img_folder and f.suffix.lower() in (".jpg", ".png", ".jpeg")])
        if imgs:
          gt = self.registry.GET(imgs[0], VerdisBalesGroundTruth, throw_error=False)
          if gt:
            points = gt.points
            if points and len(points[0]) == 4:
              all_gt_cache[folder_str] = [[(p[0] + p[2]) / 2, (p[1] + p[3]) / 2] for p in points]
            else:
              all_gt_cache[folder_str] = points
      except:
        pass

    self._gt_cache = {k: v for k, v in all_gt_cache.items() if Path(k) in self._folder_to_img}

    old_gt_cache = self._gt_cache
    self._gt_cache = all_gt_cache
    self._save_gt_cache()
    self._gt_cache = old_gt_cache

    labeled_count = len(all_gt_cache)
    point_counts = [len(v) for v in all_gt_cache.values()]
    avg_points = sum(point_counts) / len(point_counts) if point_counts else 0

    print(f"\n{'=' * 50}")
    print("LABELED DATA DISTRIBUTION")
    print(f"{'=' * 50}")
    print(f"Total labeled folders: {labeled_count}/{len(gt_files)}")
    print(f"Total points annotated: {sum(point_counts)}")
    print(f"Average points per image: {avg_points:.1f}")
    print(f"{'=' * 50}\n")
    print(f"  ({len(self._gt_cache)} of these are in 15-minute filtered set)")

  def _get_unlabeled_folders_sorted(self) -> List[Path]:
    labeled_folders = set(self._gt_cache.keys())
    unlabeled = [f for f in self._all_folders if str(f) not in labeled_folders]
    unlabeled = sorted(unlabeled, key=lambda f: self._folder_datetime.get(f, ("", "")))

    print(f"Folders: {len(labeled_folders)} labeled, {len(unlabeled)} unlabeled, sorted oldest first")

    return unlabeled

  def _get_review_folders(self) -> List[Path]:
    labeled = [f for f in self._all_folders if str(f) in self._gt_cache]
    random.shuffle(labeled)
    return labeled

  def _get_images(self, folder: Path) -> List[Path]:
    files = self.registry.backend.LIST(str(folder))
    images = sorted([f for f in files if f.suffix.lower() in (".jpg", ".png", ".jpeg")])
    return images[:1]

  def _image_b64(self, key: Path) -> str:
    data = self.registry.GET(key, self.registry.DefaultMarkers.ORIGINAL_MARKER)
    if data is None:
      raise ValueError(f"Missing image data for {key}")
    buf = io.BytesIO()
    data.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

  def _get_image_size(self, key: Path) -> Tuple[int, int]:
    data = self.registry.GET(key, self.registry.DefaultMarkers.ORIGINAL_MARKER)
    if data is None:
      raise ValueError(f"Missing image data for {key}")
    return data.size

  def _save(self, folder: Path, points: List[List[float]]):
    images = self._get_images(folder)
    if images:
      gt = VerdisBalesGroundTruth(points=points)
      self.registry.POST(images[0], gt, VerdisBalesGroundTruth, overwrite=True)
      self._gt_cache[str(folder)] = points
      try:
        all_gt = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
      except:
        all_gt = {}
      all_gt[str(folder)] = points
      CACHE_FILE.write_text(json.dumps(all_gt, indent=2))
      print(f"Saved: {folder.name} -> {len(points)} points")

  def _setup_routes(self):
    @self.app.route("/")
    def index():
      return HTML

    @self.app.route("/next")
    def next_item():
      while self._idx < len(self._folders):
        folder = self._folders[self._idx]
        images = self._get_images(folder)
        if not images:
          self._idx += 1
          continue
        try:
          img_data = [{"key": str(img), "b64": self._image_b64(img), "size": self._get_image_size(img)} for img in images]
          existing_points = self._gt_cache.get(str(folder), [])
          response = {
            "done": False,
            "folder": str(folder),
            "idx": self._idx,
            "total": len(self._folders),
            "images": img_data,
            "points": existing_points,
            "review_mode": self._review_mode,
          }
          return jsonify(response)
        except Exception as e:
          print(f"Skipping {folder.name}: {e}")
          self._idx += 1
      return jsonify({"done": True, "total": len(self._folders)})

    @self.app.route("/save", methods=["POST"])
    def save():
      data = request.get_json(silent=True) or {}
      folder = data.get("folder")
      points = data.get("points")
      if not folder or points is None:
        return jsonify({"ok": False, "error": "Missing folder or points"}), 400
      self._save(Path(folder), points)
      self._idx += 1
      return jsonify({"ok": True})

    @self.app.route("/skip", methods=["POST"])
    def skip():
      self._idx += 1
      return jsonify({"ok": True})

    @self.app.route("/back", methods=["POST"])
    def back():
      self._idx = max(0, self._idx - 1)
      return jsonify({"ok": True})

  def run(self, host="0.0.0.0"):
    print(f"Verdis Bales Validator at http://{host}:{self.port}")
    self.app.run(host=host, port=self.port, debug=False, threaded=True)


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Verdis Bales Validator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#1a1a1a;color:#fff;height:100vh;display:flex;flex-direction:column}
#images{flex:1;display:flex;justify-content:center;align-items:center;position:relative;overflow:hidden;background:#000}
#canvas-container{position:relative;display:inline-block}
#canvas{cursor:crosshair;max-width:95vw;max-height:85vh;width:auto;height:auto}
#info-overlay{position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.85);padding:10px 14px;border-radius:6px;font-size:13px}
#bar{padding:12px;background:#222;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}
.btn{padding:8px 14px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:500}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.save{background:#2d7d46;color:#fff}.save:hover{background:#3d8a56}
.skip{background:#666}.back{background:#555}
.clear{background:#8a4a4a;color:#fff}.clear:hover{background:#9a5a5a}
.delete{background:#5a3a3a;color:#fff}.delete:hover{background:#6a4a4a}
#progress{color:#888;margin-left:auto}
#done{display:none;font-size:24px;text-align:center;padding:48px}
kbd{background:#333;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px}
.review-banner{background:#553;color:#ff8;padding:4px 12px;text-align:center;font-size:12px}
.instruction{color:#888;font-size:11px;margin-left:12px}
</style></head><body>
<div id="review-banner" class="review-banner" style="display:none">REVIEW MODE - Checking existing ground truths</div>
<div id="images">
  <div id="canvas-container">
    <canvas id="canvas"></canvas>
  </div>
  <div id="info-overlay"></div>
</div>
<div id="bar">
  <button class="btn back" onclick="back()"><kbd>B</kbd>Back</button>
  <button class="btn save" onclick="save()"><kbd>A</kbd>Save</button>
  <button class="btn clear" onclick="clearAll()"><kbd>C</kbd>Clear All</button>
  <button class="btn delete" onclick="deleteSelected()"><kbd>D</kbd>Delete Selected</button>
  <button class="btn skip" onclick="skip()"><kbd>S</kbd>Skip</button>
  <span class="instruction">Click to add point | Click point to select</span>
  <span id="progress">-/-</span>
</div>
<div id="done">All done!</div>
<script>
let currentFolder=null, currentImage=null, points=[], selectedIdx=-1;
let canvas, ctx, imgElement, cropRect=null;
// const CROP = {left: 0.2, right: 0.4, bottom: 0.4};
const CROP = {left: 0, right: 0, bottom: 0};

function initCanvas() {
  canvas = document.getElementById('canvas');
  ctx = canvas.getContext('2d');
  canvas.addEventListener('click', onClick);
}

function toCanvasCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  return {x, y};
}

function onClick(e) {
  const {x, y} = toCanvasCoords(e);
  let clickedIdx = -1;
  const hitRadius = 10;
  for (let i = points.length - 1; i >= 0; i--) {
    const [px, py] = points[i];
    if (!cropRect) continue;
    const cx = px - cropRect.x;
    const cy = py - cropRect.y;
    if (cx < 0 || cy < 0 || cx > cropRect.w || cy > cropRect.h) continue;
    const dx = x - cx;
    const dy = y - cy;
    if (Math.hypot(dx, dy) <= hitRadius) {
      clickedIdx = i;
      break;
    }
  }
  if (clickedIdx >= 0) {
    selectedIdx = clickedIdx;
  } else if (cropRect) {
    const origX = cropRect.x + x;
    const origY = cropRect.y + y;
    points.push([origX, origY]);
    selectedIdx = points.length - 1;
  }
  render();
}

function render() {
  if (!imgElement || !cropRect) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(
    imgElement,
    cropRect.x, cropRect.y, cropRect.w, cropRect.h,
    0, 0, canvas.width, canvas.height,
  );
  points.forEach((pt, i) => {
    const [px, py] = pt;
    const cx = px - cropRect.x;
    const cy = py - cropRect.y;
    if (cx < 0 || cy < 0 || cx > cropRect.w || cy > cropRect.h) return;
    ctx.beginPath();
    ctx.arc(cx, cy, i === selectedIdx ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = i === selectedIdx ? '#f00' : '#0f0';
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  const overlay = document.getElementById('info-overlay');
  overlay.innerHTML = `<div style="color:#4f8">Points: ${points.length}</div>${selectedIdx >= 0 ? '<div style="color:#f88;margin-top:4px">Selected: Bale ' + (selectedIdx + 1) + '</div>' : ''}`;
}

async function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

async function load() {
  const data = await (await fetch('/next')).json();
  renderData(data);
}

async function renderData(data) {
  if (data.done) {
    document.getElementById('done').style.display = 'block';
    document.getElementById('bar').style.display = 'none';
    document.getElementById('images').style.display = 'none';
    return;
  }
  currentFolder = data.folder;
  currentImage = data.images[0] || null;
  points = data.points || [];
  selectedIdx = -1;
  document.getElementById('review-banner').style.display = data.review_mode ? 'block' : 'none';
  await showImage();
  document.getElementById('progress').textContent = `${data.idx + 1}/${data.total}`;
}

async function showImage() {
  if (!currentImage) return;
  imgElement = await loadImage('data:image/jpeg;base64,' + currentImage.b64);
  const cropX = Math.round(imgElement.width * CROP.left);
  const cropY = Math.round(imgElement.height * (1 - CROP.bottom));
  const cropW = Math.round(imgElement.width * (1 - CROP.left - CROP.right));
  const cropH = Math.round(imgElement.height * CROP.bottom);
  if (cropW <= 0 || cropH <= 0) {
    cropRect = {x: 0, y: 0, w: imgElement.width, h: imgElement.height};
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;
  } else {
    cropRect = {x: cropX, y: cropY, w: cropW, h: cropH};
    canvas.width = cropW;
    canvas.height = cropH;
  }
  render();
}

async function save() {
  if (!currentFolder) return;
  await fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({folder: currentFolder, points: points})
  });
  await load();
}

async function skip() {
  await fetch('/skip', {method: 'POST'});
  await load();
}

async function back() {
  await fetch('/back', {method: 'POST'});
  await load();
}

function clearAll() {
  points = [];
  selectedIdx = -1;
  render();
}

function deleteSelected() {
  if (selectedIdx >= 0) {
    points.splice(selectedIdx, 1);
    selectedIdx = -1;
    render();
  }
}

document.addEventListener('keydown', e => {
  if (e.key.toLowerCase() === 'a') save();
  else if (e.key.toLowerCase() === 's') skip();
  else if (e.key.toLowerCase() === 'b') back();
  else if (e.key.toLowerCase() === 'c') clearAll();
  else if (e.key.toLowerCase() === 'd') deleteSelected();
  else if (e.key === 'Escape') { selectedIdx = -1; render(); }
});

initCanvas();
load();
</script></body></html>"""


if __name__ == "__main__":
  import sys

  review_mode = "--review" in sys.argv or "-r" in sys.argv
  force_reload = "--force-reload" in sys.argv
  all_images = "--all" in sys.argv

  if "--help" in sys.argv or "-h" in sys.argv:
    print("Usage: python verdis_bales.py [OPTIONS]")
    print("\nOptions:")
    print("  --review, -r              Review existing ground truths")
    print("  --force-reload            Force reload cache from registry")
    print("  --all                     Show all images (no 15-min filtering)")
    print("  --help, -h                Show this help message")
    print("\nControls:")
    print("  Click         Add point")
    print("  Click point   Select point")
    print("  D             Delete selected point")
    print("  C             Clear all points")
    print("  A             Save and next")
    print("  S             Skip")
    print("  B             Back")
    print("  Esc           Deselect")
    sys.exit(0)

  registry = RegistryBase(RegistryConnector(REGISTRY_LOCAL_IP))
  registry.add_id(VerdisBalesGroundTruth, "a5f6a27e-fc7c-41e3-b335-07e90f8f1320")
  validator = VerdisBalesValidator(registry, review_mode=review_mode, force_reload=force_reload, all_images=all_images)
  validator.run()
