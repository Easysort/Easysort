"""Argo Recycling Site Validator - validate GPT detections of people and items.

Shows camera-cropped video frames at detected frame indices alongside GPT's
detection (person_role, items, categories, weights, CO2). Navigate through
all detections across videos and validate each one.
"""

import sys, base64, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import requests as http_requests
from flask import Flask, jsonify, request
from tqdm import tqdm

from easysort.registry import RegistryBase, RegistryConnector
from easysort.helpers import current_timestamp, REGISTRY_LOCAL_IP, Concat
from easysort.sampler import Crop
from easyprod.scripts.argo.registry_types import (
  RECYCLING_OVERVIEW_RESULT, RECYCLING_OVERVIEW_DETECTION, RECYCLING_OVERVIEW_RESULT_ID,
)
from easysort.helpers import GEMINI_API_KEY
from easyprod.scripts.argo.argo import PROMPT as ARGO_PROMPT

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={GEMINI_API_KEY}"

CROPS = {
  "Argo-roskilde-03-01": Crop(x=574, y=108, w=255, h=604),
  "Argo-Jyllinge-Entrance-01": Crop(x=662, y=11, w=246, h=702),
  "Argo-koege": Crop(x=518, y=165, w=309, h=540),
  "Argo-kalundborg1": Crop(x=387, y=6, w=406, h=561),
  "Argo-kalundborg2": Crop(x=498, y=6, w=208, h=316),
}

CACHE_FILE = Path("argo_validation_cache.json")
ARGO_VALIDATION_ID = "c1257824-3b3b-42ac-8689-c9e944fbfee3"

ROLES = ["citizen", "personnel", "unknown"]
ITEM_CATS = ["Køkkenting", "Fritid & Have", "Møbler", "Boligting", "Legetøj", "Andet"]


@dataclass
class ArgoDetectionGT:
  frame_idx: int
  detection_idx: int
  role_correct: bool = True
  items_correct: bool = True
  is_false_positive: bool = False
  corrected_role: str = ""


@dataclass
class ArgoValidationResult:
  validations: List[ArgoDetectionGT] = field(default_factory=list)
  id: str = field(default_factory=lambda: ARGO_VALIDATION_ID)
  metadata: RegistryBase.BaseDefaultTypes.BASEMETADATA = field(
    default_factory=lambda: RegistryBase.BaseDefaultTypes.BASEMETADATA(model="human", created_at=current_timestamp())
  )


def _location_name(p: Path) -> str:
  d = p.parts[-6].lower()
  if "koege" in d: return "koege"
  if "roskilde" in d: return "roskilde"
  if "jyllinge" in d: return "jyllinge"
  if "kalundborg" in d: return "kalundborg"
  return d


def _crop_for(p: Path) -> Optional[Crop]:
  key = p.parts[-6] if len(p.parts) >= 6 else ""
  return CROPS.get(key)


def _extract_frames(registry: RegistryBase, video_path: Path, frame_idxs: list[int], crop: Optional[Crop] = None) -> dict[int, np.ndarray]:
  """Seek to specific frames in a video and return them as numpy arrays."""
  url = registry.backend.URL(video_path)
  cap = cv2.VideoCapture(url)
  if not cap.isOpened():
    return {}

  frames = {}
  sorted_idxs = sorted(set(frame_idxs))
  try:
    for target in sorted_idxs:
      cap.set(cv2.CAP_PROP_POS_FRAMES, target)
      ret, frame = cap.read()
      if not ret:
        continue
      if crop is not None:
        frame = frame[crop.y:crop.y + crop.h, crop.x:crop.x + crop.w]
      frames[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  finally:
    cap.release()
  return frames


class ArgoValidator:
  port = 8080

  def __init__(self, registry: RegistryBase, location: str | None = None, week: int = 0):
    self.registry = registry
    self.app = Flask(__name__)
    self._location = location
    self._week = week
    self._items: list[dict] = []
    self._idx = 0
    self._frame_cache: dict[str, dict[int, np.ndarray]] = {}
    self._validation_cache: dict[str, dict] = {}
    self._load_data()
    self._setup_routes()

  def _load_cache(self) -> dict[str, dict]:
    if CACHE_FILE.exists():
      try:
        return json.loads(CACHE_FILE.read_text())
      except Exception:
        pass
    return {}

  def _save_cache(self):
    CACHE_FILE.write_text(json.dumps(self._validation_cache, indent=2))

  def _load_data(self):
    self.registry.add_id(RECYCLING_OVERVIEW_RESULT, RECYCLING_OVERVIEW_RESULT_ID)
    self.registry.add_id(ArgoValidationResult, ARGO_VALIDATION_ID)

    listed = self.registry.LIST("argo", suffix=[".mp4"], return_all=True, check_exists_with_type=RECYCLING_OVERVIEW_RESULT)
    all_videos, with_results = listed[0], listed[1]

    # Filter by location and week
    def matches(p: Path) -> bool:
      ts = Concat._ts(p)
      if ts is None:
        return False
      if p.parts[-6] not in CROPS:
        return False
      if self._location and self._location.lower() not in _location_name(p):
        return False
      if self._week and ts.isocalendar()[1] != self._week:
        return False
      return True

    videos_with_results = [p for p in with_results if matches(Path(p))]
    print(f"Found {len(all_videos)} total argo videos, {len(with_results)} with results, {len(videos_with_results)} after filters")

    # Check which videos already have validation GT
    self._validation_cache = self._load_cache()
    validated_set = set(self._validation_cache.keys())

    # Build flat list of (video, frame_idx, detection_idx, detection) items
    items = []
    for vp in tqdm(videos_with_results, desc="Loading results"):
      video_path = Path(vp)
      video_key = str(video_path)

      if video_key in validated_set:
        continue

      try:
        result = self.registry.GET(video_path, RECYCLING_OVERVIEW_RESULT, throw_error=False)
      except Exception:
        continue
      if result is None or not result.frame_results:
        continue

      for frame_idx in sorted(result.frame_results.keys()):
        dets = result.frame_results[frame_idx]
        for det_idx, det in enumerate(dets):
          if not det.item_desc:
            continue
          items.append({
            "video": video_key,
            "frame_idx": frame_idx,
            "det_idx": det_idx,
            "detection": det,
            "location": _location_name(video_path),
            "ts": str(Concat._ts(video_path) or ""),
          })

    self._items = items
    print(f"Loaded {len(items)} detections to validate ({len(validated_set)} videos already validated)")

    # Count per location
    by_loc: dict[str, int] = {}
    for it in items:
      by_loc[it["location"]] = by_loc.get(it["location"], 0) + 1
    for loc in sorted(by_loc):
      print(f"  {loc}: {by_loc[loc]} detections")

  def _get_frame(self, video_key: str, frame_idx: int) -> Optional[np.ndarray]:
    if video_key not in self._frame_cache:
      video_path = Path(video_key)
      crop = _crop_for(video_path)
      # Collect all frame_idxs needed for this video
      needed = [it["frame_idx"] for it in self._items if it["video"] == video_key]
      self._frame_cache = {video_key: _extract_frames(self.registry, video_path, needed, crop)}
    cache = self._frame_cache.get(video_key, {})
    return cache.get(frame_idx)

  def _frame_b64(self, video_key: str, frame_idx: int) -> str:
    frame = self._get_frame(video_key, frame_idx)
    if frame is None:
      return ""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()

  def _det_to_dict(self, det: RECYCLING_OVERVIEW_DETECTION) -> dict:
    return {
      "person_role": det.person_role,
      "item_desc": det.item_desc,
      "item_cat": det.item_cat,
      "item_count": det.item_count,
      "weight_kg": [round(w, 2) for w in det.weight_kg],
      "co2_kg": [round(c, 2) for c in det.co2_kg],
      "personal_item": det.personal_item,
    }

  def _ask_gemini(self, image_b64: str) -> dict:
    """Send the image to Gemini Flash with the Argo prompt, return parsed detection."""
    payload = {
      "contents": [{
        "parts": [
          {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
          {"text": ARGO_PROMPT},
        ]
      }]
    }
    resp = http_requests.post(GEMINI_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
      text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
      text = text[:-3]
    return json.loads(text.strip())

  def _save_validation(self, video_key: str, frame_idx: int, det_idx: int, role_correct: bool, items_correct: bool, is_false_positive: bool, corrected_role: str):
    entry = {
      "frame_idx": frame_idx,
      "detection_idx": det_idx,
      "role_correct": role_correct,
      "items_correct": items_correct,
      "is_false_positive": is_false_positive,
      "corrected_role": corrected_role,
    }
    self._validation_cache.setdefault(video_key, {"validations": []})
    self._validation_cache[video_key]["validations"].append(entry)
    self._save_cache()

    # Save to registry
    video_path = Path(video_key)
    gt = ArgoValidationResult(validations=[
      ArgoDetectionGT(**v) for v in self._validation_cache[video_key]["validations"]
    ])
    try:
      self.registry.POST(video_path, gt, ArgoValidationResult, overwrite=True)
    except Exception as e:
      print(f"Warning: could not save to registry: {e}")

  def _setup_routes(self):
    @self.app.route("/")
    def index():
      return HTML

    @self.app.route("/next")
    def next_item():
      while self._idx < len(self._items):
        item = self._items[self._idx]
        b64 = self._frame_b64(item["video"], item["frame_idx"])
        if b64:
          return jsonify({
            "done": False,
            "idx": self._idx,
            "total": len(self._items),
            "video": item["video"],
            "frame_idx": item["frame_idx"],
            "det_idx": item["det_idx"],
            "location": item["location"],
            "timestamp": item["ts"],
            "detection": self._det_to_dict(item["detection"]),
            "b64": b64,
          })
        self._idx += 1
      return jsonify({"done": True, "total": len(self._items)})

    @self.app.route("/validate", methods=["POST"])
    def validate():
      data = request.json
      self._save_validation(
        video_key=data["video"],
        frame_idx=data["frame_idx"],
        det_idx=data["det_idx"],
        role_correct=data.get("role_correct", True),
        items_correct=data.get("items_correct", True),
        is_false_positive=data.get("is_false_positive", False),
        corrected_role=data.get("corrected_role", ""),
      )
      self._idx += 1
      return jsonify({"ok": True})

    @self.app.route("/gemini", methods=["POST"])
    def gemini():
      data = request.json
      b64 = self._frame_b64(data["video"], data["frame_idx"])
      if not b64:
        return jsonify({"error": "Could not load frame"}), 400
      try:
        result = self._ask_gemini(b64)
        return jsonify({"ok": True, "detection": result})
      except Exception as e:
        return jsonify({"error": str(e)}), 500

    @self.app.route("/skip", methods=["POST"])
    def skip():
      self._idx += 1
      return jsonify({"ok": True})

    @self.app.route("/back", methods=["POST"])
    def back():
      self._idx = max(0, self._idx - 1)
      return jsonify({"ok": True})

  def run(self, host="0.0.0.0"):
    loc_str = f", Location: {self._location}" if self._location else ""
    week_str = f", Week: {self._week}" if self._week else ""
    print(f"Argo Validator at http://{host}:{self.port}{loc_str}{week_str}")
    print(f"Detections to validate: {len(self._items)}")
    self.app.run(host=host, port=self.port, debug=False, threaded=True)


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Argo Recycling Validator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#1a1a1a;color:#fff;height:100vh;display:flex;flex-direction:column}
#main{flex:1;display:flex;overflow:hidden}
#img-panel{flex:1;display:flex;justify-content:center;align-items:center;background:#000;min-width:0}
#img-panel img{max-width:100%;max-height:100%;object-fit:contain}
#det-panel{width:360px;background:#222;padding:16px;overflow-y:auto;flex-shrink:0;border-left:1px solid #333}
#det-panel h3{color:#4fc3f7;margin-bottom:8px;font-size:14px}
#gemini-section{margin-top:14px;border-top:2px solid #4a3a6a;padding-top:10px;display:none}
#gemini-section h3{color:#bb86fc}
#gemini-section .role{color:#bb86fc}
#gemini-section .item{background:#2a2a3a}
#gemini-loading{color:#bb86fc;font-size:13px;padding:8px 0}
.role{font-size:20px;font-weight:700;margin:6px 0 12px;text-transform:capitalize}
.role.citizen{color:#4caf50}.role.personnel{color:#ff9800}.role.unknown{color:#888}
.item{background:#2a2a2a;padding:8px 10px;border-radius:4px;margin-bottom:6px;font-size:13px}
.item .cat{color:#aaa;font-size:11px}.item .weight{color:#8bc34a;font-size:11px;margin-left:8px}
.item .co2{color:#ff7043;font-size:11px;margin-left:8px}
.item .personal{color:#e040fb;font-size:11px;margin-left:8px}
.meta{color:#666;font-size:11px;margin-top:12px;border-top:1px solid #333;padding-top:8px}
.meta div{margin-bottom:2px}
.totals{background:#1a3a2a;padding:8px 10px;border-radius:4px;margin-top:10px;font-size:13px}
.totals .lbl{color:#888;font-size:11px}.totals .val{color:#8bc34a;font-weight:600}
#bar{padding:10px 16px;background:#252525;display:flex;gap:8px;align-items:center;flex-wrap:wrap;flex-shrink:0}
.btn{padding:8px 14px;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:500;color:#fff}
.accept{background:#2d7d46}.accept:hover{background:#3a9a5a}
.role-btn{background:#5a4a2a}.role-btn:hover{background:#7a6a3a}
.reject{background:#7d2d2d}.reject:hover{background:#9a3a3a}
.items-wrong{background:#5a3a5a}.items-wrong:hover{background:#7a4a7a}
.skip{background:#555}.skip:hover{background:#666}
.back{background:#444}.back:hover{background:#555}
kbd{background:#333;padding:2px 6px;border-radius:3px;font-size:11px;margin-right:4px}
#progress{color:#888;margin-left:auto;font-size:13px}
#done{display:none;font-size:24px;text-align:center;padding:48px}
#ld{color:#666;font-size:16px}
</style></head><body>
<div id="main">
  <div id="img-panel"><span id="ld">Loading...</span><img id="img" style="display:none"/></div>
  <div id="det-panel" style="display:none">
    <h3>DETECTION</h3>
    <div id="role-display"></div>
    <div id="items-list"></div>
    <div id="totals-display"></div>
    <div id="meta-display" class="meta"></div>
    <div id="gemini-section">
      <h3>GEMINI</h3>
      <div id="gemini-loading"></div>
      <div id="gemini-role"></div>
      <div id="gemini-items"></div>
      <div id="gemini-totals"></div>
    </div>
  </div>
</div>
<div id="bar">
  <button class="btn back" onclick="back()"><kbd>B</kbd>Back</button>
  <button class="btn accept" onclick="accept()"><kbd>A</kbd>Accept</button>
  <button class="btn role-btn" onclick="correctRole('citizen')"><kbd>C</kbd>Citizen</button>
  <button class="btn role-btn" onclick="correctRole('personnel')"><kbd>P</kbd>Personnel</button>
  <button class="btn items-wrong" onclick="itemsWrong()"><kbd>I</kbd>Items Wrong</button>
  <button class="btn reject" onclick="reject()"><kbd>R</kbd>Reject</button>
  <button class="btn skip" onclick="skip()"><kbd>S</kbd>Skip</button>
  <button class="btn" style="background:#4a3a6a" onclick="askGemini()"><kbd>Space</kbd>Gemini</button>
  <span id="progress">-/-</span>
</div>
<div id="done">All done!</div>
<script>
let cur = null;

function renderDet(det) {
  document.getElementById('role-display').innerHTML =
    `<div class="role ${det.person_role}">${det.person_role}</div>`;

  let html = '';
  for (let i = 0; i < det.item_desc.length; i++) {
    const personal = det.personal_item[i] ? '<span class="personal">personal</span>' : '';
    html += `<div class="item">
      <div>${det.item_desc[i]} <span style="color:#fff8">x${det.item_count[i]}</span></div>
      <div><span class="cat">${det.item_cat[i]}</span>
        <span class="weight">${det.weight_kg[i]} kg</span>
        <span class="co2">${det.co2_kg[i]} kg CO2</span>
        ${personal}</div></div>`;
  }
  document.getElementById('items-list').innerHTML = html || '<div style="color:#666">No items detected</div>';

  const tw = det.weight_kg.reduce((a, b) => a + b, 0);
  const tc = det.co2_kg.reduce((a, b) => a + b, 0);
  const ti = det.item_count.reduce((a, b) => a + b, 0);
  document.getElementById('totals-display').innerHTML =
    `<div class="totals"><span class="lbl">Total:</span> <span class="val">${ti} items, ${tw.toFixed(1)} kg, ${tc.toFixed(1)} kg CO2</span></div>`;
}

function render(data) {
  if (data.done) {
    document.getElementById('done').style.display = 'block';
    document.getElementById('bar').style.display = 'none';
    document.getElementById('main').style.display = 'none';
    return;
  }
  cur = data;
  document.getElementById('gemini-section').style.display = 'none';
  document.getElementById('img').src = 'data:image/jpeg;base64,' + data.b64;
  document.getElementById('img').style.display = 'block';
  document.getElementById('ld').style.display = 'none';
  document.getElementById('det-panel').style.display = 'block';
  renderDet(data.detection);
  document.getElementById('meta-display').innerHTML =
    `<div>Location: ${data.location}</div>
     <div>Time: ${data.timestamp}</div>
     <div>Frame: ${data.frame_idx}, Det #${data.det_idx + 1}</div>
     <div style="font-size:10px;color:#444;word-break:break-all;margin-top:4px">${data.video.split('/').slice(-2).join('/')}</div>`;
  document.getElementById('progress').textContent = `${data.idx + 1}/${data.total}`;
}

async function load() { render(await (await fetch('/next')).json()); }

async function accept() {
  if (!cur) return;
  await fetch('/validate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video: cur.video, frame_idx: cur.frame_idx, det_idx: cur.det_idx,
      role_correct: true, items_correct: true, is_false_positive: false, corrected_role: ''})});
  await load();
}

async function correctRole(role) {
  if (!cur) return;
  await fetch('/validate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video: cur.video, frame_idx: cur.frame_idx, det_idx: cur.det_idx,
      role_correct: false, items_correct: true, is_false_positive: false, corrected_role: role})});
  await load();
}

async function itemsWrong() {
  if (!cur) return;
  await fetch('/validate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video: cur.video, frame_idx: cur.frame_idx, det_idx: cur.det_idx,
      role_correct: true, items_correct: false, is_false_positive: false, corrected_role: ''})});
  await load();
}

async function reject() {
  if (!cur) return;
  await fetch('/validate', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video: cur.video, frame_idx: cur.frame_idx, det_idx: cur.det_idx,
      role_correct: false, items_correct: false, is_false_positive: true, corrected_role: ''})});
  await load();
}

async function askGemini() {
  if (!cur) return;
  const sec = document.getElementById('gemini-section');
  sec.style.display = 'block';
  document.getElementById('gemini-loading').textContent = 'Asking Gemini...';
  document.getElementById('gemini-role').innerHTML = '';
  document.getElementById('gemini-items').innerHTML = '';
  document.getElementById('gemini-totals').innerHTML = '';
  try {
    const resp = await fetch('/gemini', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({video: cur.video, frame_idx: cur.frame_idx})});
    const data = await resp.json();
    document.getElementById('gemini-loading').textContent = '';
    if (data.error) { document.getElementById('gemini-loading').textContent = 'Error: ' + data.error; return; }
    const det = data.detection;
    document.getElementById('gemini-role').innerHTML =
      `<div class="role ${det.person_role}">${det.person_role}</div>`;
    let html = '';
    const descs = det.item_desc || [];
    const cats = det.item_cat || [];
    const counts = det.item_count || [];
    const weights = det.weight_kg || [];
    const co2s = det.co2_kg || [];
    const personals = det.personal_item || [];
    for (let i = 0; i < descs.length; i++) {
      const personal = personals[i] ? '<span class="personal">personal</span>' : '';
      html += `<div class="item">
        <div>${descs[i]} <span style="color:#fff8">x${counts[i]||1}</span></div>
        <div><span class="cat">${cats[i]||''}</span>
          <span class="weight">${weights[i]||0} kg</span>
          <span class="co2">${co2s[i]||0} kg CO2</span>
          ${personal}</div></div>`;
    }
    document.getElementById('gemini-items').innerHTML = html || '<div style="color:#666">No items</div>';
    const tw = weights.reduce((a, b) => a + b, 0);
    const tc = co2s.reduce((a, b) => a + b, 0);
    const ti = counts.reduce((a, b) => a + b, 0);
    document.getElementById('gemini-totals').innerHTML =
      `<div class="totals"><span class="lbl">Total:</span> <span class="val">${ti} items, ${tw.toFixed(1)} kg, ${tc.toFixed(1)} kg CO2</span></div>`;
  } catch (e) { document.getElementById('gemini-loading').textContent = 'Error: ' + e.message; }
}

async function skip() { await fetch('/skip', {method: 'POST'}); await load(); }
async function back() { await fetch('/back', {method: 'POST'}); await load(); }

document.addEventListener('keydown', e => {
  if (e.key.toLowerCase() === 'a') accept();
  else if (e.key.toLowerCase() === 'c') correctRole('citizen');
  else if (e.key.toLowerCase() === 'p') correctRole('personnel');
  else if (e.key.toLowerCase() === 'i') itemsWrong();
  else if (e.key.toLowerCase() === 'r') reject();
  else if (e.key.toLowerCase() === 's') skip();
  else if (e.key.toLowerCase() === 'b') back();
  else if (e.key === ' ') { e.preventDefault(); askGemini(); }
  else if (e.key === 'ArrowRight') skip();
  else if (e.key === 'ArrowLeft') back();
});

load();
</script></body></html>"""


if __name__ == "__main__":
  location = None
  week = 0
  for i, arg in enumerate(sys.argv):
    if arg == "--location" and i + 1 < len(sys.argv):
      location = sys.argv[i + 1]
    elif arg == "--week" and i + 1 < len(sys.argv):
      week = int(sys.argv[i + 1])

  if "--help" in sys.argv or "-h" in sys.argv:
    print("Usage: python -m easysort.validators.argo [OPTIONS]")
    print("\nValidate GPT detections of people and items at Argo recycling sites.")
    print("Shows camera-cropped video frames alongside GPT detection info.")
    print("\nOptions:")
    print("  --location NAME    Filter by location (roskilde, jyllinge, koege, kalundborg)")
    print("  --week N           Filter by ISO week number")
    print("  --help, -h         Show this help message")
    print("\nControls:")
    print("  A         Accept detection (role + items correct)")
    print("  C         Correct role to citizen")
    print("  P         Correct role to personnel")
    print("  I         Mark items as wrong (role still correct)")
    print("  R         Reject as false positive")
    print("  Space     Ask Gemini for a second opinion")
    print("  S / →     Skip")
    print("  B / ←     Go back")
    sys.exit(0)

  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  validator = ArgoValidator(registry, location=location, week=week)
  validator.run()
