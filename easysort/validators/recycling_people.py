"""Manual validator for reviewing people across all frames in a recycling video."""

import base64
import functools
import random
import sys
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request
from tqdm import tqdm

from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.registry import RegistryBase
from easyprod.products.recycling.concat import Concat
from easyprod.products.recycling.helpers import RECYCLING_YOLO_DETECTION, RECYCLING_YOLO_RESULT, RECYCLING_YOLO_RESULT_ID, RecyclingVideo
from easyprod.products.recycling.people_validation import (
  DEFAULT_VALIDATION_CACHE_PATH,
  RECYCLING_PEOPLE_GT_ID,
  RecyclingPeopleGT,
  box_inside_ratio,
  choose_next_video,
  deduplicate_detections,
  grounding_dino_frame_bboxes,
  hour_from_video_path,
  list_recycling_videos_by_camera,
  load_people_validation_config,
  load_people_validation_cache,
  merge_frame_box_lists,
  normalise_box,
  save_people_validation_cache,
  update_people_validation_cache,
)


REJECTION_REASONS = ("size", "overlap", "duplicate")
REJECTION_REASON_LABELS = {
  "size": "too small (min_w/min_h)",
  "overlap": "overlaps a larger box (>80% inside)",
  "duplicate": "ReID duplicate in a nearby frame",
}


@functools.cache
def _proposal_trainer(weights_path: str, batch_size: int = 16):
  from easysort.trainer import Trainer
  from easysort.trainers.recycling_people import SCHEMA

  schema = replace(SCHEMA, weights_path=Path(weights_path))
  trainer = Trainer(schema)
  model = trainer.model
  optimize = getattr(model, "optimize_for_inference", None)
  if callable(optimize):
    try:
      print(f"[Validator] Optimizing model '{Path(weights_path).name}' for inference (batch_size={batch_size})...")
      optimize(batch_size=batch_size)
      print("[Validator] Model optimized for inference")
    except Exception as err:
      print(f"[Validator] optimize_for_inference failed, continuing unoptimized: {err}")
  return trainer


def _trained_model_frame_bboxes(
  frames: list[np.ndarray],
  weights_path: Path,
  *,
  batch_size: int = 16,
  conf_threshold: float = 0.25,
) -> dict[int, list[list[int]]]:
  trainer = _proposal_trainer(str(weights_path), batch_size)
  frame_bboxes: dict[int, list[list[int]]] = {}
  for start in tqdm(range(0, len(frames), batch_size), desc=f"Model proposals ({weights_path.name})"):
    batch = frames[start:start + batch_size]
    real_count = len(batch)
    if real_count < batch_size and real_count > 0:
      batch = list(batch) + [batch[-1]] * (batch_size - real_count)
    results = trainer.predict(batch)
    results = list(results)[:real_count]
    for offset, result in enumerate(results):
      frame_idx = start + offset
      boxes = getattr(result, "boxes", None)
      if boxes is not None:
        confs = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else [1.0] * len(boxes.xyxy)
        coords_iter = boxes.xyxy.tolist()
      elif hasattr(result, "xyxy"):
        coords_iter = result.xyxy.tolist() if hasattr(result.xyxy, "tolist") else list(result.xyxy)
        confs = (
          result.confidence.tolist()
          if getattr(result, "confidence", None) is not None and hasattr(result.confidence, "tolist")
          else list(getattr(result, "confidence", [1.0] * len(coords_iter)))
        )
      else:
        continue

      for coords, score in zip(coords_iter, confs):
        if float(score) < conf_threshold:
          continue
        valid = normalise_box(tuple(coords), frames[frame_idx].shape[1], frames[frame_idx].shape[0])
        if valid is not None:
          frame_bboxes.setdefault(frame_idx, []).append(valid)
  return frame_bboxes


def _frame_b64(frame: np.ndarray) -> str:
  _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
  return base64.b64encode(buf.tobytes()).decode()


def _sort_detections_with_reasons(video, detections: list, config, frames: list[np.ndarray]):
  rejected_by_reason: dict[str, list] = {reason: [] for reason in REJECTION_REASONS}

  passed_size = []
  for det in detections:
    if (det.x2 - det.x1) > config.min_w and (det.y2 - det.y1) > config.min_h:
      passed_size.append(det)
    else:
      rejected_by_reason["size"].append(det)

  by_frame: dict[int, list] = {}
  for det in passed_size:
    by_frame.setdefault(det.frame_idx, []).append(det)
  passed_overlap: list = []
  for frame_detections in by_frame.values():
    frame_detections.sort(key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
    kept: list = []
    for det in frame_detections:
      overlaps = any(
        box_inside_ratio((det.x1, det.y1, det.x2, det.y2), (k.x1, k.y1, k.x2, k.y2)) > 0.8
        for k in kept
      )
      if overlaps:
        rejected_by_reason["overlap"].append(det)
      else:
        kept.append(det)
    passed_overlap.extend(kept)

  final = deduplicate_detections(passed_overlap, frames) if passed_overlap else []
  final_ids = {id(det) for det in final}
  for det in passed_overlap:
    if id(det) not in final_ids:
      rejected_by_reason["duplicate"].append(det)

  return final, rejected_by_reason


def _rejected_frame_bboxes_with_reasons(
  rejected_by_reason: dict[str, list],
  frames: list[np.ndarray],
) -> dict[int, list[dict]]:
  rejected: dict[int, list[dict]] = {}
  for reason, detections in rejected_by_reason.items():
    for det in detections:
      if det.frame_idx < 0 or det.frame_idx >= len(frames):
        continue
      frame = frames[det.frame_idx]
      box = normalise_box((det.x1, det.y1, det.x2, det.y2), frame.shape[1], frame.shape[0])
      if box is None:
        continue
      rejected.setdefault(det.frame_idx, []).append({
        "box": box,
        "reason": reason,
        "reason_label": REJECTION_REASON_LABELS[reason],
      })
  return rejected


class RecyclingPeopleValidator:
  port = 8080

  def __init__(
    self,
    registry: RegistryBase,
    config_name: str | None = None,
    seed: int = 0,
    cache_path: Path | None = None,
    refresh_cache: bool = False,
    model_path: Path | None = None,
  ):
    self.registry = registry
    self.app = Flask(__name__)
    self._lock = Lock()
    self._rng = random.Random(seed)
    self._config_name = config_name
    self._config = load_people_validation_config(config_name)
    self._model_path = Path(model_path) if model_path is not None else None
    self._cache_path = cache_path or DEFAULT_VALIDATION_CACHE_PATH
    self._refresh_cache = refresh_cache
    self._label_cache = load_people_validation_cache(self._cache_path)
    self._cache_dirty = False
    self._remaining_by_camera: dict[str, list[Path]] = {camera: [] for camera in self._config.cameras}
    self._validated_counts: dict[str, int] = {camera: 0 for camera in self._config.cameras}
    self._validated_hour_counts: dict[int, int] = {hour: 0 for hour in range(24)}
    self._current_item: Optional[dict] = None
    self._session_saved = 0
    self._session_skipped = 0
    self._load_index()
    self._setup_routes()

  def _save_cache(self, force: bool = False):
    if not force and not self._cache_dirty:
      return
    save_people_validation_cache(self._label_cache, self._cache_path)
    self._cache_dirty = False

  def _set_cache_status(self, video_path: Path, camera: str, has_human_label: bool):
    self._cache_dirty = update_people_validation_cache(
      self._label_cache,
      video_path,
      camera,
      has_human_label,
      hour=hour_from_video_path(video_path),
    ) or self._cache_dirty

  def _load_index(self):
    print("Initializing recycling people validator index...")
    self.registry.add_id(RecyclingPeopleGT, RECYCLING_PEOPLE_GT_ID)
    self.registry.add_id(RECYCLING_YOLO_RESULT, RECYCLING_YOLO_RESULT_ID)
    check_exists_with_type = RECYCLING_YOLO_RESULT if self._config.proposal_source == "yolo_only" else None
    if check_exists_with_type is None:
      print("Validator indexing raw videos for selected cameras; YOLO proposals will be used only when available.")
    videos_by_camera = list_recycling_videos_by_camera(
      self.registry,
      self._config,
      check_exists_with_type=check_exists_with_type,
    )
    total_videos = sum(len(videos) for videos in videos_by_camera.values())
    print(f"Using cache file: {self._cache_path}")
    print(f"Checking existing human labels across {total_videos} videos...")
    cache_hits = 0
    cache_misses = 0
    for camera, videos in videos_by_camera.items():
      print(f"Scanning camera {camera}: {len(videos)} videos")
      cache_miss_videos: list[Path] = []
      for video_path in videos:
        cached_entry = None if self._refresh_cache else self._label_cache.get(str(video_path))
        if isinstance(cached_entry, dict) and "has_human_label" in cached_entry:
          cache_hits += 1
          if cached_entry.get("has_human_label"):
            self._validated_counts[camera] += 1
            hour = cached_entry.get("hour")
            if isinstance(hour, int) and 0 <= hour <= 23:
              self._validated_hour_counts[hour] += 1
            elif (resolved_hour := hour_from_video_path(video_path)) is not None:
              self._validated_hour_counts[resolved_hour] += 1
          else:
            self._remaining_by_camera[camera].append(video_path)
        else:
          cache_misses += 1
          cache_miss_videos.append(video_path)
      if cache_miss_videos:
        print(f"Cache miss for {camera}: checking {len(cache_miss_videos)} videos in registry")
      for video_path in tqdm(cache_miss_videos, desc=f"Checking labels for {camera}"):
        has_label = self.registry.EXISTS(video_path, RecyclingPeopleGT)
        self._set_cache_status(video_path, camera, has_label)
        if has_label:
          self._validated_counts[camera] += 1
          if (hour := hour_from_video_path(video_path)) is not None:
            self._validated_hour_counts[hour] += 1
        else:
          self._remaining_by_camera[camera].append(video_path)
      print(
        f"Finished {camera}: {self._validated_counts[camera]} labeled, "
        f"{len(self._remaining_by_camera[camera])} remaining"
      )
    print(f"Cache hits: {cache_hits}, cache misses: {cache_misses}")
    self._save_cache()
    counts = {camera: len(videos) for camera, videos in self._remaining_by_camera.items()}
    print(f"Loaded recycling people validator with remaining videos: {counts}")
    print(f"Labeled hour counts: {self._validated_hour_counts}")

  def _stats(self) -> list[dict]:
    stats = []
    for camera in self._config.cameras:
      stats.append({
        "camera": camera,
        "labeled": self._validated_counts.get(camera, 0),
        "remaining": len(self._remaining_by_camera.get(camera, [])),
      })
    return stats

  def _remove_video(self, camera: str, video_path: Path):
    self._remaining_by_camera[camera] = [path for path in self._remaining_by_camera.get(camera, []) if path != video_path]

  def _frame_payload(self, item: dict, frame_idx: int) -> Optional[dict]:
    if frame_idx < 0 or frame_idx >= item["frame_count"]:
      return None
    cached = item["frame_cache"].get(frame_idx)
    if cached is not None:
      return cached
    frame = item["video_obj"].frames[frame_idx]
    payload = {
      "frame_idx": frame_idx,
      "frame_w": frame.shape[1],
      "frame_h": frame.shape[0],
      "suggested_bboxes": item["suggested_by_frame"].get(frame_idx, []),
      "yolo_suggested_bboxes": item["yolo_suggested_by_frame"].get(frame_idx, []),
      "rejected_bboxes": item.get("rejected_by_frame", {}).get(frame_idx, []),
      "b64": _frame_b64(frame),
    }
    item["frame_cache"][frame_idx] = payload
    return payload

  def _build_item(self, camera: str, video_path: Path) -> Optional[dict]:
    print(f"Preparing video for review: {camera} / {video_path.name}")
    video = RecyclingVideo(video_path)
    all_frames = video.frames
    if not all_frames:
      print(f"Skipping {video_path.name}: video has no readable frames")
      return None

    rejected_by_frame: dict[int, list[dict]] = {}
    if self._model_path is not None:
      raw_model_frame_bboxes = _trained_model_frame_bboxes(all_frames, self._model_path)
      raw_model_count = sum(len(boxes) for boxes in raw_model_frame_bboxes.values())
      raw_model_detections: list[RECYCLING_YOLO_DETECTION] = []
      for frame_idx, boxes in raw_model_frame_bboxes.items():
        if frame_idx < 0 or frame_idx >= len(all_frames):
          continue
        for box in boxes:
          x1, y1, x2, y2 = box
          raw_model_detections.append(RECYCLING_YOLO_DETECTION(frame_idx=frame_idx, x1=x1, y1=y1, x2=x2, y2=y2))
      if raw_model_detections:
        sorted_model_detections, rejected_model_by_reason = _sort_detections_with_reasons(
          video, raw_model_detections, self._config, all_frames,
        )
        print(
          f"Sorted {len(raw_model_detections)} detections to {len(sorted_model_detections)} "
          f"(rejected: "
          f"size={len(rejected_model_by_reason['size'])}, "
          f"overlap={len(rejected_model_by_reason['overlap'])}, "
          f"duplicate={len(rejected_model_by_reason['duplicate'])})"
        )
      else:
        sorted_model_detections = []
        rejected_model_by_reason = {reason: [] for reason in REJECTION_REASONS}
      rejected_by_frame = _rejected_frame_bboxes_with_reasons(rejected_model_by_reason, all_frames)
      model_suggested_by_frame: dict[int, list[list[int]]] = {}
      for det in sorted_model_detections:
        if det.frame_idx < 0 or det.frame_idx >= len(all_frames):
          continue
        frame = all_frames[det.frame_idx]
        suggested_bbox = normalise_box((det.x1, det.y1, det.x2, det.y2), frame.shape[1], frame.shape[0])
        if suggested_bbox is None:
          continue
        model_suggested_by_frame.setdefault(det.frame_idx, []).append(suggested_bbox)
      yolo_suggested_by_frame = model_suggested_by_frame
      grounding_dino_suggested_by_frame: dict[int, list[list[int]]] = {}
      suggested_by_frame = merge_frame_box_lists(
        model_suggested_by_frame,
        iou_threshold=self._config.proposal_merge_iou,
      )
      model_count = sum(len(boxes) for boxes in model_suggested_by_frame.values())
      rejected_count = sum(len(boxes) for boxes in rejected_by_frame.values())
      merged_count = sum(len(boxes) for boxes in suggested_by_frame.values())
      print(
        f"Prepared full video review for {video_path.name}: "
        f"{len(all_frames)} frames, {raw_model_count} raw model boxes, "
        f"{model_count} model boxes after sorting, {rejected_count} rejected model boxes "
        f"(model-only mode, YOLO and Grounding DINO skipped), "
        f"{merged_count} merged boxes"
      )
    else:
      yolo_result = self.registry.GET(video.video_path, RECYCLING_YOLO_RESULT, throw_error=False)
      raw_detections = [] if yolo_result is None else list(yolo_result.detections)
      if raw_detections:
        detections, rejected_yolo_by_reason = _sort_detections_with_reasons(
          video, raw_detections, self._config, all_frames,
        )
        print(
          f"Sorted {len(raw_detections)} detections to {len(detections)} "
          f"(rejected: "
          f"size={len(rejected_yolo_by_reason['size'])}, "
          f"overlap={len(rejected_yolo_by_reason['overlap'])}, "
          f"duplicate={len(rejected_yolo_by_reason['duplicate'])})"
        )
      else:
        detections = []
        rejected_yolo_by_reason = {reason: [] for reason in REJECTION_REASONS}
      rejected_by_frame = _rejected_frame_bboxes_with_reasons(rejected_yolo_by_reason, all_frames)
      yolo_suggested_by_frame = {}
      for det in detections:
        if det.frame_idx < 0 or det.frame_idx >= len(all_frames):
          continue
        frame = all_frames[det.frame_idx]
        suggested_bbox = normalise_box((det.x1, det.y1, det.x2, det.y2), frame.shape[1], frame.shape[0])
        if suggested_bbox is None:
          continue
        yolo_suggested_by_frame.setdefault(det.frame_idx, []).append(suggested_bbox)

      grounding_dino_suggested_by_frame = grounding_dino_frame_bboxes(all_frames, self._config)
      if self._config.proposal_source == "grounding_dino_only":
        suggested_by_frame = merge_frame_box_lists(
          grounding_dino_suggested_by_frame,
          iou_threshold=self._config.proposal_merge_iou,
        )
      else:
        suggested_by_frame = merge_frame_box_lists(
          yolo_suggested_by_frame,
          grounding_dino_suggested_by_frame,
          iou_threshold=self._config.proposal_merge_iou,
        )

      yolo_count = sum(len(boxes) for boxes in yolo_suggested_by_frame.values())
      grounding_dino_count = sum(len(boxes) for boxes in grounding_dino_suggested_by_frame.values())
      rejected_count = sum(len(boxes) for boxes in rejected_by_frame.values())
      merged_count = sum(len(boxes) for boxes in suggested_by_frame.values())

      print(
        f"Prepared full video review for {video_path.name}: "
        f"{len(all_frames)} frames, {yolo_count} YOLO boxes, "
        f"{grounding_dino_count} Grounding DINO boxes, "
        f"{rejected_count} rejected YOLO boxes, "
        f"{merged_count} merged boxes"
      )
    timestamp = Concat._ts(video.video_path)
    return {
      "camera": camera,
      "video": str(video.video_path),
      "video_name": video.video_path.name,
      "timestamp": str(timestamp) if timestamp else "",
      "frame_count": len(all_frames),
      "suggested_by_frame": suggested_by_frame,
      "suggested_frame_indices": sorted(suggested_by_frame),
      "yolo_suggested_by_frame": yolo_suggested_by_frame,
      "yolo_frame_indices": sorted(yolo_suggested_by_frame),
      "rejected_by_frame": rejected_by_frame,
      "video_obj": video,
      "frame_cache": {},
    }

  def _next_item(self) -> Optional[dict]:
    while True:
      camera, video_path = choose_next_video(
        self._remaining_by_camera,
        self._validated_counts,
        labeled_hour_counts=self._validated_hour_counts,
        hour_targets=self._config.hour_targets,
        priority_hours=self._config.priority_hours,
        rng=self._rng,
      )
      if camera is None or video_path is None:
        return None
      item = self._build_item(camera, video_path)
      if item is not None:
        return item
      print(f"Skipping {video_path} in validator: no readable video content")
      self._remove_video(camera, video_path)
      self._session_skipped += 1

  def _payload(self, item: dict) -> dict:
    return {
      "done": False,
      "camera": item["camera"],
      "video": item["video"],
      "video_name": item["video_name"],
      "timestamp": item["timestamp"],
      "frame_count": item["frame_count"],
      "suggested_frame_count": len(item["suggested_frame_indices"]),
      "suggested_frame_indices": item["suggested_frame_indices"],
      "yolo_frame_count": len(item["yolo_frame_indices"]),
      "yolo_frame_indices": item["yolo_frame_indices"],
      "accept_yolo_by_default": self._config.accept_yolo_by_default,
      "stats": self._stats(),
      "hour_counts": self._validated_hour_counts,
      "priority_hours": self._config.priority_hours,
      "session_saved": self._session_saved,
      "session_skipped": self._session_skipped,
      "config_name": self._config_name or "",
      "model_path": str(self._model_path) if self._model_path else "",
    }

  def _setup_routes(self):
    @self.app.route("/")
    def index():
      return HTML

    @self.app.route("/next")
    def next_item():
      with self._lock:
        if self._current_item is None:
          self._current_item = self._next_item()
        if self._current_item is None:
          return jsonify({
            "done": True,
            "stats": self._stats(),
            "session_saved": self._session_saved,
            "session_skipped": self._session_skipped,
          })
        return jsonify(self._payload(self._current_item))

    @self.app.route("/frame/<int:frame_idx>")
    def frame_item(frame_idx: int):
      with self._lock:
        if self._current_item is None:
          return jsonify({"ok": False, "error": "No active video"}), 400
        payload = self._frame_payload(self._current_item, frame_idx)
        if payload is None:
          return jsonify({"ok": False, "error": "Invalid frame index"}), 400
        return jsonify({"ok": True, "frame": payload})

    @self.app.route("/save", methods=["POST"])
    def save():
      data = request.json or {}
      with self._lock:
        if self._current_item is None:
          return jsonify({"ok": False, "error": "No active video"}), 400
        if data.get("video") != self._current_item["video"]:
          return jsonify({"ok": False, "error": "Video mismatch"}), 400
        frame_bboxes = {}
        for frame_idx_raw, boxes in (data.get("frame_bboxes") or {}).items():
          frame_idx = int(frame_idx_raw)
          if frame_idx < 0 or frame_idx >= self._current_item["frame_count"]:
            continue
          frame = self._current_item["video_obj"].frames[frame_idx]
          valid_boxes = []
          for box in boxes:
            valid = normalise_box(tuple(box), frame.shape[1], frame.shape[0])
            if valid is not None:
              valid_boxes.append(valid)
          if valid_boxes:
            frame_bboxes[frame_idx] = valid_boxes
        video_path = Path(self._current_item["video"])
        gt = RecyclingPeopleGT(frame_bboxes=frame_bboxes)
        self.registry.POST(video_path, gt, RecyclingPeopleGT, overwrite=True)
        self._validated_counts[self._current_item["camera"]] += 1
        if (hour := hour_from_video_path(video_path)) is not None:
          self._validated_hour_counts[hour] += 1
        self._session_saved += 1
        self._set_cache_status(video_path, self._current_item["camera"], True)
        self._save_cache()
        self._remove_video(self._current_item["camera"], video_path)
        self._current_item = None
        return jsonify({"ok": True})

    @self.app.route("/accept_yolo", methods=["POST"])
    def accept_yolo():
      with self._lock:
        if self._current_item is None:
          return jsonify({"ok": False, "error": "No active video"}), 400
        frame_bboxes = {
          frame_idx: boxes
          for frame_idx, boxes in self._current_item["yolo_suggested_by_frame"].items()
          if boxes
        }
        if not frame_bboxes:
          return jsonify({"ok": False, "error": "No YOLO detections to accept"}), 400
        video_path = Path(self._current_item["video"])
        gt = RecyclingPeopleGT(frame_bboxes=frame_bboxes)
        self.registry.POST(video_path, gt, RecyclingPeopleGT, overwrite=True)
        self._validated_counts[self._current_item["camera"]] += 1
        if (hour := hour_from_video_path(video_path)) is not None:
          self._validated_hour_counts[hour] += 1
        self._session_saved += 1
        self._set_cache_status(video_path, self._current_item["camera"], True)
        self._save_cache()
        self._remove_video(self._current_item["camera"], video_path)
        self._current_item = None
        return jsonify({"ok": True})

    @self.app.route("/skip", methods=["POST"])
    def skip():
      with self._lock:
        if self._current_item is None:
          return jsonify({"ok": True})
        video_path = Path(self._current_item["video"])
        self._set_cache_status(video_path, self._current_item["camera"], False)
        self._save_cache()
        self._remove_video(self._current_item["camera"], video_path)
        self._current_item = None
        self._session_skipped += 1
        return jsonify({"ok": True})

  def run(self, host: str = "0.0.0.0"):
    print(f"Recycling People Validator at http://{host}:{self.port}")
    print(f"Cameras: {', '.join(self._config.cameras)}")
    self.app.run(host=host, port=self.port, debug=False, threaded=True)


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Recycling People Validator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui;background:#181818;color:#fff;height:100vh;display:flex;flex-direction:column}
#main{flex:1;display:flex;min-height:0}
#canvas-panel{flex:1;display:flex;justify-content:center;align-items:center;background:#000;padding:12px;min-width:0}
#canvas{max-width:100%;max-height:100%;border:1px solid #333;cursor:crosshair}
#side{width:380px;background:#222;border-left:1px solid #333;padding:14px;overflow-y:auto}
#side h2{font-size:18px;margin-bottom:10px;color:#6ec1ff}
#side h3{font-size:13px;margin:14px 0 6px;color:#888;text-transform:uppercase}
.meta{font-size:13px;line-height:1.5}
.meta div{margin-bottom:4px;word-break:break-word}
.note{font-size:12px;color:#bbb;line-height:1.45}
.stats{display:grid;gap:6px;margin-top:6px}
.stat{background:#2b2b2b;padding:8px 10px;border-radius:4px;font-size:12px}
.stat .camera{color:#fff;font-weight:600}
.stat .counts{color:#9ecbff}
.hour-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:6px}
.hour-pill{background:#2b2b2b;padding:6px 8px;border-radius:4px;font-size:11px;text-align:center}
.hour-pill.priority{outline:1px solid #ffb74d}
#bar{padding:10px 14px;background:#252525;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn{padding:8px 12px;border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:13px;font-weight:500}
.nav{background:#444}.nav:hover{background:#555}
.use-yolo{background:#1976d2}.use-yolo:hover{background:#2a89e6}
.accept-all{background:#00695c}.accept-all:hover{background:#00796b}
.clear{background:#6b4d1f}.clear:hover{background:#866128}
.save{background:#2e7d32}.save:hover{background:#3f9143}
.skip{background:#7d2d2d}.skip:hover{background:#994040}
#progress{margin-left:auto;color:#aaa;font-size:13px}
#done{display:none;flex:1;justify-content:center;align-items:center;font-size:24px}
kbd{background:#333;padding:2px 5px;border-radius:3px;font-size:11px;margin-right:4px}
</style></head><body>
<div id="main">
  <div id="canvas-panel"><canvas id="canvas"></canvas></div>
  <div id="side">
    <h2>People Review</h2>
    <div class="note">Browse every frame in the video. Green boxes are selected for saving. YOLO boxes start selected by default, dashed blue boxes are extra proposals you can add or remove. Dashed boxes that were rejected by the pre-sorting step are shown but cannot be selected, color-coded by reason: red = too small (min_w/min_h), orange = overlaps a larger box, purple = ReID duplicate in a nearby frame. Each box is labeled with its width x height in pixels.</div>
    <h3>Video</h3>
    <div id="meta" class="meta"></div>
    <h3>Speed Controls</h3>
    <div class="note">
      <kbd>A</kbd> accept all YOLO boxes for the full video<br>
      <kbd>Y</kbd> copy YOLO boxes for current frame<br>
      <kbd>C</kbd> clear accepted boxes on current frame<br>
      <kbd>[</kbd>/<kbd>]</kbd> jump between frames with YOLO detections<br>
      <kbd>Enter</kbd> save the review for this video, even if there are zero boxes<br>
      <kbd>S</kbd> skip video<br>
      <kbd>←</kbd>/<kbd>→</kbd> previous/next frame
    </div>
    <h3>Camera Balance</h3>
    <div id="stats" class="stats"></div>
    <h3>Hour Coverage</h3>
    <div id="hours" class="hour-grid"></div>
  </div>
</div>
<div id="bar">
  <button class="btn nav" onclick="prevFrame()"><kbd>&larr;</kbd>Prev Frame</button>
  <button class="btn nav" onclick="nextFrame()"><kbd>&rarr;</kbd>Next Frame</button>
  <button class="btn nav" onclick="prevSuggestedFrame()"><kbd>[</kbd>Prev YOLO</button>
  <button class="btn nav" onclick="nextSuggestedFrame()"><kbd>]</kbd>Next YOLO</button>
  <button class="btn use-yolo" onclick="useYoloFrame()"><kbd>Y</kbd>Use YOLO Frame</button>
  <button class="btn accept-all" onclick="acceptYoloVideo()"><kbd>A</kbd>Accept YOLO Video</button>
  <button class="btn clear" onclick="clearFrame()"><kbd>C</kbd>Clear Frame</button>
  <button class="btn save" onclick="saveManual()"><kbd>Enter</kbd>Save Review</button>
  <button class="btn skip" onclick="skipVideo()"><kbd>S</kbd>Skip</button>
  <span id="progress">-</span>
</div>
<div id="done">All videos done.</div>
<script>
let cur = null;
let frameIdx = 0;
let currentFrameData = null;
let acceptedByFrame = {};
let drawing = false;
let dragStart = null;
let draftBox = null;
let frameRequestToken = 0;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const frameImg = new Image();

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function normaliseBox(box) {
  let [x1, y1, x2, y2] = box.map(v => Math.round(v));
  if (x1 > x2) [x1, x2] = [x2, x1];
  if (y1 > y2) [y1, y2] = [y2, y1];
  x1 = clamp(x1, 0, canvas.width - 1);
  y1 = clamp(y1, 0, canvas.height - 1);
  x2 = clamp(x2, 1, canvas.width);
  y2 = clamp(y2, 1, canvas.height);
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

function sameBox(a, b) {
  return a && b && a.length === 4 && b.length === 4 && a.every((v, i) => v === b[i]);
}

function currentFrame() {
  return currentFrameData;
}

function acceptedFor(frame) {
  if (!frame) return [];
  return acceptedByFrame[frame.frame_idx] || [];
}

function setAcceptedFor(frame, boxes) {
  if (!frame) return;
  acceptedByFrame[frame.frame_idx] = boxes.map(box => [...box]);
}

function ensureDefaultAcceptedForFrame(frame) {
  if (!frame || !cur || !cur.accept_yolo_by_default) return;
  if (acceptedByFrame[frame.frame_idx] !== undefined) return;
  setAcceptedFor(frame, frame.yolo_suggested_bboxes || []);
}

function totalAcceptedBoxes() {
  return Object.values(acceptedByFrame).reduce((sum, boxes) => sum + boxes.length, 0);
}

function drawBox(box, color, width, dash=[]) {
  if (!box) return;
  const [x1, y1, x2, y2] = box;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.restore();
}

function drawTextLabel(x, y, text, color) {
  ctx.save();
  ctx.font = '14px system-ui, sans-serif';
  ctx.textBaseline = 'bottom';
  const padding = 4;
  const textHeight = 16;
  const metrics = ctx.measureText(text);
  const textWidth = metrics.width;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
  ctx.fillRect(x, y - textHeight, textWidth + padding * 2, textHeight);
  ctx.fillStyle = color;
  ctx.fillText(text, x + padding, y - 2);
  ctx.restore();
  return textHeight;
}

function drawBoxSizeLabel(box, color) {
  if (!box) return;
  const [x1, y1, x2, y2] = box;
  const w = Math.max(0, Math.round(x2 - x1));
  const h = Math.max(0, Math.round(y2 - y1));
  const label = `${w}x${h}`;
  let labelY = y1 - 2;
  if (labelY - 16 < 0) labelY = y1 + 16 + 2;
  drawTextLabel(x1, labelY, label, color);
}

const REJECTION_COLOR = {
  size: '#ef5350',
  overlap: '#ff9800',
  duplicate: '#ab47bc',
};

function rejectedReasonSummary(entries) {
  if (!entries || !entries.length) return '';
  const counts = {};
  entries.forEach(entry => {
    const reason = (entry && entry.reason) || 'unknown';
    counts[reason] = (counts[reason] || 0) + 1;
  });
  const parts = Object.entries(counts).map(([reason, count]) => `${reason}: ${count}`);
  return ` (${parts.join(', ')})`;
}

function drawRejectedBox(entry) {
  const box = entry && entry.box ? entry.box : null;
  if (!box) return;
  const reason = entry.reason || 'size';
  const color = REJECTION_COLOR[reason] || '#ef5350';
  drawBox(box, color, 2, [4, 4]);
  drawBoxSizeLabel(box, color);
  const [x1, y1, x2, y2] = box;
  let labelY = y1 + 16 + 2;
  if (labelY > y2) labelY = Math.min(canvas.height, y2 + 16);
  drawTextLabel(x1, labelY, `rejected: ${reason}`, color);
}

function renderCanvas() {
  if (!frameImg.width || !currentFrameData) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(frameImg, 0, 0);
  (currentFrameData.rejected_bboxes || []).forEach(entry => drawRejectedBox(entry));
  currentFrameData.suggested_bboxes.forEach(box => {
    drawBox(box, '#4fc3f7', 2, [8, 4]);
    drawBoxSizeLabel(box, '#4fc3f7');
  });
  acceptedFor(currentFrameData).forEach(box => {
    drawBox(box, '#4caf50', 3);
    drawBoxSizeLabel(box, '#81c784');
  });
  if (draftBox) {
    drawBox(draftBox, '#ffb74d', 2);
    drawBoxSizeLabel(draftBox, '#ffb74d');
  }
}

function updateMeta() {
  if (!cur) return;
  const frame = currentFrame();
  const hasYoloSuggestion = frame ? frame.yolo_suggested_bboxes.length > 0 : false;
  const hasExtraSuggestion = frame ? frame.suggested_bboxes.length > frame.yolo_suggested_bboxes.length : false;
  document.getElementById('meta').innerHTML = `
    <div><strong>Camera:</strong> ${cur.camera}</div>
    <div><strong>Video:</strong> ${cur.video_name}</div>
    <div><strong>Timestamp:</strong> ${cur.timestamp || 'unknown'}</div>
    <div><strong>Frame:</strong> ${frameIdx + 1}/${cur.frame_count}${frame ? ` (${frame.frame_idx})` : ''}</div>
    <div><strong>Frames with YOLO:</strong> ${cur.yolo_frame_count}/${cur.frame_count}</div>
    <div><strong>Frames with any proposal:</strong> ${cur.suggested_frame_count}/${cur.frame_count}</div>
    <div><strong>YOLO boxes in frame:</strong> ${frame ? frame.yolo_suggested_bboxes.length : 0}</div>
    <div><strong>Suggested boxes in frame:</strong> ${frame ? frame.suggested_bboxes.length : 0}</div>
    <div><strong>Rejected boxes in frame:</strong> ${frame ? (frame.rejected_bboxes || []).length : 0}${frame ? rejectedReasonSummary(frame.rejected_bboxes || []) : ''}</div>
    <div><strong>Frame status:</strong> ${hasYoloSuggestion ? 'YOLO sees person(s)' : 'No YOLO person in this frame'}${hasExtraSuggestion ? ' | extra proposal available' : ''}</div>
    <div><strong>Accepted boxes in frame:</strong> ${frame ? acceptedFor(frame).length : 0}</div>
    <div><strong>Accepted boxes in video:</strong> ${totalAcceptedBoxes()}</div>
    <div><strong>Session saved:</strong> ${cur.session_saved}</div>
    <div><strong>Session skipped:</strong> ${cur.session_skipped}</div>
  `;
  document.getElementById('progress').textContent = `frame ${frameIdx + 1}/${cur.frame_count} | ${cur.yolo_frame_count} YOLO frames | ${totalAcceptedBoxes()} accepted boxes`;
  document.getElementById('stats').innerHTML = cur.stats.map(stat => `
    <div class="stat">
      <div class="camera">${stat.camera}</div>
      <div class="counts">${stat.labeled} labeled, ${stat.remaining} remaining</div>
    </div>
  `).join('');
  const hours = Object.entries(cur.hour_counts || {}).map(([hour, count]) => {
    const isPriority = (cur.priority_hours || []).includes(Number(hour));
    return `<div class="hour-pill ${isPriority ? 'priority' : ''}">${hour.padStart(2, '0')}: ${count}</div>`;
  }).join('');
  document.getElementById('hours').innerHTML = hours;
}

function showDone(data) {
  document.getElementById('main').style.display = 'none';
  document.getElementById('bar').style.display = 'none';
  document.getElementById('done').style.display = 'flex';
  document.getElementById('done').textContent = `All videos done. Saved ${data.session_saved}, skipped ${data.session_skipped}.`;
}

async function loadFrame(targetFrameIdx = frameIdx) {
  if (!cur) return;
  frameIdx = clamp(targetFrameIdx, 0, cur.frame_count - 1);
  currentFrameData = null;
  draftBox = null;
  updateMeta();
  document.getElementById('progress').textContent = `Loading frame ${frameIdx + 1}/${cur.frame_count}...`;
  const token = ++frameRequestToken;
  const res = await fetch(`/frame/${frameIdx}`);
  const data = await res.json();
  if (token !== frameRequestToken || !data.ok) return;
  currentFrameData = data.frame;
  ensureDefaultAcceptedForFrame(currentFrameData);
  frameImg.onload = () => {
    if (token !== frameRequestToken) return;
    canvas.width = frameImg.width;
    canvas.height = frameImg.height;
    renderCanvas();
    updateMeta();
  };
  frameImg.src = 'data:image/jpeg;base64,' + currentFrameData.b64;
}

function render(data) {
  if (data.done) {
    showDone(data);
    return;
  }
  cur = data;
  frameIdx = 0;
  currentFrameData = null;
  acceptedByFrame = {};
  loadFrame();
}

async function load() {
  const data = await (await fetch('/next')).json();
  render(data);
}

function prevFrame() {
  if (!cur) return;
  loadFrame((frameIdx - 1 + cur.frame_count) % cur.frame_count);
}

function nextFrame() {
  if (!cur) return;
  loadFrame((frameIdx + 1) % cur.frame_count);
}

function moveSuggestedFrame(step) {
  if (!cur || !cur.yolo_frame_indices.length) return;
  const indices = cur.yolo_frame_indices;
  if (step > 0) {
    const next = indices.find(idx => idx > frameIdx);
    loadFrame(next === undefined ? indices[0] : next);
    return;
  }
  const reversed = [...indices].reverse();
  const prev = reversed.find(idx => idx < frameIdx);
  loadFrame(prev === undefined ? reversed[0] : prev);
}

function prevSuggestedFrame() {
  moveSuggestedFrame(-1);
}

function nextSuggestedFrame() {
  moveSuggestedFrame(1);
}

function useYoloFrame() {
  const frame = currentFrame();
  if (!frame) return;
  setAcceptedFor(frame, frame.yolo_suggested_bboxes || []);
  renderCanvas();
  updateMeta();
}

function clearFrame() {
  const frame = currentFrame();
  if (!frame) return;
  setAcceptedFor(frame, []);
  draftBox = null;
  renderCanvas();
  updateMeta();
}

function pointInBox(point, box) {
  const [x1, y1, x2, y2] = box;
  return point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2;
}

function toggleSuggestedAtPoint(point) {
  const frame = currentFrame();
  if (!frame) return false;
  const accepted = acceptedFor(frame);
  for (const suggested of frame.suggested_bboxes) {
    if (!pointInBox(point, suggested)) continue;
    const idx = accepted.findIndex(box => sameBox(box, suggested));
    if (idx >= 0) accepted.splice(idx, 1);
    else accepted.push([...suggested]);
    setAcceptedFor(frame, accepted);
    return true;
  }
  return false;
}

function removeAcceptedAtPoint(point) {
  const frame = currentFrame();
  if (!frame) return false;
  const accepted = acceptedFor(frame);
  const idx = accepted.findIndex(box => pointInBox(point, box));
  if (idx < 0) return false;
  accepted.splice(idx, 1);
  setAcceptedFor(frame, accepted);
  return true;
}

async function acceptYoloVideo() {
  await fetch('/accept_yolo', {method: 'POST'});
  await load();
}

async function saveManual() {
  const nonEmpty = {};
  for (const [frame, boxes] of Object.entries(acceptedByFrame)) {
    if (boxes.length) nonEmpty[frame] = boxes;
  }
  await fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      video: cur.video,
      frame_bboxes: nonEmpty,
    }),
  });
  await load();
}

async function skipVideo() {
  await fetch('/skip', {method: 'POST'});
  await load();
}

function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (evt.clientX - rect.left) * (canvas.width / rect.width),
    y: (evt.clientY - rect.top) * (canvas.height / rect.height),
  };
}

canvas.addEventListener('mousedown', evt => {
  if (!cur || !currentFrameData) return;
  drawing = true;
  dragStart = canvasPoint(evt);
  draftBox = [dragStart.x, dragStart.y, dragStart.x, dragStart.y];
  renderCanvas();
});

canvas.addEventListener('mousemove', evt => {
  if (!drawing) return;
  const pt = canvasPoint(evt);
  draftBox = [dragStart.x, dragStart.y, pt.x, pt.y];
  renderCanvas();
});

window.addEventListener('mouseup', evt => {
  if (!drawing) return;
  drawing = false;
  const pt = canvasPoint(evt);
  const moved = Math.hypot(pt.x - dragStart.x, pt.y - dragStart.y);
  if (moved < 6) {
    draftBox = null;
    if (!toggleSuggestedAtPoint(pt)) removeAcceptedAtPoint(pt);
    renderCanvas();
    updateMeta();
    return;
  }
  const newBox = normaliseBox([dragStart.x, dragStart.y, pt.x, pt.y]);
  draftBox = null;
  if (!newBox) return;
  const frame = currentFrame();
  const accepted = acceptedFor(frame);
  accepted.push(newBox);
  setAcceptedFor(frame, accepted);
  renderCanvas();
  updateMeta();
});

canvas.addEventListener('dblclick', evt => {
  const pt = canvasPoint(evt);
  if (removeAcceptedAtPoint(pt)) {
    renderCanvas();
    updateMeta();
  }
});

document.addEventListener('keydown', evt => {
  if (evt.key === 'ArrowLeft') prevFrame();
  else if (evt.key === 'ArrowRight') nextFrame();
  else if (evt.key === '[') prevSuggestedFrame();
  else if (evt.key === ']') nextSuggestedFrame();
  else if (evt.key.toLowerCase() === 'y') useYoloFrame();
  else if (evt.key.toLowerCase() === 'a') acceptYoloVideo();
  else if (evt.key.toLowerCase() === 'c') clearFrame();
  else if (evt.key.toLowerCase() === 's') skipVideo();
  else if (evt.key === 'Enter') saveManual();
});

load();
</script></body></html>"""


if __name__ == "__main__":
  config_name = None
  seed = 0
  cache_path = None
  refresh_cache = False
  model_path = None
  for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
      config_name = sys.argv[i + 1]
    elif arg == "--seed" and i + 1 < len(sys.argv):
      seed = int(sys.argv[i + 1])
    elif arg == "--cache" and i + 1 < len(sys.argv):
      cache_path = Path(sys.argv[i + 1])
    elif arg == "--model" and i + 1 < len(sys.argv):
      model_path = Path(sys.argv[i + 1])
    elif arg == "--refresh-cache":
      refresh_cache = True

  if "--help" in sys.argv or "-h" in sys.argv:
    print(
      "Usage: python -m easysort.validators.recycling_people "
      "[--config NAME] [--seed N] [--cache PATH] [--model PATH] [--refresh-cache]"
    )
    sys.exit(0)

  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  validator = RecyclingPeopleValidator(
    registry,
    config_name=config_name,
    seed=seed,
    cache_path=cache_path,
    refresh_cache=refresh_cache,
    model_path=model_path,
  )
  validator.run()
