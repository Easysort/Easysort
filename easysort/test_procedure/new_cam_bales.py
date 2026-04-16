"""New camera bale estimation from MP4 video.

Samples frames at 30s and 2min intervals, runs the same template-matching
pipeline as production (easyprod/scripts/verdis/bales.py), and reports
per-interval and total bale counts.

First run: user draws a polygon around the conveyor region.
Polygon is cached for reuse; pass --redraw to redo it.

Usage:
  python -m easysort.test_procedure.new_cam_bales [path.mp4] [--redraw] [--debug] [--line]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

POLYGON_CACHE = Path(__file__).parent / "new_cam_bales_polygon.json"
POLYGON = [
  (1486, 729), (1705, 795), (1895, 841), (1950, 850),
  (1970, 740), (1731, 688), (1564, 634), (1356, 564), (1319, 668),
]
PATCH_RAD = 30
MAX_SEGMENTS = 12
TOP_N = 4
PIXELS_PER_BALE = 140


# --- Polygon persistence + drawing ---

def _load_polygon() -> list[tuple[int, int]] | None:
  if not POLYGON_CACHE.exists():
    return None
  try:
    pts = json.loads(POLYGON_CACHE.read_text()).get("points", [])
    return [(int(p[0]), int(p[1])) for p in pts] if pts else None
  except (json.JSONDecodeError, KeyError, TypeError, IndexError):
    return None


def _save_polygon(points: list[tuple[int, int]]) -> None:
  POLYGON_CACHE.write_text(json.dumps({"points": list(points)}))


def _draw_polygon(img_bgr: np.ndarray, max_display_side: int = 900) -> list[tuple[int, int]] | None:
  """Click to add points. Enter=done, r=reset, q=cancel. Returns points in original image coords."""
  points: list[tuple[int, int]] = []
  h, w = img_bgr.shape[:2]
  scale = min(1.0, max_display_side / max(h, w))
  display = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr.copy()
  if display.ndim == 2:
    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
  win = "Draw polygon (Enter=done, r=reset, q=cancel)"

  def to_orig(x, y):
    return (int(round(x / scale)), int(round(y / scale))) if scale < 1.0 else (x, y)

  def to_disp(p):
    return (int(p[0] * scale), int(p[1] * scale)) if scale < 1.0 else p

  def redraw():
    canvas = display.copy()
    for i, p in enumerate(points):
      cv2.circle(canvas, to_disp(p), max(2, int(5 * scale)), (0, 255, 0), -1)
      if i > 0:
        cv2.line(canvas, to_disp(points[i - 1]), to_disp(p), (0, 255, 0), 2)
    if len(points) >= 2:
      cv2.line(canvas, to_disp(points[-1]), to_disp(points[0]), (0, 255, 0), 2)
    cv2.putText(canvas, "Click=add point, Enter=done, r=reset, q=cancel", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow(win, canvas)

  def on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
      points.append(to_orig(x, y))
      redraw()

  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.setMouseCallback(win, on_mouse)
  redraw()
  while True:
    key = cv2.waitKey(50) & 0xFF
    if key in (ord("q"), ord("Q")):
      cv2.destroyWindow(win)
      return None
    if key in (ord("r"), ord("R")):
      points.clear()
      redraw()
    if key in (13, 10) and len(points) >= 3:
      cv2.destroyWindow(win)
      return points


def _measure_line(img_bgr: np.ndarray, max_display_side: int = 900) -> None:
  """Click two points to measure pixel distance. Click again to start a new line. q/Enter to close."""
  h, w = img_bgr.shape[:2]
  scale = min(1.0, max_display_side / max(h, w))
  dw, dh = int(w * scale), int(h * scale)
  display = cv2.resize(img_bgr, (dw, dh)) if scale < 1.0 else img_bgr.copy()
  if display.ndim == 2:
    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
  base = display.copy()
  points: list[tuple[int, int]] = []
  win = "Measure line (click 2 points, r=reset, q/Enter=close)"
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)

  def redraw():
    canvas = base.copy()
    for p in points:
      cv2.circle(canvas, p, 5, (0, 0, 255), -1)
    if len(points) == 2:
      cv2.line(canvas, points[0], points[1], (0, 255, 0), 2)
      dx = (points[1][0] - points[0][0]) / scale
      dy = (points[1][1] - points[0][1]) / scale
      length = float(np.hypot(dx, dy))
      mx = (points[0][0] + points[1][0]) // 2
      my = (points[0][1] + points[1][1]) // 2
      cv2.putText(canvas, f"{length:.0f} px", (mx + 10, my - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(canvas, f"PIXELS_PER_BALE = {length:.0f}", (10, dh - 15),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
      print(f"  Line: {length:.1f} px  (set PIXELS_PER_BALE = {int(round(length))})")
    cv2.imshow(win, canvas)

  def on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
      if len(points) >= 2:
        points.clear()
      points.append((x, y))
      redraw()

  cv2.setMouseCallback(win, on_mouse)
  cv2.putText(base, "Click start + end of one bale width to measure", (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
  redraw()
  while True:
    key = cv2.waitKey(50) & 0xFF
    if key in (ord("q"), ord("Q"), 27, 13, 10):
      break
    if key in (ord("r"), ord("R")):
      points.clear()
      redraw()
  cv2.destroyWindow(win)


# --- Video sampling ---

def _sample_frames(video_path: str, interval_sec: float) -> list[tuple[float, np.ndarray]]:
  """Extract RGB frames at fixed intervals. Returns [(timestamp_sec, rgb_frame), ...]."""
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")
  fps = cap.get(cv2.CAP_PROP_FPS)
  total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration = total / fps
  print(f"  Video: {fps:.1f} fps, {total} frames, {duration:.1f}s ({duration / 60:.1f} min)")

  frames = []
  t = 0.0
  while t <= duration:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if not ret:
      break
    frames.append((t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    t += interval_sec
  cap.release()
  print(f"  Sampled {len(frames)} frames at {interval_sec}s intervals")
  return frames


# --- Motion estimation (direction-agnostic, smaller patches) ---

SEG_COLORS = [
  (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
  (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]


def _poly_mask(shape, pts):
  m = np.zeros(shape[:2], dtype=np.uint8)
  if pts:
    cv2.fillPoly(m, [np.array(pts, dtype=np.int32)], 255)
  return m


def _seg_centers(h, w, polygon):
  mask = _poly_mask((h, w), polygon) if polygon else np.ones((h, w), dtype=np.uint8) * 255
  ys, xs = np.where(mask > 0)
  if len(ys) == 0:
    return []
  ymin, ymax = int(ys.min()), int(ys.max())
  xmin, xmax = int(xs.min()), int(xs.max())
  sp = PATCH_RAD * 2
  cands = []
  for y in range(max(ymin + PATCH_RAD, PATCH_RAD), min(ymax, h - PATCH_RAD) + 1, sp):
    for x in range(max(xmin + PATCH_RAD, PATCH_RAD), min(xmax, w - PATCH_RAD) + 1, sp):
      if mask[y, x] > 0:
        cands.append((x, y))
  if not cands:
    ys2, xs2 = ys, xs
    idx = np.linspace(0, len(ys2) - 1, min(MAX_SEGMENTS, len(ys2)), dtype=int)
    return [(int(xs2[i]), int(ys2[i])) for i in idx]
  if len(cands) <= MAX_SEGMENTS:
    return cands
  idx = np.linspace(0, len(cands) - 1, MAX_SEGMENTS, dtype=int)
  return [cands[i] for i in idx]


def _match_patch(patch, img_b, polygon=None):
  ph, pw = patch.shape[:2]
  if ph == 0 or pw == 0:
    return None
  h, w = img_b.shape[:2]
  if w < pw or h < ph:
    return None
  res = cv2.matchTemplate(img_b, patch, cv2.TM_CCOEFF_NORMED)
  if polygon:
    pts = np.array(polygon, dtype=np.int32).copy()
    pts[:, 0] -= pw // 2
    pts[:, 1] -= ph // 2
    rm = np.zeros(res.shape[:2], dtype=np.uint8)
    cv2.fillPoly(rm, [pts], 255)
    res = np.where(rm > 0, res, -1.0)
  _, mv, _, (bx, by) = cv2.minMaxLoc(res)
  return (bx + pw // 2, by + ph // 2, mv) if mv >= 0.2 else None


def _estimate_bales(img_a, img_b, polygon):
  """Rightward-motion estimation: filter to dx > 0, keep top N by magnitude."""
  if img_b.shape[:2] != img_a.shape[:2]:
    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
  h, w = img_a.shape[:2]
  ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY) if img_a.ndim == 3 else img_a
  gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY) if img_b.ndim == 3 else img_b
  ox, oy, lp = 0, 0, polygon
  if polygon:
    pts = np.array(polygon, dtype=np.int32)
    x1 = max(0, int(pts[:, 0].min()) - PATCH_RAD)
    y1 = max(0, int(pts[:, 1].min()) - PATCH_RAD)
    x2 = min(w, int(pts[:, 0].max()) + PATCH_RAD)
    y2 = min(h, int(pts[:, 1].max()) + PATCH_RAD)
    ga, gb = ga[y1:y2, x1:x2], gb[y1:y2, x1:x2]
    lp = [(px - x1, py - y1) for px, py in polygon]
    ox, oy = x1, y1
    h, w = ga.shape[:2]
  centers = _seg_centers(h, w, lp)
  if not centers:
    return 0.0, [], []

  all_m = []
  for cx, cy in centers:
    r1, r2 = max(0, cy - PATCH_RAD), min(h, cy + PATCH_RAD)
    c1, c2 = max(0, cx - PATCH_RAD), min(w, cx + PATCH_RAD)
    patch = ga[r1:r2, c1:c2]
    if patch.size == 0:
      continue
    m = _match_patch(patch, gb, lp)
    if m is None:
      continue
    dx, dy = m[0] - cx, m[1] - cy
    mag = float(np.hypot(dx, dy))
    if mag < 1.0:
      continue
    all_m.append((cx + ox, cy + oy, int(m[0]) + ox, int(m[1]) + oy, m[2], dx, dy))

  if not all_m:
    return 0.0, all_m, []

  # Filter to rightward motion only, then pick top N by magnitude
  rightward = [m for m in all_m if m[5] > 0]
  pool = rightward if rightward else all_m
  ranked = sorted(pool, key=lambda m: np.hypot(m[5], m[6]), reverse=True)
  kept = ranked[:TOP_N]

  # Outlier filter on the kept set
  mags = [float(np.hypot(m[5], m[6])) for m in kept]
  med = float(np.median(mags))
  if med > 0:
    filt = [m for m, mg in zip(kept, mags) if 0.5 * med <= mg <= 1.5 * med]
    if filt:
      kept = filt

  adx = float(np.mean([m[5] for m in kept]))
  ady = float(np.mean([m[6] for m in kept]))
  return float(np.hypot(adx, ady)) / PIXELS_PER_BALE, all_m, kept


# --- Debug visualization (4-panel: Frame A | Frame B / Motion diff | Info panel) ---

def _put_centered(img, text, cx, cy, scale, color, thickness):
  (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
  cv2.putText(img, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _build_debug_image(
  img_a_rgb: np.ndarray, img_b_rgb: np.ndarray, polygon: list[tuple[int, int]],
  all_matches: list, kept_matches: list, bales: float, time_a: str, time_b: str,
  running_total: float, pair_idx: int, total_pairs: int,
) -> np.ndarray:
  """4-panel debug: [Frame A | Frame B] / [Motion diff | Info summary]."""
  h, w = img_a_rgb.shape[:2]
  a = cv2.cvtColor(img_a_rgb, cv2.COLOR_RGB2BGR)
  b = cv2.cvtColor(img_b_rgb, cv2.COLOR_RGB2BGR)

  kept_set = set(id(m) for m in kept_matches)

  # --- All matches: dim gray boxes + rejection reason ---
  for m in all_matches:
    xa, ya, xb, yb, cf, dx, dy = m
    if id(m) in kept_set:
      continue
    cv2.rectangle(a, (xa - PATCH_RAD, ya - PATCH_RAD), (xa + PATCH_RAD, ya + PATCH_RAD), (80, 80, 80), 1)
    cv2.rectangle(b, (xb - PATCH_RAD, yb - PATCH_RAD), (xb + PATCH_RAD, yb + PATCH_RAD), (80, 80, 80), 1)
    mag = float(np.hypot(dx, dy))
    reason = "left" if dx <= 0 else "low"
    cv2.arrowedLine(a, (xa, ya), (xa + int(dx), ya + int(dy)), (80, 80, 80), 1, tipLength=0.2)
    cv2.putText(a, f"{mag:.0f}px [{reason}]", (xa + PATCH_RAD + 3, ya),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

  # --- Kept matches: colored boxes + arrows + labels ---
  for i, (xa, ya, xb, yb, cf, dx, dy) in enumerate(kept_matches):
    c = SEG_COLORS[i % len(SEG_COLORS)]
    cv2.rectangle(a, (xa - PATCH_RAD, ya - PATCH_RAD), (xa + PATCH_RAD, ya + PATCH_RAD), c, 3)
    cv2.rectangle(b, (xb - PATCH_RAD, yb - PATCH_RAD), (xb + PATCH_RAD, yb + PATCH_RAD), c, 3)
    cv2.arrowedLine(a, (xa, ya), (xa + int(dx), ya + int(dy)), c, 2, tipLength=0.2)
    mag = float(np.hypot(dx, dy))
    cv2.putText(a, f"#{i + 1} {mag:.0f}px c={cf:.2f}", (xa + PATCH_RAD + 3, ya),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    cv2.putText(b, f"#{i + 1}", (xb + PATCH_RAD + 3, yb),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

  # --- Polygon outline ---
  if polygon:
    pp = np.array(polygon, dtype=np.int32)
    cv2.polylines(a, [pp], True, (0, 255, 255), 2)
    cv2.polylines(b, [pp], True, (0, 255, 255), 2)

  # --- Frame labels ---
  cv2.putText(a, f"Frame A  {time_a}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(b, f"Frame B  {time_b}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
  n_right = sum(1 for m in all_matches if m[5] > 0)
  cv2.putText(a, f"Bales: {bales:.4f}  (kept {len(kept_matches)} / {n_right} right / {len(all_matches)} total)",
              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

  # --- Bottom-left: motion diff ---
  diff = cv2.absdiff(img_a_rgb, img_b_rgb)
  gd = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.ndim == 3 else diff
  _, mb = cv2.threshold(gd, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  if polygon:
    pmask = _poly_mask(mb.shape, polygon)
    mb = mb * (pmask > 0)
  md = cv2.cvtColor(mb, cv2.COLOR_GRAY2BGR)
  motion_pct = float(np.sum(mb > 0)) / max(1, np.sum(pmask > 0) if polygon else mb.size) * 100
  cv2.putText(md, f"Motion diff ({motion_pct:.1f}% active)", (10, 35),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

  # --- Bottom-right: info summary panel ---
  pnl = np.full((h, w, 3), 30, dtype=np.uint8)

  # Header
  cv2.putText(pnl, f"Pair {pair_idx + 1}/{total_pairs}", (10, 35),
              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
  cv2.putText(pnl, f"{time_a} -> {time_b}", (10, 65),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

  if kept_matches:
    adx = float(np.mean([m[5] for m in kept_matches]))
    ady = float(np.mean([m[6] for m in kept_matches]))
    amag = float(np.hypot(adx, ady))
    acf = float(np.mean([m[4] for m in kept_matches]))

    # Big arrow showing average direction
    ax, ay = w // 4, h // 3
    arrow_sc = min(w * 0.25, h * 0.25) / max(amag, 1.0)
    ux, uy = adx / max(amag, 1e-6), ady / max(amag, 1e-6)
    al = min(amag * arrow_sc, min(w, h) * 0.3)
    cv2.arrowedLine(pnl,
      (int(ax - ux * al * 0.4), int(ay - uy * al * 0.4)),
      (int(ax + ux * al * 0.6), int(ay + uy * al * 0.6)),
      (0, 255, 0), 5, tipLength=0.25)
    _put_centered(pnl, f"{amag:.0f} px avg", ax, ay + 60, 1.2, (0, 255, 0), 2)
    _put_centered(pnl, f"conf {acf:.2f}", ax, ay + 95, 0.7, (180, 180, 180), 1)

    # Bale count + running total
    bx, by = w * 3 // 4, h // 5
    _put_centered(pnl, f"{bales:.4f}", bx, by, 1.8, (0, 200, 255), 3)
    _put_centered(pnl, "bales this pair", bx, by + 45, 0.6, (140, 140, 140), 1)
    _put_centered(pnl, f"{running_total:.4f}", bx, by + 100, 1.4, (255, 200, 0), 2)
    _put_centered(pnl, "running total", bx, by + 135, 0.6, (140, 140, 140), 1)

    # Per-segment detail table
    ty = h // 2 + 20
    cv2.putText(pnl, "Kept segments:", (15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    ty += 30
    for i, (xa, ya, xb, yb, cf, dx, dy) in enumerate(kept_matches):
      c = SEG_COLORS[i % len(SEG_COLORS)]
      mag = float(np.hypot(dx, dy))
      ang = float(np.degrees(np.arctan2(-dy, dx)))
      line = f"#{i + 1}  dx={dx:+.0f}  dy={dy:+.0f}  mag={mag:.0f}px  ang={ang:.0f} deg  conf={cf:.2f}"
      cv2.putText(pnl, line, (20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
      # Mini arrow
      mx = w - 60
      sl = min(25, mag * 0.3)
      ndx, ndy = dx / max(mag, 1e-6), dy / max(mag, 1e-6)
      cv2.arrowedLine(pnl, (mx, ty - 5), (int(mx + ndx * sl), int(ty - 5 + ndy * sl)), c, 2, tipLength=0.3)
      ty += 25
      if ty > h - 60:
        break

    # Rejected segments summary
    n_rejected = len(all_matches) - len(kept_matches)
    if n_rejected > 0:
      ty += 10
      n_left = sum(1 for m in all_matches if m[5] <= 0)
      n_outlier = n_rejected - n_left
      cv2.putText(pnl, f"Rejected: {n_rejected} ({n_left} leftward, {n_outlier} outlier/low rank)",
                  (15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
  else:
    _put_centered(pnl, "No rightward motion", w // 2, h // 3, 1.2, (0, 0, 255), 2)
    _put_centered(pnl, f"{len(all_matches)} matches, none moving right", w // 2, h // 3 + 40, 0.6, (150, 150, 150), 1)
    bx, by = w // 2, h // 2 + 30
    _put_centered(pnl, f"Running total: {running_total:.4f}", bx, by, 1.0, (255, 200, 0), 2)

  # --- Compose 4-panel grid ---
  ht = min(a.shape[0], b.shape[0])
  a = cv2.resize(a, (int(a.shape[1] * ht / a.shape[0]), ht))
  b = cv2.resize(b, (int(b.shape[1] * ht / b.shape[0]), ht))
  top = np.hstack([a, b])
  hw = top.shape[1] // 2
  bottom = np.hstack([cv2.resize(md, (hw, ht)), cv2.resize(pnl, (hw, ht))])
  grid = np.vstack([top, bottom])

  sc = min(1900 / grid.shape[1], 1000 / grid.shape[0], 1.0)
  if sc < 1.0:
    grid = cv2.resize(grid, (int(grid.shape[1] * sc), int(grid.shape[0] * sc)))
  return grid


def _debug_viewer(debug_images: list[np.ndarray], results: list[tuple[str, str, float]], label: str):
  """Navigate debug images. ←/→ or a/d to move, q/Esc to quit."""
  idx = 0
  win = "Debug"
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  while True:
    cv2.imshow(win, debug_images[idx])
    ta, tb, bales = results[idx]
    running = sum(r[2] for r in results[:idx + 1])
    cv2.setWindowTitle(win,
      f"[{idx + 1}/{len(debug_images)}] {ta}->{tb}  {bales:.4f} bales  total={running:.4f} — {label}")
    key = cv2.waitKey(0) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
      break
    elif key in (3, 83, ord("d"), ord("D")):
      idx = min(idx + 1, len(debug_images) - 1)
    elif key in (2, 81, ord("a"), ord("A")):
      idx = max(idx - 1, 0)
  cv2.destroyWindow(win)


# --- Estimation + reporting ---

def _fmt(sec: float) -> str:
  m, s = divmod(int(sec), 60)
  return f"{m:02d}:{s:02d}"


def _run_estimation(
  frames: list[tuple[float, np.ndarray]], polygon: list[tuple[int, int]], debug: bool = False,
) -> tuple[list[tuple[str, str, float]], list[np.ndarray]]:
  results = []
  debug_images = []
  running_total = 0.0
  n_pairs = len(frames) - 1
  for i in range(n_pairs):
    t_a, img_a = frames[i]
    t_b, img_b = frames[i + 1]
    bales, all_m, kept = _estimate_bales(img_a, img_b, polygon)
    if bales < 0.05:
      bales = 0.0
    bales = round(bales, 4)
    running_total += bales
    ta, tb = _fmt(t_a), _fmt(t_b)
    results.append((ta, tb, bales))
    if debug:
      debug_images.append(_build_debug_image(
        img_a, img_b, polygon, all_m, kept, bales, ta, tb, running_total, i, n_pairs))
  return results, debug_images


def _print_results(results: list[tuple[str, str, float]], label: str) -> float:
  total = sum(r[2] for r in results)
  print(f"\n{'=' * 55}")
  print(f"  Bale estimation — {label} intervals")
  print(f"{'=' * 55}")
  print(f"  {'#':<4} {'From':>7}  →  {'To':<7} {'Bales':>8}")
  print(f"  {'-' * 47}")
  for i, (ta, tb, b) in enumerate(results):
    bar = "█" * int(b * 40) if b > 0 else ""
    print(f"  {i + 1:<4} {ta:>7}  →  {tb:<7} {b:>8.4f}  {bar}")
  print(f"  {'-' * 47}")
  print(f"  {'Total':<4} {'':>7}     {'':>7} {total:>8.4f}")
  print(f"  {'Pairs':<4} {'':>7}     {'':>7} {len(results):>8}")
  print(f"{'=' * 55}")
  return total


def _get_polygon(first_frame_bgr: np.ndarray, force_redraw: bool = False) -> list[tuple[int, int]]:
  if not force_redraw:
    print(f"Using hardcoded polygon ({len(POLYGON)} points)")
    return POLYGON
  print("Draw a new polygon on the first frame.")
  print("  Click to add points, Enter to confirm, r to reset, q to cancel.")
  polygon = _draw_polygon(first_frame_bgr)
  if polygon is None or len(polygon) < 3:
    print("Polygon cancelled or too few points. Using hardcoded polygon.")
    return POLYGON
  _save_polygon(polygon)
  print(f"Polygon saved ({len(polygon)} points) to {POLYGON_CACHE.name}")
  return polygon


def main():
  mp4_path = "/Users/lucasvilsen/Downloads/Record/DownLoad/verdis_ch9_20260305104529_20260305104903.mp4"
  redraw = "--redraw" in sys.argv
  debug = "--debug" in sys.argv
  measure = "--line" in sys.argv
  for arg in sys.argv[1:]:
    if not arg.startswith("--") and arg.endswith(".mp4"):
      mp4_path = arg

  if not Path(mp4_path).exists():
    print(f"Video not found: {mp4_path}")
    sys.exit(1)

  print(f"Video: {mp4_path}")

  cap = cv2.VideoCapture(mp4_path)
  ret, first_bgr = cap.read()
  cap.release()
  if not ret:
    print("Cannot read first frame")
    sys.exit(1)

  if measure:
    print("Draw a line across one bale to measure its width in pixels.")
    print("  Click two points, r to reset, q/Enter to close.")
    _measure_line(first_bgr)
    sys.exit(0)

  polygon = _get_polygon(first_bgr, force_redraw=redraw)

  # --- 30-second intervals ---
  print("\n--- Sampling every 30 seconds ---")
  frames_30s = _sample_frames(mp4_path, 30)
  results_30s, dbg_30s = _run_estimation(frames_30s, polygon, debug=debug)
  total_30s = _print_results(results_30s, "30s")
  if debug and dbg_30s:
    print("  [Debug] Showing 30s intervals (←/→ or a/d to navigate, q to quit)")
    _debug_viewer(dbg_30s, results_30s, "30s intervals")

  # --- 2-minute intervals ---
  print("\n--- Sampling every 2 minutes ---")
  frames_2m = _sample_frames(mp4_path, 120)
  results_2m, dbg_2m = _run_estimation(frames_2m, polygon, debug=debug)
  total_2m = _print_results(results_2m, "2min")
  if debug and dbg_2m:
    print("  [Debug] Showing 2min intervals (←/→ or a/d to navigate, q to quit)")
    _debug_viewer(dbg_2m, results_2m, "2min intervals")

  print(f"\nSummary:")
  print(f"  30s intervals:  {total_30s:.4f} bales ({len(results_30s)} pairs)")
  print(f"  2min intervals: {total_2m:.4f} bales ({len(results_2m)} pairs)")


if __name__ == "__main__":
  main()
