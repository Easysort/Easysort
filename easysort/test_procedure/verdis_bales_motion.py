"""Load two images from registry folders, show originals + binary motion (full-frame absdiff)."""

import json
from pathlib import Path

import cv2
import numpy as np

from easysort.helpers import REGISTRY_LOCAL_IP
from easysort.registry import RegistryBase

PREFIX = "verdis/gadstrup/4"
POLYGON_CACHE = Path(__file__).parent / "verdis_bales_motion_polygon.json"


def load_polygon() -> list[tuple[int, int]] | None:
  """Load polygon from cache file. Returns None if missing or invalid."""
  if not POLYGON_CACHE.exists():
    return None
  try:
    data = json.loads(POLYGON_CACHE.read_text())
    pts = data.get("points", [])
    return [(int(p[0]), int(p[1])) for p in pts] if pts else None
  except (json.JSONDecodeError, KeyError, TypeError, IndexError):
    return None


def save_polygon(points: list[tuple[int, int]]) -> None:
  """Overwrite cache file with the given polygon."""
  POLYGON_CACHE.write_text(json.dumps({"points": list(points)}))


def motion_image(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
  """Binary motion from full-frame absdiff (same size as first image)."""
  if img_b.shape[:2] != img_a.shape[:2]:
    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
  diff = cv2.absdiff(img_a, img_b)
  gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if len(diff.shape) == 3 else diff
  _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return binary


def apply_polygon_mask(img: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
  """Zero out everything outside the polygon; keep values inside."""
  if not points:
    return img
  mask = np.zeros(img.shape[:2], dtype=np.uint8)
  pts = np.array(points, dtype=np.int32)
  cv2.fillPoly(mask, [pts], 255)
  if len(img.shape) == 3:
    mask = mask[:, :, np.newaxis]
  return img * (mask > 0).astype(img.dtype)


def polygon_mask_shape(shape: tuple[int, ...], points: list[tuple[int, int]]) -> np.ndarray:
  """Binary mask (0/255) of polygon region, shape (H,W)."""
  mask = np.zeros(shape[:2], dtype=np.uint8)
  if not points:
    return mask
  pts = np.array(points, dtype=np.int32)
  cv2.fillPoly(mask, [pts], 255)
  return mask


def motion_inside_crop(motion_binary: np.ndarray, polygon: list[tuple[int, int]] | None) -> bool:
  """True if there is significant motion inside the crop (or full frame if no polygon)."""
  if polygon:
    mask = polygon_mask_shape(motion_binary.shape, polygon)
    motion_in_crop = motion_binary * (mask > 0)
    area = max(1, np.sum(mask > 0))
    # motion if > 0.5% of crop area is moving
    return float(np.sum(motion_in_crop > 0)) / area > 0.005
  return float(np.sum(motion_binary > 0)) / motion_binary.size > 0.005


# Segment patch radius (patch size = 2*PATCH_RAD). Search the full image to find segment in image 2.
PATCH_RAD = 60
MAX_SEGMENTS = 8

# Colors for segment boxes (BGR). Cycles if more than 8 segments.
SEGMENT_COLORS = [
  (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255),
  (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]


def _segment_centers_lower_half(h: int, w: int, polygon: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
  """Up to 4 well-spaced segment centers in the lower half of the region (polygon or full image)."""
  mask = polygon_mask_shape((h, w), polygon) if polygon else np.ones((h, w), dtype=np.uint8) * 255
  # Find the bounding box of the mask, then use its lower half
  ys, xs = np.where(mask > 0)
  if len(ys) == 0:
    return []
  my_min, my_max = int(ys.min()), int(ys.max())
  mx_min, mx_max = int(xs.min()), int(xs.max())
  mid_y = (my_min + my_max) // 2
  # Sample a grid inside the mask's lower half, spaced by PATCH_RAD*2 to avoid overlap
  spacing = PATCH_RAD * 2
  candidates = []
  for y in range(max(mid_y, PATCH_RAD), min(my_max, h - PATCH_RAD) + 1, spacing):
    for x in range(max(mx_min + PATCH_RAD, PATCH_RAD), min(mx_max, w - PATCH_RAD) + 1, spacing):
      if mask[y, x] > 0:
        candidates.append((x, y))
  if not candidates:
    # Fallback: any point in mask lower half
    lower_mask = mask.copy()
    lower_mask[:mid_y, :] = 0
    ys2, xs2 = np.where(lower_mask > 0)
    if len(ys2) == 0:
      ys2, xs2 = ys, xs  # last resort: use full mask
    idx = np.linspace(0, len(ys2) - 1, min(MAX_SEGMENTS, len(ys2)), dtype=int)
    return [(int(xs2[i]), int(ys2[i])) for i in idx]
  if len(candidates) <= MAX_SEGMENTS:
    return candidates
  # Pick MAX_SEGMENTS evenly spaced from candidates
  idx = np.linspace(0, len(candidates) - 1, MAX_SEGMENTS, dtype=int)
  return [candidates[i] for i in idx]


def _find_patch_in_image(
  patch: np.ndarray, img_b: np.ndarray, center_x: int, center_y: int, polygon: list[tuple[int, int]] | None = None
) -> tuple[int, int, float] | None:
  """Search for the patch inside the polygon region of img_b. Returns (match_cx, match_cy, confidence) or None."""
  ph, pw = patch.shape[:2]
  if ph == 0 or pw == 0:
    return None
  h, w = img_b.shape[:2]
  if w < pw or h < ph:
    return None
  result = cv2.matchTemplate(img_b, patch, cv2.TM_CCOEFF_NORMED)
  if polygon:
    # Mask the result so only matches whose center falls inside the polygon are considered
    rh, rw = result.shape[:2]
    result_mask = np.zeros((rh, rw), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    # Shift polygon by -pw//2, -ph//2 since result[y,x] corresponds to match center at (x+pw//2, y+ph//2)
    shifted = pts.copy()
    shifted[:, 0] -= pw // 2
    shifted[:, 1] -= ph // 2
    cv2.fillPoly(result_mask, [shifted], 255)
    result = np.where(result_mask > 0, result, -1.0)
  _, max_val, _, (bx, by) = cv2.minMaxLoc(result)
  if max_val < 0.2:
    return None
  match_cx = bx + pw // 2
  match_cy = by + ph // 2
  return (match_cx, match_cy, max_val)


def movement_panel(
  img_a: np.ndarray,
  img_b: np.ndarray,
  polygon: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int, tuple[int, int, int]]]]:
  """
  Take up to 8 segments from the lower half, find in image 2 via template matching.
  Keep only up-right matches (dx > 0, dy < 0). Average those into one main arrow.
  Returns (panel, segment_boxes) where segment_boxes is [(cx_a, cy_a, cx_b, cy_b, color), ...].
  """
  if img_b.shape[:2] != img_a.shape[:2]:
    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
  h, w = img_a.shape[:2]
  img_a_bgr = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR) if len(img_a.shape) == 3 else cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)
  img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY) if len(img_b.shape) == 3 else img_b
  img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY) if len(img_a.shape) == 3 else img_a

  centers = _segment_centers_lower_half(h, w, polygon)
  if not centers:
    out = img_a_bgr.copy()
    cv2.putText(out, "No segments in crop region", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return out, []

  # Match each segment in image 2, keep only up-right (dx > 0, dy < 0)
  matches: list[tuple[int, int, np.ndarray, float, float, float]] = []  # cx, cy, patch_bgr, dx, dy, conf
  for (cx, cy) in centers:
    y1, y2 = max(0, cy - PATCH_RAD), min(h, cy + PATCH_RAD)
    x1, x2 = max(0, cx - PATCH_RAD), min(w, cx + PATCH_RAD)
    patch = img_a_gray[y1:y2, x1:x2]
    if patch.size == 0:
      continue
    match = _find_patch_in_image(patch, img_b_gray, cx, cy, polygon=polygon)
    if match is None:
      continue
    mcx, mcy, conf = match
    dx, dy = float(mcx - cx), float(mcy - cy)
    print(f"  Segment ({cx},{cy}): dx={dx:.0f}, dy={dy:.0f}, conf={conf:.2f}", end="")
    if dx > 0 and dy < 0:
      print(" [up-right, kept]")
      patch_bgr = img_a_bgr[y1:y2, x1:x2].copy()
      matches.append((cx, cy, patch_bgr, dx, dy, conf))
    else:
      print(" [rejected]")

  panel = np.zeros((h, w, 3), dtype=np.uint8)
  panel[:] = (40, 40, 40)

  if not matches:
    cv2.putText(panel, "No up-right matches found", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    return panel, []

  # Outlier filtering: discard matches whose magnitude is <50% or >150% of the median
  mags = [float(np.hypot(m[3], m[4])) for m in matches]
  med_mag = float(np.median(mags))
  if med_mag > 0:
    filtered = [(m, mag) for m, mag in zip(matches, mags) if 0.5 * med_mag <= mag <= 1.5 * med_mag]
    rejected = len(matches) - len(filtered)
    if rejected > 0:
      print(f"  Outlier filter: {rejected} removed (median={med_mag:.1f}, kept 50%-150% range)")
    matches = [m for m, _ in filtered] if filtered else matches

  # Build segment_boxes: (cx_a, cy_a, cx_b, cy_b, color) for each match
  segment_boxes = []
  for i, (cx, cy, _p, dx, dy, _c) in enumerate(matches):
    color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
    segment_boxes.append((cx, cy, int(cx + dx), int(cy + dy), color))

  # Average displacement from all kept matches
  avg_dx = float(np.mean([m[3] for m in matches]))
  avg_dy = float(np.mean([m[4] for m in matches]))
  avg_mag = float(np.hypot(avg_dx, avg_dy))
  avg_conf = float(np.mean([m[5] for m in matches]))
  bales = avg_mag / 450.0
  print(f"  Average: dx={avg_dx:.1f}, dy={avg_dy:.1f}, mag={avg_mag:.1f}px, conf={avg_conf:.2f} ({len(matches)}/{len(centers)} segments)")
  print(f"  Estimated bales: {bales:.2f}")

  # --- Left side: big main arrow (average) ---
  mid_x = w // 4
  mid_y = h // 2
  arrow_scale = min(w * 0.3, h * 0.3) / max(avg_mag, 1.0)
  ux, uy = avg_dx / max(avg_mag, 1e-6), avg_dy / max(avg_mag, 1e-6)
  arrow_len = min(avg_mag * arrow_scale, min(w, h) * 0.35)
  x_s = int(mid_x - ux * arrow_len * 0.4)
  y_s = int(mid_y - uy * arrow_len * 0.4)
  x_e = int(mid_x + ux * arrow_len * 0.6)
  y_e = int(mid_y + uy * arrow_len * 0.6)
  cv2.arrowedLine(panel, (x_s, y_s), (x_e, y_e), (0, 255, 0), 6, tipLength=0.25)
  label = f"{int(round(avg_mag))} px"
  (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
  cv2.putText(panel, label, (mid_x - tw // 2, mid_y + th + 30), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
  bales_label = f"~{bales:.1f} bales"
  (bw, bh), _ = cv2.getTextSize(bales_label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
  cv2.putText(panel, bales_label, (mid_x - bw // 2, mid_y + th + 30 + bh + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3, cv2.LINE_AA)
  info = f"avg of {len(matches)} segments, conf {avg_conf:.2f}"
  (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
  cv2.putText(panel, info, (mid_x - iw // 2, mid_y - int(arrow_len * 0.5) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
  cv2.putText(panel, "Average movement (up-right)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

  # --- Right side: small individual arrows + segment thumbnails ---
  right_x = w // 2 + 10
  n = len(matches)
  rows = (n + 1) // 2  # 2 columns
  cell_h = h // max(rows, 1)
  cell_w = (w // 2 - 20) // 2
  for i, (cx, cy, patch_bgr, dx, dy, conf) in enumerate(matches):
    row, col = i // 2, i % 2
    bx = right_x + col * cell_w
    by = row * cell_h
    # Thumbnail
    ph, pw = patch_bgr.shape[:2]
    thumb_size = min(cell_h - 20, cell_w // 2 - 10, 80)
    sc = thumb_size / max(ph, pw)
    thumb = cv2.resize(patch_bgr, (int(pw * sc), int(ph * sc)))
    ty_off = by + (cell_h - thumb.shape[0]) // 2
    tx_off = bx + 5
    if ty_off + thumb.shape[0] <= h and tx_off + thumb.shape[1] <= w:
      panel[ty_off : ty_off + thumb.shape[0], tx_off : tx_off + thumb.shape[1]] = thumb
    # Small arrow next to thumbnail
    ax = tx_off + thumb.shape[1] + 15
    ay = by + cell_h // 2
    mag = float(np.hypot(dx, dy))
    sux, suy = dx / max(mag, 1e-6), dy / max(mag, 1e-6)
    sal = min(cell_w * 0.3, 40)
    cv2.arrowedLine(panel, (ax, ay), (int(ax + sux * sal), int(ay + suy * sal)), (0, 255, 0), 2, tipLength=0.3)
    slabel = f"{int(round(mag))}px"
    cv2.putText(panel, slabel, (int(ax + sux * sal) + 5, ay + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

  return panel, segment_boxes


def draw_polygon(img: np.ndarray, max_display_side: int = 900) -> list[tuple[int, int]] | None:
  """Let user click points. Returns list of (x,y) in original image coords, or None. Enter=done, r=reset, q=cancel."""
  points: list[tuple[int, int]] = []
  h, w = img.shape[:2]
  scale = min(1.0, max_display_side / max(h, w))
  display_img = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
  if len(display_img.shape) == 2:
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
  win = "Draw polygon (Enter=done, r=reset, q=cancel)"

  def to_image_coords(x: int, y: int) -> tuple[int, int]:
    return (int(round(x / scale)), int(round(y / scale))) if scale < 1.0 else (x, y)

  def redraw():
    canvas = display_img.copy()
    for i, p in enumerate(points):
      disp = (int(p[0] * scale), int(p[1] * scale)) if scale < 1.0 else p
      cv2.circle(canvas, disp, max(2, int(5 * scale)), (0, 255, 0), -1)
      if i > 0:
        prev = (int(points[i - 1][0] * scale), int(points[i - 1][1] * scale)) if scale < 1.0 else points[i - 1]
        cv2.line(canvas, prev, disp, (0, 255, 0), 2)
    if len(points) >= 2:
      first = (int(points[0][0] * scale), int(points[0][1] * scale)) if scale < 1.0 else points[0]
      last = (int(points[-1][0] * scale), int(points[-1][1] * scale)) if scale < 1.0 else points[-1]
      cv2.line(canvas, last, first, (0, 255, 0), 2)
    cv2.putText(canvas, "Click=add point, Enter=done, r=reset, q=cancel", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow(win, canvas)

  def on_mouse(_event, x, y, _flags, _param):
    if _event == cv2.EVENT_LBUTTONDOWN:
      points.append(to_image_coords(x, y))
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


def measure_line(img: np.ndarray, max_display_side: int = 900) -> None:
  """Show image, let user draw a line (click start + click end). Display length in px. Enter/q to close."""
  h, w = img.shape[:2]
  scale = min(1.0, max_display_side / max(h, w))
  dw, dh = int(w * scale), int(h * scale)
  display = cv2.resize(img, (dw, dh)) if scale < 1.0 else img.copy()
  if len(display.shape) == 2:
    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
  base = display.copy()
  points: list[tuple[int, int]] = []
  win = "Measure line (click start + end, r=reset, q/Enter=close)"
  cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

  def redraw():
    canvas = base.copy()
    for p in points:
      cv2.circle(canvas, p, 5, (0, 0, 255), -1)
    if len(points) == 2:
      cv2.line(canvas, points[0], points[1], (0, 255, 0), 2)
      dx = (points[1][0] - points[0][0]) / scale
      dy = (points[1][1] - points[0][1]) / scale
      length = float(np.hypot(dx, dy))
      label = f"{int(round(length))} px"
      mx = (points[0][0] + points[1][0]) // 2
      my = (points[0][1] + points[1][1]) // 2
      cv2.putText(canvas, label, (mx + 10, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
      print(f"  Line length: {length:.1f} px (original image coords)")
    cv2.imshow(win, canvas)
    cv2.waitKey(1)

  def on_mouse(_event, x, y, _flags, _param):
    if _event == cv2.EVENT_LBUTTONDOWN:
      if len(points) >= 2:
        points.clear()
      points.append((x, y))
      redraw()

  cv2.setMouseCallback(win, on_mouse)
  redraw()
  while True:
    key = cv2.waitKey(50) & 0xFF
    if key in (ord("q"), ord("Q"), 13, 10):
      break
    if key in (ord("r"), ord("R")):
      points.clear()
      redraw()
  cv2.destroyWindow(win)


def list_folders_with_images(registry: RegistryBase) -> list[Path]:
  files = registry.backend.LIST(PREFIX)
  folders: dict[Path, list[Path]] = {}
  for f in files:
    if f.suffix.lower() in (".jpg", ".png", ".jpeg"):
      folders.setdefault(f.parent, []).append(f)
  return sorted([p for p, imgs in folders.items() if imgs])


def get_first_image(registry: RegistryBase, folder: Path) -> Path | None:
  files = registry.backend.LIST(str(folder))
  imgs = sorted([f for f in files if f.suffix.lower() in (".jpg", ".png", ".jpeg")])
  return imgs[0] if imgs else None


def build_grid(
  registry: RegistryBase,
  folder_a: Path,
  folder_b: Path,
  polygon: list[tuple[int, int]] | None = None,
  max_display_width: int = 1800,
  max_display_height: int = 1000,
) -> np.ndarray | None:
  path_a = get_first_image(registry, folder_a)
  path_b = get_first_image(registry, folder_b)
  if not path_a or not path_b:
    return None
  img_a = np.array(registry.GET(path_a, registry.DefaultMarkers.ORIGINAL_MARKER))
  img_b = np.array(registry.GET(path_b, registry.DefaultMarkers.ORIGINAL_MARKER))
  motion_binary = motion_image(img_a, img_b)
  if polygon:
    motion_binary = apply_polygon_mask(motion_binary, polygon)
  motion_disp = cv2.cvtColor(motion_binary, cv2.COLOR_GRAY2BGR)
  has_motion = motion_inside_crop(motion_binary, polygon)
  movement_disp, segment_boxes = movement_panel(img_a, img_b, polygon)
  cv2.putText(
    movement_disp, "Motion" if has_motion else "No motion", (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if has_motion else (0, 0, 255), 2
  )

  def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 and img.shape[2] == 3 else img

  a = to_bgr(img_a)
  b = to_bgr(img_b)

  # Draw colored segment rectangles on images A (source) and B (matched position)
  for (cx_a, cy_a, cx_b, cy_b, color) in segment_boxes:
    r = PATCH_RAD
    cv2.rectangle(a, (cx_a - r, cy_a - r), (cx_a + r, cy_a + r), color, 3)
    cv2.rectangle(b, (cx_b - r, cy_b - r), (cx_b + r, cy_b + r), color, 3)
  # Top row: A and B side by side (same height)
  h_top = min(a.shape[0], b.shape[0])
  a = cv2.resize(a, (int(a.shape[1] * h_top / a.shape[0]), h_top))
  b = cv2.resize(b, (int(b.shape[1] * h_top / b.shape[0]), h_top))
  top_row = np.hstack([a, b])
  top_width = top_row.shape[1]
  half_w = top_width // 2
  h_bottom = h_top  # same height as one top image
  motion_row = cv2.resize(motion_disp, (half_w, h_bottom))
  movement_row = cv2.resize(movement_disp, (half_w, h_bottom))
  bottom_row = np.hstack([motion_row, movement_row])
  grid = np.vstack([top_row, bottom_row])
  # Scale entire grid to fit on screen so nothing is cropped
  scale = min(max_display_width / grid.shape[1], max_display_height / grid.shape[0], 1.0)
  grid = cv2.resize(grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale)))
  return grid


def get_motion_for_pair(registry: RegistryBase, folder_a: Path, folder_b: Path) -> np.ndarray | None:
  """Load two images and return motion binary (same size as first image), or None."""
  path_a = get_first_image(registry, folder_a)
  path_b = get_first_image(registry, folder_b)
  if not path_a or not path_b:
    return None
  img_a = np.array(registry.GET(path_a, registry.DefaultMarkers.ORIGINAL_MARKER))
  img_b = np.array(registry.GET(path_b, registry.DefaultMarkers.ORIGINAL_MARKER))
  return motion_image(img_a, img_b)


def main():
  registry = RegistryBase(base=REGISTRY_LOCAL_IP)
  folders = list_folders_with_images(registry)
  if len(folders) < 2:
    print("Need at least 2 folders with images")
    return
  cv2.namedWindow("Original A | Original B | Motion (binary)", cv2.WINDOW_NORMAL)
  pair_index = 0
  polygon = load_polygon()
  while True:
    i, j = pair_index * 2, pair_index * 2 + 1
    if j >= len(folders):
      print("No more folder pairs")
      break
    grid = build_grid(registry, folders[i], folders[j], polygon=polygon)
    if grid is None:
      pair_index += 1
      continue
    cv2.imshow("Original A | Original B | Motion (binary)", grid)
    cv2.setWindowTitle("Original A | Original B | Motion (binary)", f"Pair {pair_index + 1} | n=next p=polygon l=line q=quit")
    key = cv2.waitKey(0) & 0xFF
    if key in (ord("q"), ord("Q")):
      break
    if key in (ord("n"), ord("N")):
      pair_index += 1
    if key in (ord("l"), ord("L")):
      path_b = get_first_image(registry, folders[j])
      if path_b:
        img_b = np.array(registry.GET(path_b, registry.DefaultMarkers.ORIGINAL_MARKER))
        img_b_bgr = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)
        measure_line(img_b_bgr)
    if key in (ord("p"), ord("P")):
      motion = get_motion_for_pair(registry, folders[i], folders[j])
      if motion is not None:
        new_poly = draw_polygon(motion)
        if new_poly is not None:
          save_polygon(new_poly)
          polygon = new_poly
        else:
          polygon = []  # cancel: clear for this session only
        grid = build_grid(registry, folders[i], folders[j], polygon=polygon if polygon else None)
        if grid is not None:
          cv2.imshow("Original A | Original B | Motion (binary)", grid)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
