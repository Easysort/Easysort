"""
Slug Movement Detector
======================
Given two images from the same fixed camera (taken minutes apart),
this script measures how far the waste bale slug has moved and
estimates the number of bales introduced.

Usage:
    python slug_movement.py image1.png image2.png [--pixels_per_bale 30]

Output:
    - Displacement in pixels along the slug axis
    - Estimated bales moved (fractional)
    - Diagnostic visualization image
"""

import cv2
import numpy as np
import argparse
import sys
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION — Adjust these for your specific camera setup
# =============================================================================

# Region of interest: polygon defining the slug area (in pixel coordinates)
# These are for the "Udendørs Hal 2" camera at 1091x614 resolution.
# Format: list of (x, y) points defining the slug boundary
SLUG_POLYGON = np.array(
  [
    [170, 545],
    [130, 470],
    [250, 380],
    [380, 370],
    [550, 340],
    [680, 330],
    [720, 380],
    [700, 430],
    [620, 500],
    [460, 540],
    [300, 560],
  ],
  dtype=np.int32,
)

# The slug's movement axis direction (unit vector along which bales travel)
# Measured from the image: slug runs from lower-left to upper-right
# Start point (tail of slug) and end point (head, toward the machine)
SLUG_AXIS_START = np.array([170, 505])  # lower-left end
SLUG_AXIS_END = np.array([700, 365])  # upper-right end (toward machine)

# Default pixels per bale (along the slug axis)
# Measured from the image: bales are roughly 28-35 pixels wide along the axis.
# You should calibrate this by counting bales and measuring total pixel span.
DEFAULT_PIXELS_PER_BALE = 30.0

# Reference image dimensions (used to scale ROI if input images differ)
REF_WIDTH = 1091
REF_HEIGHT = 614


def create_slug_mask(shape, polygon):
  """Create a binary mask from the slug polygon."""
  mask = np.zeros(shape[:2], dtype=np.uint8)
  cv2.fillPoly(mask, [polygon], 255)
  return mask


def select_axis_points(image, window_name="Redraw slug axis"):
  points = []

  def on_mouse(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
      if len(points) >= 2:
        points.clear()
      points.append((int(x), int(y)))

  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.setMouseCallback(window_name, on_mouse)

  while True:
    canvas = image.copy()
    if len(points) >= 1:
      cv2.circle(canvas, points[0], 6, (0, 255, 0), -1)
    if len(points) == 2:
      cv2.circle(canvas, points[1], 6, (0, 255, 0), -1)
      cv2.line(canvas, points[0], points[1], (0, 255, 0), 2)
    cv2.putText(
      canvas,
      "Click start and end points. Press Enter to accept, r to reset, q to quit.",
      (15, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
    )
    cv2.imshow(window_name, canvas)

    key = cv2.waitKey(20) & 0xFF
    if key in (ord("q"), ord("Q")):
      cv2.destroyWindow(window_name)
      return None
    if key in (ord("r"), ord("R")):
      points.clear()
    if key in (13, 10, 32) and len(points) == 2:
      cv2.destroyWindow(window_name)
      return np.array(points[0], dtype=float), np.array(points[1], dtype=float)


def scale_polygon(polygon, img_shape, ref_w, ref_h):
  """Scale polygon coordinates if image size differs from reference."""
  h, w = img_shape[:2]
  if w == ref_w and h == ref_h:
    return polygon
  scale_x = w / ref_w
  scale_y = h / ref_h
  scaled = polygon.copy().astype(float)
  scaled[:, 0] *= scale_x
  scaled[:, 1] *= scale_y
  return scaled.astype(np.int32)


def align_images(img1, img2):
  """
  Align img2 to img1 using feature matching on static background.
  Returns the aligned img2 and the transformation matrix.
  """
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  gray1_blur = cv2.GaussianBlur(gray1, (5, 5), 0)
  gray2_blur = cv2.GaussianBlur(gray2, (5, 5), 0)

  orb = cv2.ORB_create(nfeatures=3000)
  kp1, des1 = orb.detectAndCompute(gray1_blur, None)
  kp2, des2 = orb.detectAndCompute(gray2_blur, None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  matches = sorted(matches, key=lambda x: x.distance)

  if len(matches) < 10:
    print("WARNING: Very few feature matches found. Alignment may be unreliable.")
    return img2, np.eye(2, 3, dtype=np.float64)

  pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:200]]).reshape(-1, 1, 2)
  pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:200]]).reshape(-1, 1, 2)

  M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)

  if M is None:
    print("WARNING: Could not estimate alignment transform.")
    return img2, np.eye(2, 3, dtype=np.float64)

  h, w = img1.shape[:2]
  aligned = cv2.warpAffine(img2, M, (w, h))

  n_inliers = np.sum(inliers) if inliers is not None else 0
  print(
    f"  Alignment: dx={M[0, 2]:.2f}px, dy={M[1, 2]:.2f}px, rotation={np.degrees(np.arctan2(M[1, 0], M[0, 0])):.3f}°, inliers={n_inliers}/{len(matches[:200])}"
  )

  return aligned, M


def compute_slug_displacement(img1, img2_aligned, slug_mask, axis_direction):
  """
  Compute the displacement of the slug along its axis using optical flow.

  Returns:
      displacement_px: displacement in pixels along the slug axis (positive = toward machine)
      flow: the raw optical flow field
      stats: dictionary with detailed statistics
  """
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)

  gray1_blur = cv2.GaussianBlur(gray1, (5, 5), 0)
  gray2_blur = cv2.GaussianBlur(gray2, (5, 5), 0)

  # Compute dense optical flow
  flow = cv2.calcOpticalFlowFarneback(gray1_blur, gray2_blur, None, pyr_scale=0.5, levels=5, winsize=21, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)

  # Only consider flow within the slug mask
  mask_bool = slug_mask > 0

  flow_x = flow[..., 0][mask_bool]
  flow_y = flow[..., 1][mask_bool]
  magnitudes = np.sqrt(flow_x**2 + flow_y**2)

  # Filter: only consider pixels with significant motion (> 1 pixel)
  sig_mask = magnitudes > 1.0

  if np.sum(sig_mask) < 100:
    print("WARNING: Very few pixels with significant motion detected.")
    return 0.0, flow, {"significant_pixels": int(np.sum(sig_mask))}

  flow_x_sig = flow_x[sig_mask]
  flow_y_sig = flow_y[sig_mask]

  # Project flow onto the slug axis direction
  # axis_direction is a unit vector pointing from tail to head of slug
  axis_unit = axis_direction / np.linalg.norm(axis_direction)

  # For each pixel, compute the component of flow along the slug axis
  projections = flow_x_sig * axis_unit[0] + flow_y_sig * axis_unit[1]

  # Use median for robustness (less sensitive to outliers than mean)
  displacement_median = float(np.median(projections))
  displacement_mean = float(np.mean(projections))

  # Also compute using percentile-trimmed mean (remove top/bottom 10%)
  p10, p90 = np.percentile(projections, [10, 90])
  trimmed = projections[(projections >= p10) & (projections <= p90)]
  displacement_trimmed = float(np.mean(trimmed)) if len(trimmed) > 0 else displacement_median

  stats = {
    "significant_pixels": int(np.sum(sig_mask)),
    "total_slug_pixels": int(np.sum(mask_bool)),
    "fraction_moving": float(np.sum(sig_mask)) / float(np.sum(mask_bool)),
    "median_displacement_px": displacement_median,
    "mean_displacement_px": displacement_mean,
    "trimmed_mean_displacement_px": displacement_trimmed,
    "mean_raw_dx": float(np.mean(flow_x_sig)),
    "mean_raw_dy": float(np.mean(flow_y_sig)),
    "median_magnitude": float(np.median(magnitudes[sig_mask])),
    "max_magnitude": float(np.max(magnitudes[sig_mask])),
  }

  # Use trimmed mean as the primary estimate (robust but uses more data than median)
  return displacement_trimmed, flow, stats


def create_diagnostic_image(img1, img2_aligned, flow, slug_mask, slug_polygon, axis_start, axis_end, displacement_px, bales_moved, stats, output_path):
  """Create a diagnostic visualization."""
  img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
  img2_rgb = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)

  fig, axes = plt.subplots(2, 2, figsize=(20, 14))
  fig.suptitle(f"Slug Movement Analysis — {bales_moved:+.2f} bales ({displacement_px:+.1f} px)", fontsize=18, fontweight="bold")

  # Top-left: Image 1 with slug outline
  axes[0, 0].imshow(img1_rgb)
  poly = plt.Polygon(slug_polygon, fill=False, edgecolor="lime", linewidth=2)
  axes[0, 0].add_patch(poly)
  axes[0, 0].set_title("Image 1 with slug region", fontsize=13)
  axes[0, 0].axis("off")

  # Top-right: Image 2 aligned
  axes[0, 1].imshow(img2_rgb)
  poly2 = plt.Polygon(slug_polygon, fill=False, edgecolor="lime", linewidth=2)
  axes[0, 1].add_patch(poly2)
  axes[0, 1].set_title("Image 2 (aligned)", fontsize=13)
  axes[0, 1].axis("off")

  # Bottom-left: Signed difference in slug region
  gray1 = cv2.GaussianBlur(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (5, 5), 0)
  gray2 = cv2.GaussianBlur(cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY), (5, 5), 0)
  signed_diff = gray2.astype(np.float32) - gray1.astype(np.float32)

  overlay = img1_rgb.copy().astype(np.float32)
  mask_bool = slug_mask > 0
  darker = (signed_diff < -10) & mask_bool
  lighter = (signed_diff > 10) & mask_bool
  overlay[darker] = overlay[darker] * 0.4 + np.array([255, 30, 30]) * 0.6
  overlay[lighter] = overlay[lighter] * 0.4 + np.array([30, 100, 255]) * 0.6

  axes[1, 0].imshow(overlay.astype(np.uint8))
  axes[1, 0].set_title("Red = material arrived | Blue = material departed", fontsize=13)
  axes[1, 0].axis("off")

  # Bottom-right: Motion vectors in slug region
  h, w = img1.shape[:2]
  step = 15
  Y, X = np.mgrid[0:h:step, 0:w:step]
  U = flow[::step, ::step, 0]
  V = flow[::step, ::step, 1]
  mag = np.sqrt(U**2 + V**2)
  mask_grid = (slug_mask[::step, ::step] > 0) & (mag > 1.0)

  axes[1, 1].imshow(img1_rgb)
  if np.any(mask_grid):
    axes[1, 1].quiver(
      X[mask_grid],
      Y[mask_grid],
      U[mask_grid] * 8,
      V[mask_grid] * 8,
      mag[mask_grid],
      cmap="hot",
      angles="xy",
      scale_units="xy",
      scale=1,
      width=0.003,
      headwidth=4,
    )

  # Draw movement axis arrow
  axes[1, 1].annotate("", xy=axis_end, xytext=axis_start, arrowprops=dict(arrowstyle="->", color="lime", lw=2.5))

  axes[1, 1].set_title(f"Motion vectors (8x magnified) | {stats['significant_pixels']:,} moving pixels ({stats['fraction_moving']:.0%})", fontsize=12)
  axes[1, 1].axis("off")

  # Add text summary
  summary = (
    f"Displacement along axis: {displacement_px:+.2f} px\n"
    f"Bales moved: {bales_moved:+.3f}\n"
    f"Median magnitude: {stats['median_magnitude']:.1f} px\n"
    f"Raw mean flow: dx={stats['mean_raw_dx']:.2f}, dy={stats['mean_raw_dy']:.2f}"
  )
  fig.text(0.02, 0.02, summary, fontsize=11, family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

  plt.tight_layout(rect=[0, 0.06, 1, 1])
  plt.savefig(output_path, dpi=150, bbox_inches="tight")
  plt.close()


def analyze_slug_movement(img1_path, img2_path, pixels_per_bale=DEFAULT_PIXELS_PER_BALE, output_dir=None, diagnostic=True, axis_start=None, axis_end=None):
  """
  Main analysis function.

  Args:
      img1_path: Path to first (earlier) image
      img2_path: Path to second (later) image
      pixels_per_bale: Calibration value - how many pixels along the slug axis = 1 bale
      output_dir: Where to save diagnostic image (default: same dir as img1)
      diagnostic: Whether to generate diagnostic visualization

  Returns:
      dict with results:
          - displacement_px: pixel displacement along slug axis
          - bales_moved: estimated fractional bales introduced
          - stats: detailed statistics
  """
  print(f"Loading images...")
  img1 = cv2.imread(img1_path)
  img2 = cv2.imread(img2_path)

  if img1 is None:
    raise FileNotFoundError(f"Could not load image: {img1_path}")
  if img2 is None:
    raise FileNotFoundError(f"Could not load image: {img2_path}")

  print(f"  Image 1: {img1.shape[1]}x{img1.shape[0]}")
  print(f"  Image 2: {img2.shape[1]}x{img2.shape[0]}")

  # Resize img2 to match img1 if needed
  if img1.shape[:2] != img2.shape[:2]:
    print(f"  Resizing image 2 to match image 1...")
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

  # Scale ROI polygon if image size differs from reference
  polygon = scale_polygon(SLUG_POLYGON, img1.shape, REF_WIDTH, REF_HEIGHT)
  if axis_start is None or axis_end is None:
    axis_start = SLUG_AXIS_START.copy().astype(float)
    axis_end = SLUG_AXIS_END.copy().astype(float)
    if img1.shape[1] != REF_WIDTH or img1.shape[0] != REF_HEIGHT:
      sx = img1.shape[1] / REF_WIDTH
      sy = img1.shape[0] / REF_HEIGHT
      axis_start[0] *= sx
      axis_start[1] *= sy
      axis_end[0] *= sx
      axis_end[1] *= sy

  # Compute axis direction
  axis_direction = axis_end - axis_start

  # Create slug mask
  slug_mask = create_slug_mask(img1.shape, polygon)

  # Align images
  print(f"Aligning images...")
  img2_aligned, transform = align_images(img1, img2)

  # Compute displacement
  print(f"Computing optical flow...")
  displacement_px, flow, stats = compute_slug_displacement(img1, img2_aligned, slug_mask, axis_direction)

  # Convert to bales
  bales_moved = displacement_px / pixels_per_bale

  # Print results
  print(f"\n{'=' * 50}")
  print(f"RESULTS")
  print(f"{'=' * 50}")
  print(f"  Displacement along slug axis: {displacement_px:+.2f} pixels")
  print(f"  Pixels per bale (calibration): {pixels_per_bale:.1f}")
  print(f"  Bales moved: {bales_moved:+.3f}")
  print(f"  Moving pixels: {stats['significant_pixels']:,} / {stats['total_slug_pixels']:,} ({stats['fraction_moving']:.1%})")
  print(f"  Median flow magnitude: {stats['median_magnitude']:.2f} px")
  print(f"{'=' * 50}")

  # Generate diagnostic image
  if diagnostic:
    if output_dir is None:
      output_dir = os.path.dirname(os.path.abspath(img1_path))
    os.makedirs(output_dir, exist_ok=True)

    diag_path = os.path.join(output_dir, "slug_diagnostic.png")
    print(f"\nSaving diagnostic image to: {diag_path}")
    create_diagnostic_image(
      img1, img2_aligned, flow, slug_mask, polygon, axis_start.astype(int), axis_end.astype(int), displacement_px, bales_moved, stats, diag_path
    )

  return {
    "displacement_px": displacement_px,
    "bales_moved": bales_moved,
    "pixels_per_bale": pixels_per_bale,
    "stats": stats,
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Measure waste slug movement between two images")
  parser.add_argument("image1", help="Path to first (earlier) image")
  parser.add_argument("image2", help="Path to second (later) image")
  parser.add_argument(
    "--pixels_per_bale", type=float, default=DEFAULT_PIXELS_PER_BALE, help=f"Pixels per bale along slug axis (default: {DEFAULT_PIXELS_PER_BALE})"
  )
  parser.add_argument("--output_dir", type=str, default=None, help="Directory for diagnostic output (default: same as image1)")
  parser.add_argument("--no_diagnostic", action="store_true", help="Skip generating diagnostic image")
  parser.add_argument("--redraw", action="store_true", help="Interactively redraw the slug axis on image1")

  args = parser.parse_args()

  axis_start = None
  axis_end = None
  if args.redraw:
    img = cv2.imread(args.image1)
    if img is None:
      raise FileNotFoundError(f"Could not load image: {args.image1}")
    selection = select_axis_points(img)
    if selection is None:
      sys.exit(0)
    axis_start, axis_end = selection
    sx = REF_WIDTH / img.shape[1]
    sy = REF_HEIGHT / img.shape[0]
    ref_start = (axis_start[0] * sx, axis_start[1] * sy)
    ref_end = (axis_end[0] * sx, axis_end[1] * sy)
    print("Redraw result (image coords):")
    print(f"  SLUG_AXIS_START = np.array([{axis_start[0]:.0f}, {axis_start[1]:.0f}])")
    print(f"  SLUG_AXIS_END   = np.array([{axis_end[0]:.0f}, {axis_end[1]:.0f}])")
    print("Redraw result (ref coords):")
    print(f"  SLUG_AXIS_START = np.array([{ref_start[0]:.0f}, {ref_start[1]:.0f}])")
    print(f"  SLUG_AXIS_END   = np.array([{ref_end[0]:.0f}, {ref_end[1]:.0f}])")

  result = analyze_slug_movement(
    args.image1,
    args.image2,
    pixels_per_bale=args.pixels_per_bale,
    output_dir=args.output_dir,
    diagnostic=not args.no_diagnostic,
    axis_start=axis_start,
    axis_end=axis_end,
  )
