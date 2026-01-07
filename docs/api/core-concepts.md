# Easysort Core Concepts Reference

This document provides an overview of the core modules in the Easysort project.

---

## Registry (`easysort/registry.py`)

The **Registry** is a centralized data management system that handles file storage, synchronization, and retrieval. It acts as the single source of truth for all data in Easysort.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Registry Path** | The local filesystem path where all registry data is stored |
| **Projects** | Logical groupings of data, tracked in `projects.txt` |
| **Supabase Sync** | Cloud storage integration for syncing data across machines |

### Core Operations

- **`SYNC(allow_special_cleanup=False)`** – Synchronizes local registry with Supabase cloud storage. Downloads missing files (excluding `.jpg` files) using parallel downloads. Cleans up old videos older than 2 weeks. Set `allow_special_cleanup=True` to also delete `.jpg` files from remote storage. Missing files are logged to `missing_files.txt`. Deletions are batched (100 files at a time) to avoid rate limiting.

- **`GET(key, loader)`** – Retrieves data from the registry. Supports automatic loading for `.json`, `.npy`, and raw bytes.

- **`POST(key, data)`** – Stores data in the registry. Automatically handles serialization for `dict`, `list`, `np.ndarray`, and `bytes`.

- **`LIST(prefix, suffix)`** – Lists all files matching a prefix/suffix pattern. Defaults to `.mp4` files.

- **`EXISTS(key)`** – Checks if a file or directory exists in the registry.

### Path Helpers

- `_registry_path(path)` – Converts a relative key to an absolute registry path.
- `_unregistry_path(path)` – Converts an absolute path back to a registry key.
- `construct_path(path, model, project, identifier)` – Builds a structured path for storing model results.

### Usage Example

```python
from easysort.registry import Registry

# List all MP4 files under a prefix
videos = Registry.LIST("argo/camera-01", suffix=".mp4")

# Store analysis results
Registry.POST("argo/camera-01/2025/12/10/results", {"count": 42})

# Retrieve stored data
data = Registry.GET("argo/camera-01/2025/12/10/results.json")
```

---

## GPT Trainer (`easysort/gpt_trainer.py`)

The **GPTTrainer** provides an interface for making batch OpenAI API calls with image inputs, useful for vision-based classification and labeling tasks.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Batch Processing** | Uses `ThreadPoolExecutor` for parallel API calls |
| **Structured Output** | Returns responses as typed dataclasses |
| **Image Encoding** | Converts numpy arrays to base64-encoded JPEGs |

### GPTTrainer Class

```python
class GPTTrainer:
    def __init__(self, model: str = "gpt-5-2025-08-07")
```

Initializes the OpenAI client and validates the API key.

### `_openai_call()` Method

```python
def _openai_call(
    model: str,
    prompt: str,
    image_paths: List[List[np.ndarray]],
    output_schema: dataclass,
    max_workers: int = 10
) -> List[dataclass]
```

Makes parallel API calls to OpenAI's vision model.

| Parameter | Description |
|-----------|-------------|
| `model` | OpenAI model identifier |
| `prompt` | Text prompt sent with each image set |
| `image_paths` | List of image arrays (each item is a list of images for one call) |
| `output_schema` | Dataclass defining the expected JSON response structure |
| `max_workers` | Maximum parallel API calls (default: 10) |

### YoloTrainer Class

A wrapper around Ultralytics YOLO for object detection tasks.

```python
class YoloTrainer:
    def __init__(self, model_path: str = "yolov8m.pt")
```

---

## Cropper (`easysort/cropper.py`)

The **Cropper** is a utility script for testing and calibrating crop regions on camera frames.

### Purpose

When processing video frames or images, you often need to crop to a specific region of interest. This script helps you:

1. Load a sample frame from the registry (supports both video files and JPG images)
2. Apply a test crop
3. Save the result for visual inspection

### Crop Definition

```python
from easysort.sampler import Crop

crop = Crop(x=0, y=0, w=1000, h=1000)
```

| Field | Description |
|-------|-------------|
| `x` | Left edge offset (pixels) |
| `y` | Top edge offset (pixels) |
| `w` | Width of crop region |
| `h` | Height of crop region |

### Workflow

```python
from easysort.sampler import Sampler, Crop
from easysort.registry import Registry
import cv2

# Define crop region
crop = Crop(x=0, y=0, w=1000, h=1000)

# Load and crop a frame (works with both .mp4 and .jpg files)
path = Registry._registry_path("argo/camera/2025/12/10/08/photo.jpg")
frames = Sampler.unpack(path, crop=crop)

# Save for inspection
cv2.imwrite("cropped_output.jpg", frames[0])
```

> **Note:** `Sampler.unpack()` automatically detects the file type. For JPG files, it returns a single-element list with the cropped image. For video files, it extracts all frames.

---

## Viewer (`easysort/viewer.py`)

The **ImageViewer** is a lightweight Flask-based web server for browsing large collections of images.

### Key Features

- **Grid Layout** – Displays images in an 8-column responsive grid
- **Pagination** – Handles large image sets with configurable page sizes (default: 1000 per page)
- **Lazy Loading** – Images load on-demand as they scroll into view
- **Format Support** – JPG, JPEG, PNG, WebP, GIF

### Usage

```python
from easysort.viewer import ImageViewer

# List of image paths
paths = ["/path/to/image1.jpg", "/path/to/image2.png", ...]

# Start the viewer (runs Flask server on port 8000)
ImageViewer(paths, title="My Image Collection")
```

### URL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `page` | 1 | Current page number |
| `per_page` | 1000 | Images per page |

### Routes

| Route | Description |
|-------|-------------|
| `/` | Main grid view with pagination |
| `/img/<idx>` | Direct access to image by index |

### Example Integration

```python
from easysort.registry import Registry
from easysort.viewer import ImageViewer

# View all JPGs from a specific camera
paths = Registry.LIST("argo/camera-01/2025/12", suffix=".jpg")
full_paths = [Registry._registry_path(p) for p in paths]

ImageViewer(full_paths, title="Camera 01 - December 2025")
```

---

## Module Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                        Registry                             │
│  (Central data store: GET, POST, LIST, SYNC)               │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Sampler     │    │  GPTTrainer   │    │  ImageViewer  │
│ (Video/Image) │    │ (Vision API)  │    │ (Browse imgs) │
└───────────────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌───────────────┐
│   Cropper     │
│ (Calibration) │
└───────────────┘
```

The **Registry** is the foundation—all modules read from and write to it. The **Sampler** extracts frames from videos or loads JPG images, with optional cropping. The **Cropper** helps calibrate those crops. **GPTTrainer** processes images through vision models. **ImageViewer** provides visual inspection of results.
