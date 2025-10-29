import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm
import urllib.request


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")


ALLOWED_CATEGORIES = [
    "cardboard/paper",
    "plastics (tubes, buckets, mixed plastics)",
    "residual fraction (very mixed waste, looks white, but not paper though)",
    "empty",
]

# Optional predefined crops by context
# Coordinates are for full-size images in pixels
CONTEXT_CROPS = {
    "belt": {"x": 862, "y": 45, "w": 332, "h": 1076},
}


def encode_image_b64(path: Path, max_px: int = 1024, quality: int = 85) -> Tuple[str, str]:
    """
    Returns (media_type, base64_data)
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max(w, h) / max_px
        if scale > 1:
            im = im.resize((int(w / scale), int(h / scale)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return "image/jpeg", base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_pil_image_b64(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> Tuple[str, str]:
    im = image.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format=fmt, quality=quality, optimize=(fmt == "JPEG"))
    media_type = "image/jpeg" if fmt == "JPEG" else "image/png"
    return media_type, base64.b64encode(buf.getvalue()).decode("utf-8")


def ollama_chat(model: str, prompt: str, images_b64: List[str], timeout: int = 60) -> str:
    """
    Calls Ollama local API for multimodal chat.
    Uses /api/chat with a single user message containing multiple images.
    Returns the response content string.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    # expected structure: {message: {content: "..."}}
    msg = body.get("message", {}).get("content", "")
    return msg


def build_prompt_group(group_id: str, filenames: List[str], context: Optional[str] = None) -> str:
    cats = " | ".join(ALLOWED_CATEGORIES)
    context_line = ""
    if context:
        context_line = f"Context: The image has been cropped to show only the {context}.\\n"
    return (
        "You are classifying what is on a conveyor belt from a single still image.\n"
        f"{context_line}"
        "Choose EXACTLY one category from: \n"
        f"{cats}.\n"
        "Rule: If any material covers over 5% of the belt area, it is NOT 'empty'.\n"
        "Return STRICT JSON ONLY (no extra text):\n"
        "{\n"
        "  \"group_id\": string,\n"
        "  \"category\": string,\n"
        "  \"confidence\": number\n"
        "}\n"
        "Category must be from the provided list; confidence is 0..1.\n"
        f"Image filename: {filenames}.\n"
        "Return only JSON."
    )


def group_images(images_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in sorted(images_dir.glob("*.jpg")):
        name = p.name
        if name.startswith("._") or name.startswith("."):
            continue
        # assume pattern like HH_MM_SS_00.jpg, group key is before last underscore
        stem = p.stem
        if "_" not in stem:
            key = stem
        else:
            key = stem.rsplit("_", 1)[0]
        groups.setdefault(key, []).append(p)
    return groups


def _select_one(paths: List[Path]) -> List[Path]:
    if not paths:
        return []
    mid = len(paths) // 2
    return [paths[mid]]


def analyze(clips_dir: Path, model: str = DEFAULT_MODEL, force: bool = False, max_px: int = 1024, context: Optional[str] = None) -> None:
    """
    For a clips folder created by VerdisRunner.analyze (containing an images/ dir),
    analyze each image group via Ollama using a single representative image and
    save a JSON under an ai_pred/ folder next to images/.
    """
    images_dir = clips_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"images directory not found under {clips_dir}")
    out_dir = clips_dir / "ai_pred"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_images(images_dir)
    if len(groups) == 0:
        print("No images found in", images_dir)
        return

    print(f"Found {len(groups)} groups in {images_dir}")
    for key in tqdm(sorted(groups.keys()), desc="Analyzing groups", unit="group"):
        imgs = groups[key]
        # output file per group (one JSON per video/clip)
        out_path = out_dir / f"{key}.json"
        if out_path.exists() and not force:
            continue

        # pick 1 representative image per group
        rep_imgs = _select_one(imgs)
        images_b64: List[str] = []
        filenames: List[str] = []
        for img_path in rep_imgs:
            if context and context in CONTEXT_CROPS:
                # Crop without downscaling for maximum fidelity
                crop = CONTEXT_CROPS[context]
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    x = max(0, min(crop["x"], max(0, W - 1)))
                    y = max(0, min(crop["y"], max(0, H - 1)))
                    w = max(1, min(crop["w"], max(1, W - x)))
                    h = max(1, min(crop["h"], max(1, H - y)))
                    cropped = im.crop((x, y, x + w, y + h))
                    _, b64 = encode_pil_image_b64(cropped, fmt="JPEG", quality=90)
                images_b64.append(b64)
                filenames.append(img_path.name)
            else:
                _, b64 = encode_image_b64(img_path, max_px=max_px)
                images_b64.append(b64)
                filenames.append(img_path.name)

        prompt = build_prompt_group(key, filenames, context=context)
        try:
            content = ollama_chat(model, prompt, images_b64)
        except Exception as e:
            content = json.dumps({"group_id": key, "error": f"ollama request failed: {e}"})

        # attempt to parse, but save raw if not
        try:
            data = json.loads(content)
        except Exception:
            data = {"group_id": key, "raw": content}

        with open(out_path, "w") as f:
            json.dump(data, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze image groups for clips with local Ollama model.")
    parser.add_argument("--clips_dir", type=str, required=True, help="Directory containing <id>_clips with images/")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name (default from OLLAMA_MODEL or gemma3:4b)")
    parser.add_argument("--force", action="store_true", help="Recompute even if output JSON exists")
    parser.add_argument("--max_px", type=int, default=1024, help="Resize max dimension for encoding")
    parser.add_argument("--context", type=str, default=None, help="Optional context (e.g., 'belt') to apply a predefined crop without downscaling")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise SystemExit(f"clips_dir not found or not a directory: {clips_dir}")
    analyze(clips_dir, model=args.model, force=args.force, max_px=args.max_px, context=args.context)


if __name__ == "__main__":
    main()


