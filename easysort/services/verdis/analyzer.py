import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm
import urllib.request
from openai import OpenAI

from easysort.common.environment import Env


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OPENAI_API_KEY = Env.OPENAI_API_KEY
OPENAI_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-nano-2025-08-07")


ALLOWED_CATEGORIES = [
    "Cardboard",
    "Paper",
    "Residual",
    "Plastics",
    "Empty",
]

CATEGORIES_DESCRIPTION = (
    "Cardboard — belt mostly brown corrugated sheets/boxes.\n"
    "Paper — belt full of small bright white/light 2D paper scraps.\n"
    "Residual — big soft bags/film blobs (white/black/blue), shiny, amorphous.\n"
    "Plastics — few large rigid objects (hard plastics, e-waste, bins), strong edges.\n"
    "Empty — Belt is more than 95% empty.\n"
)

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


def openai_chat(model: str, prompt: str, images_b64: List[str], timeout: int = 60, client: Optional[OpenAI] = None) -> str:
    """
    Calls OpenAI chat completions via SDK with a multimodal user message (text + base64 image URLs).
    Returns the response content string.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")

    client = client or OpenAI(api_key=OPENAI_API_KEY)
    content = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "text"},
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"OpenAI SDK request failed: {e}")


def build_prompt_group(group_id: str, filenames: List[str], context: Optional[str] = None) -> str:
    cats = ", ".join(ALLOWED_CATEGORIES)
    context_line = ""
    if context:
        context_line = f"Context: The image has been cropped to show only the {context}.\\n"
    return (
        "You are classifying what is on a conveyor belt from a single still image.\n"
        f"{context_line}"
        "Choose EXACTLY one category from: "
        f"{cats}.\n"
        "Use these definitions:\n"
        f"{CATEGORIES_DESCRIPTION}"
        "Rule: If any material covers over 5% of the belt area, it is NOT 'Empty'.\n"
        "Return STRICT JSON ONLY (no extra text):\n"
        "{\n"
        "  \"group_id\": string,\n"
        "  \"category\": string,\n"
        "  \"confidence\": number\n"
        "}\n"
        "Category must be EXACTLY one of: "
        f"{cats}. Confidence is 0..1.\n"
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


def analyze(clips_dir: Path, model: str = DEFAULT_MODEL, force: bool = False, max_px: int = 1024, context: Optional[str] = None, use_openai: bool = False, openai_model: Optional[str] = None, concurrency: int = 8) -> None:
    """
    For a clips folder created by VerdisRunner.analyze (containing an images/ dir),
    analyze each image group via Ollama using a single representative image and
    save a JSON under an ai_pred/ folder next to images/.
    """
    images_dir = clips_dir / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"images directory not found under {clips_dir}")
    out_dir = clips_dir / ("open_ai_pred" if use_openai else "ai_pred")
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_images(images_dir)
    if len(groups) == 0:
        print("No images found in", images_dir)
        return

    print(f"Found {len(groups)} groups in {images_dir}")

    keys_sorted = sorted(groups.keys())
    keys_to_process = []
    for key in keys_sorted:
        out_path = out_dir / f"{key}.json"
        if out_path.exists() and not force:
            continue
        keys_to_process.append(key)

    oai_client: Optional[OpenAI] = None
    if use_openai:
        oai_client = OpenAI(api_key=OPENAI_API_KEY)

    def do_one(key: str) -> None:
        imgs = groups[key]
        out_path = out_dir / f"{key}.json"
        # pick 1 representative image per group
        rep_imgs = _select_one(imgs)
        images_b64: List[str] = []
        filenames: List[str] = []
        for img_path in rep_imgs:
            if context and context in CONTEXT_CROPS:
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
            if use_openai:
                oai_model = openai_model or OPENAI_DEFAULT_MODEL
                content = openai_chat(oai_model, prompt, images_b64, client=oai_client)
            else:
                content = ollama_chat(model, prompt, images_b64)
        except Exception as e:
            src = "openai" if use_openai else "ollama"
            content = json.dumps({"group_id": key, "error": f"{src} request failed: {e}"})

        try:
            data = json.loads(content)
        except Exception:
            data = {"group_id": key, "raw": content}
        with open(out_path, "w") as f:
            json.dump(data, f)

    if use_openai and concurrency > 1 and len(keys_to_process) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(do_one, k): k for k in keys_to_process}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Analyzing groups", unit="group"):
                pass
    else:
        for key in tqdm(keys_to_process, desc="Analyzing groups", unit="group"):
            do_one(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze image groups for clips with local Ollama model.")
    parser.add_argument("--clips_dir", type=str, required=True, help="Directory containing <id>_clips with images/")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name (default from OLLAMA_MODEL or gemma3:4b)")
    parser.add_argument("--force", action="store_true", help="Recompute even if output JSON exists")
    parser.add_argument("--max_px", type=int, default=1024, help="Resize max dimension for encoding")
    parser.add_argument("--context", type=str, default=None, help="Optional context (e.g., 'belt') to apply a predefined crop without downscaling")
    parser.add_argument("--openai", action="store_true", help="Send requests to OpenAI instead of Ollama; saves under open_ai_pred/")
    parser.add_argument("--openai_model", type=str, default=None, help=f"OpenAI model (default from OPENAI_MODEL or {OPENAI_DEFAULT_MODEL})")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel requests when using --openai (default: 8)")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise SystemExit(f"clips_dir not found or not a directory: {clips_dir}")
    analyze(
        clips_dir,
        model=args.model,
        force=args.force,
        max_px=args.max_px,
        context=args.context,
        use_openai=bool(args.openai),
        openai_model=args.openai_model,
        concurrency=int(args.concurrency),
    )


if __name__ == "__main__":
    main()


