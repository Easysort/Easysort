import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict

from easysort.common.environment import Env
from supabase import create_client


def parse_date_from_filename(path: Path) -> str:
    """Return dd_mm_yyyy derived from filenames like
    ARGO_ch5_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.json
    """
    stem = path.stem
    m = re.search(r"_(\d{14})_", stem)
    if not m:
        # fallback: try first 8 digits anywhere
        m = re.search(r"(\d{8})", stem)
        if not m:
            raise ValueError(f"Could not parse date from filename: {path.name}")
        ymd = m.group(1)
    else:
        ymd = m.group(1)[:8]
    dt = datetime.strptime(ymd, "%Y%m%d")
    return dt.strftime("%d_%m_%Y")


def category_to_key(category: str) -> str:
    """Convert category name to snake_case key format.
    E.g., "Hard plastics" -> "hard_plastics"
    """
    return category.lower().replace(" ", "_")


def load_and_transform(src: Path, threshold: float = 0.2) -> List[Dict[str, Any]]:
    text = src.read_text()
    # Allow files that are either a raw JSON array or newline/spacing variants
    data = json.loads(text)
    out: List[Dict[str, Any]] = []
    for item in data:
        gid = str(item.get("group_id"))
        motion = float(item.get("motion", 0.0))
        ai = item.get("ai_category")
        out.append({
            "group_id": gid,
            "ai_category": ai,
            "motion": bool(motion >= threshold),
        })
    return out


def upload_json(bucket: str, object_key: str, payload: Any) -> None:
    client = create_client(Env.SUPABASE_URL, Env.SUPABASE_KEY)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    # Prefer upsert to overwrite existing for the day
    # storage3 expects header-like strings for options; pass strings not bools
    client.storage.from_(bucket).upload(
        object_key,
        body,
        file_options={"contentType": "application/json", "upsert": "true"},
    )


def compute_info(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute high-level decimal percentages from uploaded items.
    Generates keys for all categories from prompts.py: Plastics, Hard plastics, Tubes,
    Cardboard, Paper, Folie, Empty - each with _running and _not_running variants.
    """
    # Categories from prompts.py
    CATEGORIES = [
        "Plastics",
        "Hard plastics",
        "Tubes",
        "Cardboard",
        "Paper",
        "Folie",
        "Empty",
    ]
    
    # Generate all possible keys
    KEYS = []
    for cat in CATEGORIES:
        key_base = category_to_key(cat)
        KEYS.append(f"{key_base}_not_running")
        KEYS.append(f"{key_base}_running")
    
    info: Dict[str, Any] = {k: 0 for k in KEYS}
    
    for item in items:
        running = bool(item["motion"])
        category = str(item["ai_category"]).strip()
        
        # Normalize category name (handle variations)
        category_normalized = category
        # Try to match against known categories (case-insensitive)
        for cat in CATEGORIES:
            if category.lower() == cat.lower():
                category_normalized = cat
                break
        
        key_base = category_to_key(category_normalized)
        key = f"{key_base}_{'running' if running else 'not_running'}"
        
        # Only increment if it's a known category key
        if key in info:
            info[key] += 1
    
    # Convert to percentages
    total = len(items)
    if total > 0:
        info = {k: round(info[k] / total, 4) for k in KEYS}
    else:
        info = {k: 0.0 for k in KEYS}
    
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload belt motion results to Supabase easytrack bucket")
    ap.add_argument("--path", type=str, required=True, help="Path to JSON file or directory containing ARGO_ch5*.json files")
    ap.add_argument("--threshold", type=float, default=0.2, help="Motion threshold (default 0.2)")
    ap.add_argument("--dry", action="store_true", help="Print results and keys instead of uploading")
    args = ap.parse_args()

    def process_one(src: Path) -> None:
        date_key = parse_date_from_filename(src)
        out = load_and_transform(src, threshold=float(args.threshold))
        bucket = "easytrack"
        object_key = f"verdis/belt/{date_key}.json"
        object_key_info = f"verdis/belt/{date_key}_info.json"
        info_payload = compute_info(out)

        if args.dry:
            print(json.dumps({
                "file": str(src),
                "bucket": bucket,
                "key": object_key,
                "items_count": len(out),
                "info_key": object_key_info,
                "info": info_payload,
            }, ensure_ascii=False, indent=2))
            return

        upload_json(bucket, object_key, out)
        upload_json(bucket, object_key_info, info_payload)
        print(f"Uploaded {len(out)} items to {bucket}/{object_key} from {src.name}")
        print(f"Uploaded info summary to {bucket}/{object_key_info}")

    p = Path(args.path)
    if p.is_dir():
        files = sorted([f for f in p.iterdir() if f.is_file() and f.name.startswith("ARGO_ch5") and f.suffix.lower() == ".json"])
        if not files:
            print(f"No ARGO_ch5*.json files found in {p}")
            return
        for f in files:
            process_one(f)
    else:
        process_one(p)


if __name__ == "__main__":
    main()


