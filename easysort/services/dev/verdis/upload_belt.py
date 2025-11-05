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
    client.storage.from_(bucket).upload(
        object_key,
        body,
        file_options={"content-type": "application/json", "upsert": True},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload belt motion results to Supabase easytrack bucket")
    ap.add_argument("--json", type=str, required=True, help="Path to input JSON list of {group_id, motion, ai_category}")
    ap.add_argument("--threshold", type=float, default=0.2, help="Motion threshold (default 0.2)")
    ap.add_argument("--dry", action="store_true", help="Print result and key instead of uploading")
    args = ap.parse_args()

    src = Path(args.json)
    date_key = parse_date_from_filename(src)
    out = load_and_transform(src, threshold=float(args.threshold))

    bucket = "easytrack"
    object_key = f"verdis/belt/{date_key}.json"

    if args.dry:
        print(json.dumps({"bucket": bucket, "key": object_key, "items": out}, ensure_ascii=False, indent=2))
        return

    upload_json(bucket, object_key, out)
    print(f"Uploaded {len(out)} items to {bucket}/{object_key}")


if __name__ == "__main__":
    main()


