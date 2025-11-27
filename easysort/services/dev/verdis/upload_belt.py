import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Tuple, Union

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


def parse_start_end_from_filename(path: Path) -> Tuple[datetime, datetime] | None:
    """
    Try to extract start and end datetimes from filenames like:
    ARGO_ch5_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.json
    """
    stem = path.stem
    m = re.search(r"_(\d{14})_(\d{14})", stem)
    if not m:
        return None
    start = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    end = datetime.strptime(m.group(2), "%Y%m%d%H%M%S")
    return (start, end)


def _parse_gid_dt(base_date: datetime | None, gid: str) -> datetime | None:
    """Parse group_id formatted as HH_MM_SS on base_date."""
    if base_date is None:
        return None
    m = re.fullmatch(r"(\d{1,2})_(\d{1,2})_(\d{1,2})", str(gid).strip())
    if not m:
        return None
    h, mnt, s = map(int, m.groups())
    try:
        return base_date.replace(hour=h, minute=mnt, second=s, microsecond=0)
    except Exception:
        try:
            # Safer construction
            return datetime(
                year=base_date.year,
                month=base_date.month,
                day=base_date.day,
                hour=h,
                minute=mnt,
                second=s,
            )
        except Exception:
            return None


def infer_item_dt(src: Path, item: Dict[str, Any], idx: int, total: int, base_date: datetime | None = None) -> datetime | None:
    """Infer a timestamp for an item using explicit fields or filename start/end span."""
    # Prefer group_id parsing if available
    gid = item.get("group_id")
    dt_from_gid = _parse_gid_dt(base_date, gid) if gid is not None else None
    if dt_from_gid is not None:
        return dt_from_gid
    for key in ("timestamp", "time", "ts", "datetime", "time_utc"):
        if key in item:
            val = item[key]
            try:
                if isinstance(val, (int, float)):
                    # Interpret as seconds in local time for consistency with provided date
                    return datetime.fromtimestamp(float(val))
                if isinstance(val, str):
                    s = val.strip().replace("Z", "")
                    try:
                        return datetime.fromisoformat(s)
                    except Exception:
                        pass
                    if re.fullmatch(r"\d{14}", s):
                        return datetime.strptime(s, "%Y%m%d%H%M%S")
            except Exception:
                pass
    se = parse_start_end_from_filename(src)
    if se:
        start, end = se
        if total <= 1:
            return start
        frac = max(0.0, min(1.0, idx / (total - 1)))
        return start + (end - start) * frac
    return None


def category_to_key(category: str) -> str:
    """Convert category name to snake_case key format.
    E.g., "Hard plastics" -> "hard_plastics"
    """
    return category.lower().replace(" ", "_")


def load_and_transform(
    src: Path,
    threshold: float = 0.2,
    skip_windows: List[Tuple[datetime, datetime]] | None = None,
    return_debug: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    text = src.read_text()
    # Allow files that are either a raw JSON array or newline/spacing variants
    data = json.loads(text)
    out: List[Dict[str, Any]] = []
    total = len(data)
    hits = [0] * (len(skip_windows) if skip_windows else 0)
    inferred_source = None
    # Base date from filename for group_id parsing
    date_key = parse_date_from_filename(src)
    base_date = datetime.strptime(date_key, "%d_%m_%Y")
    for idx, item in enumerate(data):
        # Determine if item falls in a skip window
        if skip_windows:
            it = infer_item_dt(src, item, idx, total, base_date=base_date)
            # Track inference source (best-effort, for debug)
            if inferred_source is None:
                for key in ("timestamp", "time", "ts", "datetime", "time_utc"):
                    if key in item:
                        inferred_source = f"item_field:{key}"
                        break
                if inferred_source is None and _parse_gid_dt(base_date, item.get("group_id", "")) is not None:
                    inferred_source = "group_id"
                if inferred_source is None and parse_start_end_from_filename(src):
                    inferred_source = "filename_span_interpolation"
            if it is not None:
                matched_index = None
                for wi, (wstart, wend) in enumerate(skip_windows):
                    if wstart <= it < wend:
                        matched_index = wi
                        break
                if matched_index is not None:
                    if hits:
                        hits[matched_index] += 1
                    continue
        gid = str(item.get("group_id"))
        motion = float(item.get("motion", 0.0))
        ai = item.get("ai_category")
        out.append({
            "group_id": gid,
            "ai_category": ai,
            "motion": bool(motion >= threshold),
        })
    if return_debug:
        debug_info: Dict[str, Any] = {
            "total_items": total,
            "kept_items": len(out),
            "skipped_in_windows": sum(hits) if hits else 0,
            "window_hits": [
                {
                    "window": {"start": w[0].isoformat(), "end": w[1].isoformat()},
                    "count": hits[i],
                }
                for i, w in enumerate(skip_windows or [])
            ],
            "inferred_time_source": inferred_source or "none",
        }
        return out, debug_info
    return out


def apply_windows_on_raw(
    src: Path,
    raw_items: List[Dict[str, Any]],
    windows: List[Tuple[datetime, datetime]],
) -> Tuple[List[Dict[str, Any]], List[int], str]:
    """Legacy marker: kept for compatibility. Not used in final recomposition."""
    if not windows:
        return list(raw_items), [0] * 0, "none"
    total = len(raw_items)
    hits = [0] * len(windows)
    corrected: List[Dict[str, Any]] = []
    inferred_source = None
    date_key = parse_date_from_filename(src)
    base_date = datetime.strptime(date_key, "%d_%m_%Y")
    for idx, item in enumerate(raw_items):
        it = infer_item_dt(src, item, idx, total, base_date=base_date)
        if inferred_source is None:
            for key in ("timestamp", "time", "ts", "datetime", "time_utc"):
                if key in item:
                    inferred_source = f"item_field:{key}"
                    break
            if inferred_source is None and _parse_gid_dt(base_date, item.get("group_id", "")) is not None:
                inferred_source = "group_id"
            if inferred_source is None and parse_start_end_from_filename(src):
                inferred_source = "filename_span_interpolation"
        if it is None:
            corrected.append(dict(item))
            continue
        match_idx = None
        for wi, (ws, we) in enumerate(windows):
            if ws <= it < we:
                match_idx = wi
                break
        new_item = dict(item)
        if match_idx is not None:
            hits[match_idx] += 1
            new_item["ai_category"] = "error"
        corrected.append(new_item)
    return corrected, hits, (inferred_source or "none")

def recompose_with_error_placeholders(
    src: Path,
    raw_items: List[Dict[str, Any]],
    windows: List[Tuple[datetime, datetime]],
) -> Tuple[List[Dict[str, Any]], List[int], str]:
    """
    Create a new list where:
      - For each no-detections window, we insert placeholders labeled ai_category='error'
        with count equal to the number of original items in that window (preserving order).
      - The original items from that window are moved forward to immediately after the window.
      - Items outside windows keep their relative order.
    Returns (corrected_items, window_hits_counts, inferred_time_source).
    """
    if not windows:
        return list(raw_items), [0] * 0, "none"
    # Normalize and sort windows
    windows_sorted = sorted(windows, key=lambda w: w[0])
    # Build enriched list with inferred times
    enriched: List[Tuple[int, Dict[str, Any], datetime | None]] = []
    total = len(raw_items)
    inferred_source = None
    date_key = parse_date_from_filename(src)
    base_date = datetime.strptime(date_key, "%d_%m_%Y")
    for idx, item in enumerate(raw_items):
        t = infer_item_dt(src, item, idx, total, base_date=base_date)
        if inferred_source is None:
            for key in ("timestamp", "time", "ts", "datetime", "time_utc"):
                if key in item:
                    inferred_source = f"item_field:{key}"
                    break
            if inferred_source is None and _parse_gid_dt(base_date, item.get("group_id", "")) is not None:
                inferred_source = "group_id"
            if inferred_source is None and parse_start_end_from_filename(src):
                inferred_source = "filename_span_interpolation"
        enriched.append((idx, item, t))
    # Split items into known-time and unknown-time
    known = [(i, it, t) for (i, it, t) in enriched if t is not None]
    unknown = [(i, it) for (i, it, t) in enriched if t is None]
    # Sort known by time
    known.sort(key=lambda x: x[2])
    # Estimate step (seconds) between consecutive known items
    def median_step_seconds() -> int:
        if len(known) < 2:
            return 1
        diffs = []
        for (a_i, _a_it, a_t), (b_i, _b_it, b_t) in zip(known[:-1], known[1:]):
            try:
                diffs.append(int((b_t - a_t).total_seconds()))
            except Exception:
                continue
        diffs = [d for d in diffs if d > 0]
        if not diffs:
            return 1
        diffs.sort()
        return diffs[len(diffs) // 2]
    step_sec = median_step_seconds()
    # Prepare output and counts
    out: List[Dict[str, Any]] = []
    hits: List[int] = [0] * len(windows_sorted)
    cur_pos = 0
    # Emit items and process each window sequentially
    for wi, (ws, we) in enumerate(windows_sorted):
        # 1) Emit items strictly before ws
        while cur_pos < len(known) and known[cur_pos][2] < ws:
            out.append(known[cur_pos][1])
            cur_pos += 1
        # 2) Collect items inside [ws, we)
        inside: List[Tuple[int, Dict[str, Any], datetime]] = []
        while cur_pos < len(known) and ws <= known[cur_pos][2] < we:
            inside.append(known[cur_pos])
            cur_pos += 1
        hits[wi] = len(inside)
        # 3) Emit placeholders in-window (error + no motion) at original slots
        for _, it, _t in inside:
            # Minimal placeholder with boolean motion
            out.append({
                "group_id": str(it.get("group_id", "")),
                "ai_category": "error",
                "motion": False,
            })
        # 4) Emit moved originals immediately after the window, spaced by step_sec
        from datetime import timedelta
        cur_time = we
        for _, it, _t in inside:
            # Minimal moved detection with updated time/group_id; preserve category; motion will be coerced later
            out.append({
                "group_id": f"{cur_time.hour:02d}_{cur_time.minute:02d}_{cur_time.second:02d}",
                "ai_category": it.get("ai_category"),
                "motion": it.get("motion", False),
            })
            cur_time = cur_time + timedelta(seconds=max(1, step_sec))
    # After processing all windows, emit remaining known-time items
    while cur_pos < len(known):
        it = known[cur_pos][1]
        out.append({
            "group_id": str(it.get("group_id", "")),
            "ai_category": it.get("ai_category"),
            "motion": it.get("motion", False),
        })
        cur_pos += 1
    # Append unknown-time items at end, preserving original order among them
    out.extend([it for (_i, it) in unknown])
    # Finally, sort output strictly by time using base_date + group_id
    def _key(o: Dict[str, Any]):
        dt = _parse_gid_dt(base_date, o.get("group_id", ""))
        return (dt or base_date)
    out_sorted = sorted(out, key=_key)
    return out_sorted, hits, (inferred_source or "none")

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
    ap.add_argument(
        "--no-detections",
        action="append",
        default=[],
        help="Time window HH:MM-HH:MM when camera is down; can be repeated",
    )
    ap.add_argument("--debug", action="store_true", help="Save filtered JSON/info locally and print detailed window info")
    args = ap.parse_args()

    def parse_hhmm_range(r: str) -> Tuple[int, int, int, int]:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", r.strip())
        if not m:
            raise ValueError(f"Invalid --no-detections format: {r} (expected HH:MM-HH:MM)")
        h1, m1, h2, m2 = map(int, m.groups())
        return h1, m1, h2, m2

    def process_one(src: Path) -> None:
        date_key = parse_date_from_filename(src)
        # Build skip windows for this file's date
        date_dt = datetime.strptime(date_key, "%d_%m_%Y")
        windows: List[Tuple[datetime, datetime]] = []
        for r in args.no_detections:
            h1, m1, h2, m2 = parse_hhmm_range(r)
            wstart = date_dt.replace(hour=h1, minute=m1, second=0, microsecond=0)
            wend = date_dt.replace(hour=h2, minute=m2, second=0, microsecond=0)
            if wend <= wstart:
                # If end <= start, assume window spans to end of day
                wend = date_dt.replace(hour=23, minute=59, second=59, microsecond=0)
            windows.append((wstart, wend))

        # Debug prints on windows
        if args.debug:
            print(f"[DEBUG] File: {src.name}")
            print(f"[DEBUG] Date key: {date_key} -> {date_dt.date().isoformat()}")
            if windows:
                print("[DEBUG] No-detections windows:")
                for (ws, we) in windows:
                    print(f"         - {ws.isoformat()} to {we.isoformat()}")
            else:
                print("[DEBUG] No-detections windows: none")
            span = parse_start_end_from_filename(src)
            if span:
                print(f"[DEBUG] Filename time span: {span[0].isoformat()} -> {span[1].isoformat()}")

        # Create corrected version of ORIGINAL data by skipping items in windows
        raw_data = json.loads(src.read_text())
        corrected_raw, window_hits, inferred_source = recompose_with_error_placeholders(src, raw_data, windows)
        # Sanitize for upload: only group_id, ai_category, motion (boolean coerced by threshold)
        thr = float(args.threshold)
        def to_bool_motion(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            try:
                return float(v) >= thr
            except Exception:
                return False
        upload_items: List[Dict[str, Any]] = []
        for itm in corrected_raw:
            ai = str(itm.get("ai_category"))
            if ai.lower() == "error":
                motion_bool = False
            else:
                motion_bool = to_bool_motion(itm.get("motion", False))
            upload_items.append({
                "group_id": str(itm.get("group_id", "")),
                "ai_category": ai,
                "motion": motion_bool,
            })
        bucket = "easytrack"
        object_key = f"verdis/belt/{date_key}.json"
        object_key_info = f"verdis/belt/{date_key}_info.json"
        info_payload = compute_info(upload_items)

        if args.dry:
            print(json.dumps({
                "file": str(src),
                "bucket": bucket,
                "key": object_key,
                "items_count": len(upload_items),
                "info_key": object_key_info,
                "info": info_payload,
                "raw_original_items": len(raw_data),
                "raw_corrected_items": len(corrected_raw),
                "skipped_in_windows": sum(window_hits),
            }, ensure_ascii=False, indent=2))
            # In debug, also persist local copies
            if args.debug:
                corrected_path = src.with_suffix(".corrected.json")
                filtered_info_path = src.with_suffix(".corrected_info.json")
                debug_meta_path = src.with_suffix(".corrected_debug.json")
                corrected_path.write_text(json.dumps(upload_items, ensure_ascii=False, indent=2))
                filtered_info_path.write_text(json.dumps(info_payload, ensure_ascii=False, indent=2))
                debug_payload = {
                    "windows": [{"start": ws.isoformat(), "end": we.isoformat()} for (ws, we) in windows],
                    "stats": {
                        "raw_original_items": len(raw_data),
                        "raw_corrected_items": len(upload_items),
                        "skipped_in_windows": sum(window_hits),
                        "window_hits": [
                            {"window": {"start": ws.isoformat(), "end": we.isoformat()}, "count": window_hits[i]}
                            for i, (ws, we) in enumerate(windows)
                        ],
                        "inferred_time_source": inferred_source,
                    },
                }
                debug_meta_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2))
            return

        # Non-dry path: upload corrected ORIGINAL json instead of raw; info from corrected
        if args.debug:
            corrected_path = src.with_suffix(".corrected.json")
            filtered_info_path = src.with_suffix(".corrected_info.json")
            debug_meta_path = src.with_suffix(".corrected_debug.json")
            corrected_path.write_text(json.dumps(upload_items, ensure_ascii=False, indent=2))
            filtered_info_path.write_text(json.dumps(info_payload, ensure_ascii=False, indent=2))
            debug_payload = {
                "windows": [{"start": ws.isoformat(), "end": we.isoformat()} for (ws, we) in windows],
                "stats": {
                    "raw_original_items": len(raw_data),
                    "raw_corrected_items": len(upload_items),
                    "skipped_in_windows": sum(window_hits),
                    "window_hits": [
                        {"window": {"start": ws.isoformat(), "end": we.isoformat()}, "count": window_hits[i]}
                        for i, (ws, we) in enumerate(windows)
                    ],
                    "inferred_time_source": inferred_source,
                },
            }
            debug_meta_path.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2))
            # Also print both JSON payloads for visibility
            print("[DEBUG] Corrected detections JSON (uploaded):")
            print(json.dumps(upload_items, ensure_ascii=False, indent=2))
            print("[DEBUG] Corrected info JSON (uploaded):")
            print(json.dumps(info_payload, ensure_ascii=False, indent=2))

        # Upload corrected original items as main object
        upload_json(bucket, object_key, upload_items)
        upload_json(bucket, object_key_info, info_payload)
        print(f"Uploaded {len(upload_items)} items to {bucket}/{object_key} from {src.name}")
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


