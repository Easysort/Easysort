import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import cv2


def ffprobe_info(video_path: Path) -> Optional[Dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe,
        "-hide_banner",
        "-loglevel", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd)
        return json.loads(out.decode("utf-8", errors="ignore"))
    except Exception:
        return None


def opencv_info(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()
    codec_tag = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    return {
        "backend": "opencv",
        "width": width,
        "height": height,
        "fps": fps,
        "nb_frames": frame_count,
        "duration": duration,
        "fourcc": codec_tag,
    }


def print_summary(meta: Dict[str, Any]) -> None:
    fmt = meta.get("format", {})
    streams = meta.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]

    print("Video info")
    if fmt:
        print(f"- container: {fmt.get('format_long_name','?')} ({fmt.get('format_name','?')})")
        print(f"- duration:  {fmt.get('duration','?')} s  size: {fmt.get('size','?')} bytes  bitrate: {fmt.get('bit_rate','?')} bps")
    for i, s in enumerate(vstreams):
        print(f"- video[{i}]: codec={s.get('codec_name','?')} {s.get('codec_long_name','')}  pix_fmt={s.get('pix_fmt','?')}  profile={s.get('profile','?')}")
        print(f"            resolution={s.get('width','?')}x{s.get('height','?')}  fps={s.get('avg_frame_rate','?')}  tbr={s.get('r_frame_rate','?')}")
        if s.get('bit_rate'): print(f"            bitrate={s.get('bit_rate')} bps")
        if s.get('nb_frames'): print(f"            frames={s.get('nb_frames')}")
        color_space = s.get('color_space') or s.get('color_transfer') or s.get('color_primaries')
        if color_space:
            print(f"            color={s.get('color_space','?')}/{s.get('color_transfer','?')}/{s.get('color_primaries','?')}")
    for i, s in enumerate(astreams):
        print(f"- audio[{i}]: codec={s.get('codec_name','?')} ch={s.get('channels','?')} sr={s.get('sample_rate','?')} Hz  bitrate={s.get('bit_rate','?')} bps")


def main() -> None:
    ap = argparse.ArgumentParser(description="Print video metadata (ffprobe if available, else OpenCV)")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--json", action="store_true", help="Print raw ffprobe JSON if available")
    args = ap.parse_args()

    path = Path(args.video)
    meta = ffprobe_info(path)
    if meta is not None:
        if args.json:
            print(json.dumps(meta, indent=2))
            return
        print_summary(meta)
        return

    # Fallback
    info = opencv_info(path)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()


