"""
Truck Screen Watcher – Debug Edition
===================================
Continuously captures the primary screen, feeds the image to SmolVLM and asks:
    "Is there a truck in this image?"
If SmolVLM answers **yes**, an audible beep is played.  With `DEBUG = True` the
program also opens a live window that shows the captured frame and overlays the
current answer ("TRUCK: YES" / "TRUCK: NO").  **No external sound file is
required** – a default sine‑wave beep is generated in‑memory. Drop an
`alert.wav` next to the script if you prefer a custom sound.

Requirements
------------
::
    pip install torch torchvision transformers pillow mss simpleaudio opencv-python numpy

Usage
-----
::
    python truck_screen_watcher.py          # run with default DEBUG = False
    python truck_screen_watcher.py --debug  # enable live view
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
import cv2  # type: ignore
import mss
import simpleaudio as sa

# ----------------- Configuration -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
ALERT_WAV_PATH = Path(__file__).with_name("alert.wav")  # optional custom sound
SLEEP_SECONDS = 2  # pause between iterations (after inference)
DEBUG = False  # can be toggled at CLI
# -------------------------------------------------

# Pre‑load model and processor once
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE).eval()

# ----------- Sound handling ---------------------------------------------------
wave_obj: sa.WaveObject | None
beep_wave: sa.WaveObject

try:
    wave_obj = sa.WaveObject.from_wave_file(str(ALERT_WAV_PATH))
    print(f"[INFO] Using custom alert sound: {ALERT_WAV_PATH.name}")
except FileNotFoundError:
    wave_obj = None
    print("[INFO] No 'alert.wav' found – using default beep.")

# Generate an in‑memory sine‑wave beep (16‑bit, mono)
sample_rate = 44_100
beep_duration = 0.3  # seconds
freq = 880  # Hz (A5)

t = np.linspace(0, beep_duration, int(sample_rate * beep_duration), False)
waveform = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
beep_wave = sa.WaveObject(waveform.tobytes(), 1, 2, sample_rate)

# ------------------------------------------------------------------------------

def capture_screen() -> Image.Image:
    """Grab the entire primary monitor and return a PIL.Image."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 0 is all monitors; 1 is primary
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def has_truck(img: Image.Image) -> bool:
    """Ask SmolVLM whether a truck is present in *img*."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Is there a truck in this image? Answer yes or no.",
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
    ids = model.generate(**inputs, max_new_tokens=10)
    answer = processor.batch_decode(ids, skip_special_tokens=True)[0].lower()
    return "yes" in answer and "no" not in answer


def play_alert():
    """Play custom wave if present, otherwise default beep."""
    if wave_obj is not None:
        wave_obj.play()
    else:
        beep_wave.play()


def show_debug(frame: np.ndarray, truck_found: bool):
    """Display *frame* in an OpenCV window with overlay text."""
    text = f"TRUCK: {'YES' if truck_found else 'NO'}"
    color = (0, 255, 0) if truck_found else (0, 0, 255)
    cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.imshow("Truck Screen Watcher – Debug", frame)
    # 1 ms wait to process GUI events; exits if q pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise KeyboardInterrupt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truck Screen Watcher")
    parser.add_argument("--debug", action="store_true", help="Enable live debug window")
    return parser.parse_args()


def main():
    global DEBUG  # noqa: PLW0603
    args = parse_args()
    DEBUG = args.debug or DEBUG

    print("[INFO] Debug mode ON" if DEBUG else "[INFO] Watching for trucks…")

    try:
        while True:
            print("run loop")
            pil_img = capture_screen()
            truck_found = has_truck(pil_img)

            if truck_found:
                print("[DETECTED] Truck present – playing alert.")
                play_alert()

            if DEBUG:
                print("show debug")
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                show_debug(frame, truck_found)

            time.sleep(SLEEP_SECONDS)
    except KeyboardInterrupt:
        if DEBUG:
            cv2.destroyAllWindows()
        print("\n[INFO] Exiting – goodbye!")


if __name__ == "__main__":
    main()