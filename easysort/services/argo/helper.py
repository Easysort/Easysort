
from typing import Optional, List
import datetime
import tempfile
from pathlib import Path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from supabase import create_client, Client
from easysort.common.environment import Environment
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from easysort.common.timer import TimeIt

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
ATTN_IMPL = "flash_attention_2" if DEVICE == "cuda" else "sdpa"

@dataclass
class Locations:
    EASYSORT128 = Path("/Volumes/EASYSORT128")
    STATS = Path("/easysort/services/argo/stats")


@dataclass
class SupabaseLocations:
    @dataclass
    class Argo: 
        Roskilde01 = "ARGO-Roskilde-Entrance-01"
        bucket = "argo"
        @classmethod
        def ids(cls) -> dict: return {"Roskilde01": cls.Roskilde01}


class Downloader:
    date: datetime
    tmp_dir: Path
    files_per_hour: Optional[dict[int, List[str]]] = None
    model: Optional[AutoModelForImageTextToText] = None
    processor: Optional[AutoProcessor] = None

    def __init__(self, device_id: str, bucket: str, date: Optional[datetime] = None, location: Optional[Path] = Locations.EASYSORT128) -> None:
        self.date_time = date
        self.bucket = bucket
        self.location = location
        self.device_id = device_id
        if not os.path.exists(self.location): raise FileExistsError(f"{self.location} not found. Please check the path again.")
        self.tmp_dir = Path(tempfile.mkdtemp(dir=self.location))
        if self.date_time is None: self.date_time = datetime.datetime.now().date()
        assert self.date_time is not None and isinstance(self.date_time, (datetime.date))
        self.client: Client = create_client(Environment.SUPABASE_URL, Environment.SUPABASE_KEY)

    @TimeIt("Initializing model and one it")
    def _init_model(self) -> None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")
        print("images loaded")

        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        print("processor loaded")
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32,
            _attn_implementation=ATTN_IMPL,
        ).to(DEVICE)
        print("model loaded")
        self.processor = processor
        self.model = model

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe the two images?"}
                ]
            },
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
        inputs = inputs.to(DEVICE)
        print("inputs loaded")
        # Generate outputs
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            print("generated texts loaded")
            print(generated_texts[0])
        

    def list_hour_files(self, force_reload: bool = False) -> None:
        if self.files_per_hour is not None and not force_reload: return self.files_per_hour
        self.files_per_hour = {}
        for h in range(24):
            self.files_per_hour[h] = []
            limit, offset = 1000, 0
            hh = f"{h:02d}"
            prefix = f"{self.device_id}/{self.date_time.year}/{self.date_time.month}/{self.date_time.day}/{hh}/"
            while True:
                entries = self.client.storage.from_(self.bucket).list(prefix,
                    {"limit": limit, "offset": offset, "sortBy": {"column": "name", "order": "asc"}})
                if not entries: break
                self.files_per_hour[h].extend([prefix + e.get("name") for e in entries if e.get("name")])
                if len(entries) < limit: break
                offset += len(entries)

    def download_hour_files(self, hour: int, max_workers: int = 16) -> None:
        self.list_hour_files()
        keys = self.files_per_hour.get(hour, [])

        def _download(key: str): 
            with open(self.tmp_dir / Path(key).name, "wb") as f: f.write(self.client.storage.from_(self.bucket).download(key))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_download, k): k for k in keys}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading hour {hour} files"):
                fut.result()


    def create_detection_groups(self) -> list[list[str]]:
        # Not tested
        timestamps = {file.split("_")[1].split("T")[1]: file for file in os.listdir(self.tmp_dir)}
        seconds = {int(k[0:2])*3600+int(k[2:4])*60+int(k[4:6]): v for k, v in timestamps.items()}
        seconds_sorted = sorted(seconds.keys())
        groups: List[List[str]] = []
        for i in tqdm(range(len(seconds_sorted)), desc="Creating detection groups"):
            if i == 0: groups.append([seconds[seconds_sorted[i]]])
            else:
                if abs(seconds_sorted[i] - seconds_sorted[i-1]) < 3: groups[-1].append(seconds[seconds_sorted[i]])
                else: groups.append([seconds[seconds_sorted[i]]])
        return groups
            

    def detect_humans(self) -> None:
        for file in os.listdir(self.tmp_dir):
            image = Image.open(self.tmp_dir / file)
            prompt = "Are there any humans in this image, where more than 50% of their body is shows?"
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(DEVICE)
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

if __name__ == "__main__":
    downloader = Downloader(device_id=SupabaseLocations.Argo.Roskilde01, bucket=SupabaseLocations.Argo.bucket)
    downloader._init_model()
