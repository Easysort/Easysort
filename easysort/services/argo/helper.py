
from typing import Optional, List, Union
import datetime
import tempfile
from pathlib import Path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from supabase import create_client, Client
from easysort.common.environment import Env
from openai import OpenAI
import mimetypes
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, UnidentifiedImageError
import io
import yaml


@dataclass
class Locations:
    @dataclass 
    class SD128:
        MAC = Path("/Volumes/Easysort128")
        WINDOWS = Path("E:/Easysort128")
        LINUX = Path("/mnt/sdcard")
    STATS = Path("/easysort/services/argo/stats"),


@dataclass
class SupabaseLocations:
    @dataclass
    class Argo: 
        Roskilde01 = "ARGO-Roskilde-Entrance-01"
        bucket = "argo"
        @classmethod
        def ids(cls) -> dict: return {"Roskilde01": cls.Roskilde01}


class Downloader:
    date: datetime.datetime
    tmp_dir: Path
    files_per_hour: Optional[dict[int, List[str]]] = None

    def __init__(self, device_id: str, bucket: str, location: Path, date: Optional[Union[datetime.date, datetime.datetime]] = None, 
                tmp_dir: Optional[Path] = None) -> None:
        self.date_time = date if date is not None else datetime.datetime.now().date()
        self.bucket = bucket
        self.location = location
        self.device_id = device_id
        assert self.location is not None
        if not os.path.exists(self.location): raise FileExistsError(f"{self.location} not found. Please check the path again.")
        self.tmp_dir = tmp_dir if tmp_dir is not None else Path(tempfile.mkdtemp(dir=self.location))
        assert os.path.exists(self.tmp_dir)
        assert self.date_time is not None and isinstance(self.date_time, (datetime.date))
        self.supabase_client: Client = create_client(Env.SUPABASE_URL, Env.SUPABASE_KEY)
        # self.openai_client: OpenAI = OpenAI(base_url="https://openrouter.ai/api/v1", api_key = Env.OPENROUTER_API_KEY)
        self.openai_client: OpenAI = OpenAI(api_key = Env.OPENAI_API_KEY)


    def list_hour_files(self, force_reload: bool = False) -> None:
        if self.files_per_hour is not None and not force_reload: return
        assert self.date_time is not None
        self.files_per_hour = {}
        for h in range(24):
            self.files_per_hour[h] = []
            limit, offset = 1000, 0
            hh = f"{h:02d}"
            prefix: str = f"{self.device_id}/{self.date_time.year}/{self.date_time.month}/{self.date_time.day}/{hh}/"
            while True:
                entries = self.supabase_client.storage.from_(self.bucket).list(prefix, {"limit": limit, "offset": offset, "sortBy": {"column": "name", "order": "asc"}})
                if not entries: break
                self.files_per_hour[h].extend([prefix + (e.get("name") or "") for e in entries if e.get("name") is not None])
                if len(entries) < limit: break
                offset += len(entries)

    def download_all_hours(self, max_workers: int = 1) -> None:
        self.list_hour_files()
        assert self.files_per_hour is not None
        for hour in tqdm(self.files_per_hour.keys(), desc="Downloading all hours"):
            self.download_hour_files(hour, max_workers)

    def download_hour_files(self, hour: int, max_workers: int = 1) -> None:
        self.list_hour_files()
        assert self.files_per_hour is not None
        if not os.path.exists(self.tmp_dir / f"hour_{hour}"): os.makedirs(self.tmp_dir / f"hour_{hour}")
        keys = [k for k in tqdm(self.files_per_hour.get(hour, []), desc=f"Filtering hour {hour} files") if not os.path.exists(self.tmp_dir / f"hour_{hour}" / Path(k).name)]
        print("Missing files: ", len(keys), "out of", len(self.files_per_hour.get(hour, [])))
        
        def _download(key: str): 
            with open(self.tmp_dir / f"hour_{hour}" / Path(key).name, "wb") as f: f.write(self.supabase_client.storage.from_(self.bucket).download(key))

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

    def analyze_hour_files(self, hour: int, max_workers: int = 30) -> None:
        hour_dir = self.tmp_dir / f"hour_{hour}"
        files = [f for f in os.listdir(hour_dir) 
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.lower().startswith(".") and not os.path.exists(hour_dir / Path(f.rsplit(".", 1)[0] + ".json"))]

        def work(fname: str):
            p = hour_dir / fname
            data = self.analyze_image(str(p))
            with open(str(p).rsplit(".", 1)[0] + ".json", "w") as f:
                json.dump(data, f)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for _ in tqdm(ex.map(work, files), total=len(files), desc=f"Analyzing hour {hour} files"):
                pass

    def encode_image_b64_optimized(self, path: str, max_px: int = 1024, quality: int = 85) -> tuple[str, str]:
        media_type, _ = mimetypes.guess_type(path)
        if not media_type:
            media_type = "image/jpeg"
        if Image is None:
            with open(path, "rb") as f:
                raw = f.read()
            return media_type, base64.b64encode(raw).decode("utf-8")
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max(w, h) / max_px
            im_for_encoding = im
            if scale > 1:
                im_for_encoding = im.resize((int(w/scale), int(h/scale)))
            buf = io.BytesIO()
            im_for_encoding.save(buf, format="JPEG", quality=quality, optimize=True)
            return "image/jpeg", base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze_image(self, image_path: str) -> dict:
        media_type, image_b64 = self.encode_image_b64_optimized(image_path)
        response = self.openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Return only a YAML object with the following keys: "
                        "'number_of_people' (int), 'description_of_people' (list[str]), and "
                        "'list_of_items_people_are_carrying' (list[str]). "
                        "Make sure the people are actually carrying the items they are described as carrying, "
                        "not just standing next to. Use only valid YAML, no commentary."
                    )},  
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                ],
            }],
            response_format={"type": "text"},  # No forced JSON, want raw YAML
            timeout=30,
        )
        content = response.choices[0].message.content
        yaml_str = content
        if "---" in yaml_str: yaml_str = yaml_str[yaml_str.find("---") + 3:]
        data = yaml.safe_load(yaml_str)
        return json.dumps(data)

    def get_hour_information(self, hour: int) -> dict:
        hour_dir = self.tmp_dir / f"hour_{hour}"
        json_files = [f for f in os.listdir(hour_dir) if f.lower().endswith(".json") and not f.lower().startswith(".")]
        total_people = 0
        description_of_people = []
        list_of_items_people_are_carrying = []
        for json_file in json_files:
            with open(hour_dir / json_file, "r") as f:
                data = json.loads(json.load(f).strip())
            total_people += int(data["number_of_people"])
            description_of_people.extend(data["description_of_people"])
            list_of_items_people_are_carrying.extend(data["list_of_items_people_are_carrying"])
        with open(hour_dir / "hour_information.json", "w") as f:
            json.dump({"total_people": total_people, "description_of_people": description_of_people, "list_of_items_people_are_carrying": list_of_items_people_are_carrying}, f)

    def full_analyze_hours(self, hours: list[int]) -> None:
        for hour in tqdm(hours, desc="Analyzing hours"):
            self.analyze_hour_files(hour)
            self.get_hour_information(hour)

    def is_image_valid(self, path: str, min_bytes: int = 1024) -> bool:
        try:
            if not os.path.exists(path): return False
            if os.path.getsize(path) < min_bytes: return False
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    def clean_broken_images(self, hour: int, min_bytes: int = 1024) -> int:
        hour_dir = self.tmp_dir / f"hour_{hour}"
        if not os.path.exists(hour_dir): return 0
        deleted = 0
        for f in os.listdir(hour_dir):
            if f.lower().startswith("."): continue
            if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
            p = hour_dir / f
            if not self.is_image_valid(str(p), min_bytes=min_bytes):
                try:
                    os.remove(p)
                    # also remove any stale sidecar json for this image
                    sidecar = hour_dir / (Path(f).stem + ".json")
                    if os.path.exists(sidecar): os.remove(sidecar)
                    deleted += 1
                except Exception:
                    pass
        return deleted

    def cleanup(self) -> None:
        for hour in tqdm(range(24), desc="Cleaning up"):
            self.clean_broken_images(hour)



if __name__ == "__main__":
    date = datetime.datetime(2025, 10, 26)
    tmp_dir = Locations.SD128.LINUX / "tmpddtx4_r6"
    location = Locations.SD128.LINUX
    downloader = Downloader(device_id=SupabaseLocations.Argo.Roskilde01, bucket=SupabaseLocations.Argo.bucket, 
                    location=location, tmp_dir=Path(tmp_dir), date=date)
    downloader.cleanup()
    downloader.download_all_hours()
    downloader.full_analyze_hours([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
