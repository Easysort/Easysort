
# Improvements once storage / compute becomes an issue:
# - Use minikeyvalue with .host() option to pull data to multiple nodes
# - Store files in a compressed format
# - Remove supabase dependency
# - Data registry and compute on same machine. Once that is no longer true, GET should return bytes, not path.

from easysort.helpers import REGISTRY_PATH, T, SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET
from supabase import create_client, Client

import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Callable, Any, Dict, List
import json
from tqdm.contrib.concurrent import thread_map
from dataclasses import make_dataclass, dataclass, is_dataclass, asdict
import pandas as pd
import concurrent.futures
import time
from datetime import datetime

DEFAULT_DETECTION_DATACLASS = make_dataclass("Detection", [("x1", float), ("y1", float), ("x2", float), ("y2", float), ("conf", float), ("cls", str)])
DEFAULT_WASTE_DATACLASS = make_dataclass("WasteDetection", [("fraction", str), ("sub_fraction", str), ("purity", float), ("weight_kg", float), ("co2_kg", float)])


class RegistryBase:
    """
    For POST and GET use one of the default types or your own dataclass. The name of the dataclass should be the same.
    """
    class DefaultTypes: # Or use your own dataclass
        RESULT_PEOPLE = make_dataclass("RESULT_PEOPLE", [("frame_results", Dict[int, List[DEFAULT_DETECTION_DATACLASS]])])
        RESULT_WASTE = make_dataclass("RESULT_WASTE", [("frame_results", Dict[int, List[DEFAULT_WASTE_DATACLASS]])])

    def __init__(self, registry_path: Path): 
        self.registry_path = registry_path
        # os.makedirs(self.registry_path, exist_ok=True)
        #self.projects = open(os.path.join(self.registry_path, "projects.txt")).read().splitlines() if os.path.exists(os.path.join(self.registry_path, "projects.txt")) else []

    def _check_path(self, key: Path, _type: T) -> Path:
        assert isinstance(key, Path), f"Key {key} is not a Path, but a {type(key)}"
        assert is_dataclass(_type) or _type in self.DefaultTypes, f"Type {_type} is not a dataclass, but a {type(_type)}"
        return Path(self.registry_path / key.with_suffix("") / str(_type.__name__).lower()).with_suffix(".parquet")

    def GET(self, key: Path, _type: T, throw_error: bool = True) -> Optional[T]:
        path = self._check_path(key, _type)
        if not path.is_file() and throw_error: raise FileNotFoundError(f"File {path} not found")
        if not path.is_file() and not throw_error: return None
        return _type(**pd.read_parquet(path).iloc[0].to_dict())

    def POST(self, key: Path, data: T, _type: T, overwrite: bool = False) -> None:
        path = self._check_path(key, _type)
        assert isinstance(data, _type), f"Data {data} is not a {_type}"
        assert not overwrite or not path.is_file(), f"File {key} already exists"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(data)]).to_parquet(path)
        with open(path.with_suffix(".schema.json"), "w") as f: json.dump(asdict(_type), f)

    def LIST(self, prefix: Optional[str] = "", suffix: Optional[List[str] | str] = [".mp4", ".jpeg", ".jpg", ".png"], cond: Optional[Callable[[str], bool]] = None, return_all: bool = False) -> list[Path]:
        files = [Path(os.path.join(root, file)) for root, _, files in os.walk(Path(self.registry_path) / prefix) for file in files]
        files = [file for file in files if (file.suffix in list(suffix) if suffix else True) and not file.startswith("._")]
        cond_files = [file for file in files if cond(file)] if cond else files
        return [files, cond_files] if return_all else cond_files

    def SYNC(self) -> None:
        CONCURRENT_WORKERS = 3 # Depends on connection speed, supabase rate limit, so on. Currently 3 works fairly stable and fast
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        bucket = supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET)
        dirs, files, pbar = [Path(x["name"]) for x in bucket.list()], [], tqdm()
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            while len(dirs) > 0:
                future_to_dir = {executor.submit(bucket.list, str(d)): d for d in dirs}
                for future in concurrent.futures.as_completed(future_to_dir):
                    cur = future_to_dir[future]
                    dirs.remove(cur)
                    paths = [x["name"] for x in future.result()]
                    dirs.extend([cur / Path(x) for x in paths if not "." in x])
                    files.extend([cur / Path(x) for x in paths if "." in x])
                    pbar.update(1)
            pbar.close()
        
        missing_files = [file for file in files if not os.path.exists(os.path.join(self.registry_path, SUPABASE_DATA_REGISTRY_BUCKET, file)) if ".jpg" not in str(file)]
        print("Missing files: ", len(missing_files), "out of", len(files))

        def _download_one(file: str):
            dst = Path(self.registry_path) / SUPABASE_DATA_REGISTRY_BUCKET / Path(file)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(supabase_client.storage.from_("argo").download(str(file)))

        thread_map(_download_one, missing_files, desc="Downloading missing files", max_workers=CONCURRENT_WORKERS)
        print("Sync complete")

        print("Cleanup videos older than 2 weeks")
        files_to_delete = []
        for file in tqdm(files, desc="Cleanup videos"):
            file = str(file)
            year, month, day, hour, minute, second = file.split("/")[-5], file.split("/")[-4], file.split("/")[-3], file.split("/")[-1][:2], file.split("/")[-1][2:4], file.split("/")[-1][4:6]
            timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            if timestamp < datetime.datetime.now() - datetime.timedelta(weeks=3): files_to_delete.append(file)
        print(f"Deleting {len(files_to_delete)} files" if len(files_to_delete) > 0 else "No files to delete")
        if len(files_to_delete) == 0: return
        for i in tqdm(range(0, len(files_to_delete), 100), desc="Deleting files"):
            supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET).remove(files_to_delete[i:i+100])
            time.sleep(1)


    def EXISTS(self, key: Path, _type: T) -> bool: return self._check_path(key, _type).is_file()


Registry = RegistryBase(REGISTRY_PATH)

if __name__ == "__main__":
    Registry.SYNC(allow_special_cleanup=True)

