
# Improvements once storage / compute becomes an issue:
# - Use minikeyvalue with .host() option to pull data to multiple nodes
# - Store files in a compressed format
# - Remove supabase dependency
# - Data registry and compute on same machine. Once that is no longer true, GET should return bytes, not path.

from easysort.helpers import DATA_REGISTRY_PATH, SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET, RESULTS_REGISTRY_PATH, REGISTRY_PATH
from supabase import create_client, Client
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Callable, Any
import json
import numpy as np
import datetime
from tqdm.contrib.concurrent import thread_map
import time


class RegistryBase:
    def __init__(self, registry_path: str): 
        self.registry_path = registry_path
        os.makedirs(self.registry_path, exist_ok=True)
        self.projects = open(os.path.join(self.registry_path, "projects.txt")).read().splitlines() if os.path.exists(os.path.join(self.registry_path, "projects.txt")) else []

    def SYNC(self, allow_special_cleanup: bool = False) -> None: # with Supabase
        # Find all files in a bucket, download to registry
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        dirs, files, pbar = [Path(x["name"]) for x in supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET).list()], [], tqdm()
        while len(dirs) > 0:
            cur = dirs.pop(0)
            paths = [x["name"] for x in supabase_client.storage.from_("argo").list(str(cur))]
            dirs.extend([cur / Path(x) for x in paths if not "." in x])
            files.extend([cur / Path(x) for x in paths if "." in x])
            pbar.update(1)
        pbar.close()

        missing_files = [file for file in files if not os.path.exists(os.path.join(self.registry_path, SUPABASE_DATA_REGISTRY_BUCKET, file)) if ".jpg" not in str(file)]
        print("Missing files: ", len(missing_files), "out of", len(files))
        with open("missing_files.txt", "w") as f:
            for file in missing_files:
                f.write(str(file) + "\n")

        def _download_one(file: str):
            dst = Path(self.registry_path) / SUPABASE_DATA_REGISTRY_BUCKET / Path(file)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(supabase_client.storage.from_("argo").download(str(file)))

        thread_map(_download_one, missing_files, desc="Downloading missing files", max_workers=max(os.cpu_count(), 4)) # else rate limited

        print("Checking health: ")
        assert self.is_healthy(), "Registry is not healthy"
        print("Sync complete")

        print("Cleanup videos older than 2 weeks")
        files_to_delete = []
        for file in tqdm(files, desc="Cleanup videos"):
            file = str(file)
            year, month, day, hour, minute, second = file.split("/")[-5], file.split("/")[-4], file.split("/")[-3], file.split("/")[-1][:2], file.split("/")[-1][2:4], file.split("/")[-1][4:6]
            timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            if timestamp < datetime.datetime.now() - datetime.timedelta(weeks=2): files_to_delete.append(file)
        print(f"Deleting {len(files_to_delete)} files" if len(files_to_delete) > 0 else "No files to delete")
        if len(files_to_delete) == 0: return
        for i in tqdm(range(0, len(files_to_delete), 100), desc="Deleting files"):
            supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET).remove(files_to_delete[i:i+100])
            time.sleep(1)

    def GET(self, key: str, loader: Optional[Callable[[bytes], Any]] = None) -> bytes: # TODO
        if loader is None: loader = {"json": json.load, "npy": np.load, "bytes": lambda x: x}[Path(key).suffix.lstrip(".")]
        assert self.EXISTS(key), f"File {key} not found"
        return loader(open(Path(self.registry_path, key), "rb").read())

    def LIST(self, prefix: Optional[str] = "", suffix: Optional[str] = ".mp4") -> list[str]:
        files = [os.path.join(root, file) for root, dirs, files in os.walk(Path(self.registry_path) / prefix) for file in files]
        files = [file for file in files if file.endswith(suffix) and not file.startswith("._")]
        return [self._unregistry_path(file) for file in files]
    
    def add_project(self, model: str, project: str) -> None: 
        if os.path.join(model, project) in self.projects: return
        self.projects = sorted(list(set(self.projects + [os.path.join(model, project)])))
        open(os.path.join(self.registry_path, "projects.txt"), "w").write("\n".join(self.projects))

    def construct_path(self, path: str, model: str, project: str, identifier: str) -> str: 
        self.add_project(model, project)
        return os.path.join(str(Path(path).with_suffix("")), model, project, identifier)

    def _registry_path(self, path: str) -> str: return os.path.join(self.registry_path, path)
    def _unregistry_path(self, path: str|Path) -> str: return str(path).replace(self.registry_path, "").lstrip("/")

    def POST(self, key: str, data: dict|bytes|np.ndarray|list) -> None: 
        os.makedirs(self._registry_path(str(Path(key).with_suffix(""))), exist_ok=True)
        {dict: self._post_json, list: self._post_json, bytes: self._post_bytes, np.ndarray: self._post_numpy}[type(data)](Path(key).with_suffix(""), data)

    def _post_json(self, key: str, data: dict) -> None: json.dump(data, open(Path(os.path.join(self.registry_path, key)).with_suffix(".json"), "w"))
    def _post_bytes(self, key: str, data: bytes) -> None: open(Path(os.path.join(self.registry_path, key)).with_suffix(".bytes"), "wb").write(data)
    def _post_numpy(self, key: str, data: np.ndarray) -> None: np.save(Path(os.path.join(self.registry_path, key)).with_suffix(".npy"), data)

    def EXISTS(self, key: str) -> bool: return Path(self._registry_path(key)).is_file() or Path(self._registry_path(key)).is_dir()

Registry = RegistryBase(REGISTRY_PATH)

if __name__ == "__main__":
    Registry.SYNC(allow_special_cleanup=True)