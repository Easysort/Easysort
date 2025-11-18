
# Improvements once storage / compute becomes an issue:
# - Use minikeyvalue with .host() option to pull data to multiple nodes
# - Store files in a compressed format
# - Remove supabase dependency
# - Data registry and compute on same machine. Once that is no longer true, GET should return bytes, not path.

from easysort.helpers import DATA_REGISTRY_PATH, SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET, \
    RESULTS_REGISTRY_PATH
from supabase import create_client, Client
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import json
import numpy as np
import datetime
from easysort.helpers import Sort

class Registry:
    def __init__(self, registry_path: str): 
        self.registry_path = registry_path
        os.makedirs(self.registry_path, exist_ok=True)
        self.projects = open(os.path.join(self.registry_path, "projects.txt")).read().splitlines() if os.path.exists(os.path.join(self.registry_path, "projects.txt")) else []

    def SYNC(self) -> None: # with Supabase
        # Find all files in a bucket, download to DATA_REGISTRY_PATH
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        dirs, files, pbar = [Path(x["name"]) for x in supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET).list()], [], tqdm()
        while len(dirs) > 0:
            cur = dirs.pop(0)
            paths = [x["name"] for x in supabase_client.storage.from_("argo").list(str(cur))]
            dirs.extend([cur / Path(x) for x in paths if not "." in x])
            files.extend([cur / Path(x) for x in paths if "." in x])
            pbar.update(1)
        pbar.close()

        missing_files = [file for file in files if not os.path.exists(os.path.join(self.registry_path, SUPABASE_DATA_REGISTRY_BUCKET, file))]
        print("Missing files: ", len(missing_files), "out of", len(files))
        for file in tqdm(missing_files, desc="Downloading missing files"): 
            dst: Path = Path(self.registry_path) / Path(SUPABASE_DATA_REGISTRY_BUCKET) / file
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(supabase_client.storage.from_("argo").download(str(file)))
        print("Checking health: ")
        assert self.is_healthy(), "Registry is not healthy"
        print("Sync complete")

    def is_healthy(self, verbose: bool = True) -> bool:
        devices = [dir for dir in os.listdir(os.path.join(self.registry_path, "argo")) if os.path.isdir(os.path.join(self.registry_path, "argo", dir))]
        all_entries = self.LIST("argo")
        for entry in all_entries: pass
            # year, month, day, hour
        is_healthy = True
        for device in devices: pass

        # print last entry for each device, check last entry no longer that 1 hour away in the time between 22 and 6
        return True

    def GET(self, key: str, ) -> bytes: # TODO
        return
        assert os.path.exists(os.path.join(self.registry_path, key)), f"File {key} not found"
        return os.path.join(self.registry_path, key)

    def LIST(self, prefix: Optional[str] = "", suffix: Optional[str] = "mp4") -> list[str]:
        return [self._unregistry_path(x) for x in (Path(self.registry_path) / prefix).glob(f"**/*.{suffix}") if x.is_file() and not x.name.startswith("._")]

    def add_project(self, model: str, project: str) -> None: 
        if os.path.join(model, project) in self.projects: return
        self.projects = sorted(list(set(self.projects + [os.path.join(model, project)])))
        open(os.path.join(self.registry_path, "projects.txt"), "w").write("\n".join(self.projects))

    def construct_path(self, path: str, model: str, project: str, identifier: str) -> str: 
        self.add_project(model, project)
        return os.path.join(self.registry_path, str(Path(path).with_suffix("")), model, project, identifier)

    def _registry_path(self, path: str) -> str: return os.path.join(self.registry_path, path)
    def _unregistry_path(self, path: str|Path) -> str: return str(path).replace(self.registry_path, "").lstrip("/")

    def POST(self, key: str, data: dict|bytes|np.ndarray|list) -> None: 
        os.makedirs(self._registry_path(str(Path(key).with_suffix(""))), exist_ok=True)
        {dict: self._post_json, list: self._post_json, bytes: self._post_bytes, np.ndarray: self._post_numpy}[type(data)](Path(key).with_suffix(""), data)

    def _post_json(self, key: str, data: dict) -> None: json.dump(data, open(Path(os.path.join(self.registry_path, key)).with_suffix(".json"), "w"))
    def _post_bytes(self, key: str, data: bytes) -> None: open(Path(os.path.join(self.registry_path, key)).with_suffix(".bytes"), "wb").write(data)
    def _post_numpy(self, key: str, data: np.ndarray) -> None: np.save(Path(os.path.join(self.registry_path, key)).with_suffix(".npy"), data)

    def cleanup(self) -> None:
        # Delete used Verdis videos
        # Delete Supabase old videos
        pass

class ResultRegistryClass(Registry):

    def cleanup(self) -> None: pass

    def EXISTS(self, path: str, model: str, project: str) -> bool:
        return Path(os.path.join(self.registry_path, str(Path(path.replace(DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH)).with_suffix("")), model, project)).is_dir()

class DataRegistryClass(Registry):
    def devices(self) -> list[str]: return [os.path.join(dir, x) for dir in [dir for dir in os.listdir(self.registry_path) if os.path.isdir(os.path.join(self.registry_path, dir))] for x in os.listdir(os.path.join(self.registry_path, dir)) if os.path.isdir(os.path.join(self.registry_path, dir, x))]

    # DELETE methods

DataRegistry = DataRegistryClass(DATA_REGISTRY_PATH)
ResultRegistry = ResultRegistryClass(RESULTS_REGISTRY_PATH)

if __name__ == "__main__":
    # DataRegistry.SYNC()
    data = DataRegistry.LIST("argo")
    # print(DataRegistry.devices())
    # ResultRegistry.add_project("test", "test")
    # print(ResultRegistry.projects)