
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
    def __init__(self, registry_path: str): self.registry_path = registry_path

    def SYNC(self) -> None: # with Supabase
        assert self.registry_path == DATA_REGISTRY_PATH, "SYNC should only be applied to the data registry"
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
        print("Sync complete")

    def GET(self, key: str) -> bytes:
        assert os.path.exists(os.path.join(self.registry_path, key)), f"File {key} not found"
        return os.path.join(self.registry_path, key)

    def LIST(self, prefix: Optional[str] = "") -> list[str]:
        return [str(x) for x in (Path(self.registry_path) / prefix).glob("**/*.mp4") if x.is_file() and not x.name.startswith("._")]

    def POST(self, key: str, data: dict|bytes|np.ndarray|list) -> None: {dict: self._post_json, list: self._post_json, bytes: self._post_bytes, np.ndarray: self._post_numpy}[type(data)](Path(key).with_suffix(""), data)
    def _post_json(self, key: str, data: dict) -> None: json.dump(data, open(Path(os.path.join(self.registry_path, key)).with_suffix(".json"), "w"))
    def _post_bytes(self, key: str, data: bytes) -> None: open(Path(os.path.join(self.registry_path, key)).with_suffix(".bytes"), "wb").write(data)
    def _post_numpy(self, key: str, data: np.ndarray) -> None: np.save(Path(os.path.join(self.registry_path, key)).with_suffix(".npy"), data)

class ResultRegistryClass(Registry):
    def __init__(self, registry_path: str):
        super().__init__(registry_path)
        os.makedirs(self.registry_path, exist_ok=True)
        self.projects = open(os.path.join(self.registry_path, "projects.txt")).read().splitlines() if os.path.exists(os.path.join(self.registry_path, "projects.txt")) else []

    def POST(self, path: str, model: str, project: str, identifier: str, data: dict|bytes|np.ndarray) -> None:
        if "/mnt/" in path: path = path.replace(DATA_REGISTRY_PATH, RESULTS_REGISTRY_PATH) # This should never happen if GET is used
        self.add_project(model, project)
        os.makedirs(os.path.join(self.registry_path, str(Path(path).with_suffix("")), model, project, identifier), exist_ok=True)
        super().POST(os.path.join(self.registry_path, str(Path(path).with_suffix("")), model, project, identifier), data)

    def add_project(self, model: str, project: str) -> None: 
        if os.path.join(model, project) in self.projects: return
        self.projects = sorted(list(set(self.projects + [os.path.join(model, project)])))
        open(os.path.join(self.registry_path, "projects.txt"), "w").write("\n".join(self.projects))

    def cleanup(self) -> None: pass

    def EXISTS(self, path: str, model: str, project: str) -> bool:
        return os.path.exists(os.path.join(self.registry_path, str(Path(path).with_suffix("")), model, project))

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