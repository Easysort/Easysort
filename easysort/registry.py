
# Improvements once storage / compute becomes an issue:
# - Use minikeyvalue with .host() option to pull data to multiple nodes
# - Store files in a compressed format
# - Remove supabase dependency
# - Data registry and compute on same machine. Once that is no longer true, GET should return bytes, not path.

from easysort.helpers import REGISTRY_PATH, T, SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET, TESTING, REGISTRY_REFERENCE_TYPES
from supabase import create_client, Client

import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Callable, Dict, List, Any
import json
from tqdm.contrib.concurrent import thread_map
from dataclasses import make_dataclass, dataclass, is_dataclass, asdict, fields
import pandas as pd
import concurrent.futures
import time
import datetime
from uuid import uuid4
import hashlib
import numpy as np
from PIL import Image


class RegistryBase:
    """
    A simple registry to keep track of models and data based on dataclasses.

    You can make your own dataclass by inheriting from RegistryBase.DefaultTypes.BASECLASS.
    You are welcome to use the Default Metadata class or use your own.
    You should never change any of the DefaultTypes. If you absolutely must, then make sure your registry is empty.
    """    
    class DefaultTypes: # Or use your own dataclass by inheriting from BASECLASS
        ORIGINAL_MARKER = make_dataclass("OriginalMarker", [("OriginalMarker", Any)]) # To get the original video/image/json/etc.

        BASECLASS = make_dataclass("BaseClass", [("id", str)])
        METADATA_CLASS = make_dataclass("MetaClass", [("model", str), ("created_at", datetime.datetime)])
        DEFAULT_DETECTION_DATACLASS = make_dataclass("Detection", [("x1", float), ("y1", float), ("x2", float), ("y2", float), ("conf", float), ("cls", str)])
        DEFAULT_WASTE_DATACLASS = make_dataclass("WasteDetection", [("fraction", str), ("sub_fraction", str), ("purity", float), ("weight_kg", float), ("co2_kg", float)])
        RESULT_PEOPLE = make_dataclass("RESULT_PEOPLE", [("metadata", METADATA_CLASS), ("frame_results", Dict[int, List["RegistryBase.DefaultTypes.DEFAULT_DETECTION_DATACLASS"]])], bases=(BASECLASS,))
        RESULT_WASTE = make_dataclass("RESULT_WASTE", [("metadata", METADATA_CLASS), ("frame_results", Dict[int, List["RegistryBase.DefaultTypes.DEFAULT_WASTE_DATACLASS"]])], bases=(BASECLASS,))
        # RESULT_YOLOS = make_dataclass("RESULT_YOLOS", [("metadata", META_CLASS), ("frame_results", ...)])

    def __init__(self, registry_path: Path): 
        self.registry_path = registry_path
        self._hash_lookup = json.load(open(self.registry_path / ".hash_lookup.json", "r", encoding="utf-8")) if os.path.exists(self.registry_path / ".hash_lookup.json") else {}
        os.makedirs(self.registry_path, exist_ok=True)

    def _delete_hash(self, id: str, hash: str) -> None:
        assert self._hash_lookup[id] == hash, "The id you're trying to delete does not match with the expected hash. Make sure you the pair you are trying to delete is correct."
        del self._hash_lookup[id]
        with open(self.registry_path / ".hash_lookup.json", "w", encoding="utf-8") as f: json.dump(self._hash_lookup, f, indent=4)

    def _update_hash_lookup(self, id: str, hash: str) -> str:
        self._hash_lookup[id] = hash
        with open(self.registry_path / ".hash_lookup.json", "w", encoding="utf-8") as f: json.dump(self._hash_lookup, f, indent=4)
        return id

    def _hash(self, _type: T) -> str:
        structure = str([(f.name, str(f.type)) for f in fields(_type)])
        return hashlib.sha256(structure.encode()).hexdigest()

    def _construct_path(self, key: Path, _type: T) -> Path:
        assert isinstance(key, Path), f"Key {key} is not a Path, but a {type(key)}"
        assert is_dataclass(_type) or _type in self.DefaultTypes, f"Type {_type} is not a dataclass, but a {type(_type)}"
        assert self._hash(_type) in self._hash_lookup, f"Hash {self._hash(_type)} not found in hash lookup. You either made a mistake in the dataclass id or "
        id_value = next((f.default_factory() for f in fields(_type) if f.name == "id"), None)
        assert id_value is not None, f"Type {_type} does not have a default id value"
        return Path(self.registry_path / key.with_suffix("") / self._hash_lookup[id_value]).with_suffix(".parquet")

    def get_new_id(self, _type: T) -> str: 
        assert is_dataclass(_type), f"Type {_type} is not a dataclass, but a {type(_type)}"
        assert any(f.name == "metadata" for f in fields(_type)) and any(f.name == "id" for f in fields(_type)), f"Type {_type} does not have a metaclass and/or id field. Make sure to inherit from BASECLASS and have metadata."
        if (k := next((k for k, v in self._hash_lookup.items() if v == self._hash(_type)), None)): return k
        return self._update_hash_lookup(str(uuid4()), self._hash(_type))

    def GET(self, key: Path, _type: T, throw_error: bool = True, ref: Optional[Path] = None) -> Optional[T]:
        """
        If ref is provided, then the loaded data from the reference file will be returned.

        """
        path = self._construct_path(key, _type)
        if not path.is_file() and throw_error: raise FileNotFoundError(f"File {path} not found")
        if not path.is_file() and not throw_error: return None
        return _type(**pd.read_parquet(path).iloc[0].to_dict())


    def POST(self, key: Path, data: T, _type: T, overwrite: bool = False, refs: Optional[Dict[str | Path, REGISTRY_REFERENCE_TYPES]] = None) -> None:
        path = self._construct_path(key, _type)
        assert all(isinstance(ref, (np.ndarray, dict, Path)) for ref in refs.values()), f"Refs {refs} are not a dict of numpy arrays, json objects or paths"
        assert isinstance(data, _type), f"Data {data} is not a {_type}"
        assert not overwrite or not path.is_file(), f"File {key} already exists"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(data)]).to_parquet(path)
        with open(path.with_suffix(".schema.json"), "w") as f: json.dump(asdict(_type), f)

    def LIST(self, prefix: Optional[str] = "", suffix: Optional[List[str] | str] = [".mp4", ".jpeg", ".jpg", ".png"], cond: Optional[Callable[[str], bool]] = None, return_all: bool = False) -> list[Path]:
        files = [Path(os.path.join(root, file)) for root, _, files in tqdm(os.walk(Path(self.registry_path) / prefix), desc="Listing files") for file in files]
        files = [file for file in files if (file.suffix in list(suffix) if suffix else True) and not str(file).startswith("._")]
        cond_files = [file for file in tqdm(files, desc="Filtering files") if cond(file)] if cond else files
        return [files, cond_files] if return_all else cond_files

    def SYNC(self) -> None:
        CONCURRENT_WORKERS = 1 # Depends on connection speed, supabase rate limit, so on. Currently 3 works fairly stable and fast
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
            if ".jpg" in file: 
                files_to_delete.append(file)
                continue
            if len(file.split("/")) < 6: 
                print("Skipping file: ", file, "because it has less than 6 parts")
                continue
            year, month, day, hour, minute, second = file.split("/")[-5], file.split("/")[-4], file.split("/")[-3], file.split("/")[-1][:2], file.split("/")[-1][2:4], file.split("/")[-1][4:6]
            timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            if timestamp < datetime.datetime.now() - datetime.timedelta(weeks=4): files_to_delete.append(file)

        print(f"Deleting {len(files_to_delete)} files" if len(files_to_delete) > 0 else "No files to delete")
        if len(files_to_delete) == 0: return
        for i in tqdm(range(0, len(files_to_delete), 100), desc="Deleting files"):
            supabase_client.storage.from_(SUPABASE_DATA_REGISTRY_BUCKET).remove(files_to_delete[i:i+100])
            time.sleep(1)

    def EXISTS(self, key: Path, _type: T) -> bool: return self._check_path(key, _type).is_file()
    def EXPECT(self):
        # Potentially a EXPECT lazily generating missing results by returning list of bool, then using a specified func to generate results.
        pass


Registry = RegistryBase(REGISTRY_PATH)

if __name__ == "__main__":
    Registry.SYNC()
    print(Registry.LIST("argo")[0])

