
# Improvements once storage / compute becomes an issue:
# - Use minikeyvalue with .host() option to pull data to multiple nodes
# - Store files in a compressed format
# - Remove supabase dependency
# - Data registry and compute on same machine. Once that is no longer true, GET should return bytes, not path.

from easysort.helpers import REGISTRY_PATH, T, SUPABASE_URL, SUPABASE_KEY, SUPABASE_DATA_REGISTRY_BUCKET, REGISTRY_REFERENCE_SUFFIXES, \
    REGISTRY_REFERENCE_TYPES_MAPPING_TO_PATH, REGISTRY_REFERENCE_TYPES_MAPPING_FROM_PATH, REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE
from supabase import create_client, Client

import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Callable, Dict, List, Any
import json
from tqdm.contrib.concurrent import thread_map
from dataclasses import make_dataclass, is_dataclass, asdict, fields
from dacite import from_dict, Config
import concurrent.futures
import time
import datetime
from uuid import uuid4
import hashlib
import shutil


class RegistryBase:
    """
    A simple registry to keep track of models and data based on dataclasses.

    You can make your own dataclass by inheriting from RegistryBase.DefaultTypes.BASECLASS.
    You are welcome to use the Default Metadata class or use your own.
    You should never change any of the DefaultTypes. If you absolutely must, then make sure your registry is empty.

    DefaultTypes will create ids and refer to them by themselves.
    When you add a new dataclass as a type, you must first create an id with .get_id and set the default value in the dataclass to that value.

    The reason we have ids is to make sure the intent of the dataclass is used and from that allow it to evolve.
    If you evolve a dataclass, you will get an error stating you need to port the old dataclass to the new one.
    In this way, you can trust all your 
    If you are not able to port, you can decide to force an update, meaning all your old data will be forgotten.
    If you want to save multiple version with the same intent, rename your dataclass and generate a new id.
    """    
    class BaseDefaultTypes:
        BASECLASS = make_dataclass("BaseClass", [("id", str)])
        BASEMETADATA = make_dataclass("MetaClass", [("model", str), ("created_at", str)])
        DEFAULT_DETECTION_DATACLASS = make_dataclass("Detection", [("x1", float), ("y1", float), ("x2", float), ("y2", float), ("conf", float), ("cls", str)])
        DEFAULT_WASTE_DATACLASS = make_dataclass("WasteDetection", [("fraction", str), ("sub_fraction", str), ("purity", float), ("weight_kg", float), ("co2_kg", float)])

    class DefaultMarkers:
        ORIGINAL_MARKER = make_dataclass("OriginalMarker", [("OriginalMarker", Any)]) # To interact with the original video/image/json/etc.
        REF_MARKER = make_dataclass("RefMarker", [("RefMarker", Any)]) # To interact with the reference video/image/json/etc.

        @classmethod
        def list(cls): return [_type for _, _type in vars(cls).items() if is_dataclass(_type)]

    class DefaultTypes: # Or use your own dataclass by inheriting from BASECLASS and BASEMETADATA
        # Final Result types needs BaseClass and BaseMetaClass, so they're updated and defined after RegistryBase Ends
        RESULT_PEOPLE = ...
        RESULT_WASTE = ...

        @classmethod
        def list(cls): return [_type for _, _type in vars(cls).items() if is_dataclass(_type)]

    def __init__(self, registry_path: Path): 
        self.registry_path = registry_path
        os.makedirs(self.registry_path, exist_ok=True)
        self._hash_lookup = json.load(open(self.registry_path / ".hash_lookup.json", "r", encoding="utf-8")) if os.path.exists(self.registry_path / ".hash_lookup.json") else {}
        for _type in self.DefaultTypes.list() + self.DefaultMarkers.list(): self._update_hash_lookup(self.get_id(_type), self._hash(_type))

    def _delete_hash(self, id: str, hash: str) -> None:
        assert id in self._hash_lookup, "The id you're trying to delete is not in the hash lookup. Make sure you the pair you are trying to delete is correct."
        assert self._hash_lookup[id] == hash, "The id you're trying to delete does not match with the expected hash. Make sure you the pair you are trying to delete is correct."
        del self._hash_lookup[id]
        with open(self.registry_path / ".hash_lookup.json", "w", encoding="utf-8") as f: json.dump(self._hash_lookup, f, indent=4)

    def _update_hash_lookup(self, id: str, hash: str) -> str:
        self._hash_lookup[id] = hash
        with open(self.registry_path / ".hash_lookup.json", "w", encoding="utf-8") as f: json.dump(self._hash_lookup, f, indent=4)
        return id

    def _hash(self, _type: T) -> str:
        assert is_dataclass(_type), f"Type must be a dataclass, but is {type(_type)}"
        structure = str([(f.name, str(f.type)) for f in fields(_type)])
        return hashlib.sha256(structure.encode()).hexdigest()

    def _convert_int_keys(self, data):
        if isinstance(data, dict): return {int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k: self._convert_int_keys(v) for k, v in data.items()}
        if isinstance(data, list): return [self._convert_int_keys(item) for item in data]
        return data

    def _get_ref_path(self, key: Path, ref: Path) -> Path: 
        path = self.registry_path / key.with_suffix("") / "refs" / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _construct_path(self, key: Path, _type: T) -> Path:
        if isinstance(key, str): key = Path(key)
        assert isinstance(key, Path), f"Key {key} is not a Path, but a {type(key)}"
        assert _type is self.DefaultMarkers.ORIGINAL_MARKER or (self.registry_path / key).exists(), f"Original marker {key} does not exist"
        if _type is self.DefaultMarkers.ORIGINAL_MARKER: return Path(self.registry_path / key) # TODO: Automatic Suffix?
        assert is_dataclass(_type) or _type in self.DefaultMarkers.list(), f"Type {_type} is not a dataclass, but a {type(_type)}"
        assert self._hash(_type) in self._hash_lookup.values(), f"Hash {self._hash(_type)} not found in hash lookup. Check your id is correct using: .get_id(_type)"
        if _type in self.DefaultTypes.list() or _type in self.DefaultMarkers.list(): id_value = next((k for k, v in self._hash_lookup.items() if v == self._hash(_type)), None)# Allow reverse ID detection for static default types
        else: id_value = next((f.default_factory() for f in fields(_type) if f.name == "id"), None)
        assert id_value is not None or len(id_value) == 0, f"Type {_type} does not have a default id value or default id value is 0"
        return Path(self.registry_path / key.with_suffix("") / self._hash_lookup[id_value]).with_suffix(".json")

    def get_id(self, _type: T) -> str: 
        assert is_dataclass(_type), f"Type {_type} is not a dataclass, but a {type(_type)}"
        assert _type in self.DefaultMarkers.list() or (any(f.name == "metadata" for f in fields(_type)) and any(f.name == "id" for f in fields(_type))), f"Type {_type} does not have a metaclass and/or id field, but the following fields: {fields(_type)}"
        if (k := next((k for k, v in self._hash_lookup.items() if v == self._hash(_type)), None)): return k
        return self._update_hash_lookup(str(uuid4()), self._hash(_type))

    def add_id(self, _type: T, id: str) -> None:
        assert is_dataclass(_type), f"Type {_type} is not a dataclass, but a {type(_type)}"
        assert _type in self.DefaultMarkers.list() or (any(f.name == "metadata" for f in fields(_type)) and any(f.name == "id" for f in fields(_type))), f"Type {_type} does not have a metaclass and/or id field, but the following fields: {fields(_type)}"
        assert id is not None and len(id) == 36 and id.count("-") == 4, f"Id {id} is not a valid uuid4"
        if id in self._hash_lookup and self._hash_lookup[id] == self._hash(_type): return
        assert id not in self._hash_lookup, f"Id {id} already exists"
        self._update_hash_lookup(id, self._hash(_type))
    
    def GET(self, key: Path, _type: T, throw_error: bool = True, ref: Optional[Path] = None) -> Optional[T]:
        f"""
        If _type is REF_MARKER, ref must be provided and the reference will be returned.
        The supported reference types are: {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys()}.
        """
        if ref is not None or _type is self.DefaultMarkers.REF_MARKER:
            assert ref is not None, "ref must be provided when _type is REF_MARKER"
            return self._get_ref(key, ref)
        path = self._construct_path(key, _type)
        if not path.is_file() and throw_error: raise FileNotFoundError(f"File {path} not found")
        if not path.is_file() and not throw_error: return None
        if _type is self.DefaultMarkers.ORIGINAL_MARKER: return REGISTRY_REFERENCE_TYPES_MAPPING_FROM_PATH[key.suffix](path)
        data = REGISTRY_REFERENCE_TYPES_MAPPING_FROM_PATH[key.suffix](path)
        return from_dict(data_class=_type, data=self._convert_int_keys(data), config=Config(check_types=False)) if is_dataclass(_type) else data
    
    def _get_ref(self, key: Path, ref: Path) -> REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys():
        assert ref.suffix in REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys(), f"Ref {ref} must have a supported suffix: {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys()}"
        ref_path = self._get_ref_path(key, ref)
        if not ref_path.is_file(): raise FileNotFoundError(f"Ref {ref} not found")
        return REGISTRY_REFERENCE_TYPES_MAPPING_FROM_PATH[ref.suffix](ref_path)

    def POST(self, key: Path, data: T, _type: T, overwrite: bool = False, refs: Optional[Dict[str | Path, REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys()]] = {}) -> None:
        path = self._construct_path(key, _type)
        assert all(isinstance(ref_data, REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE[ref_path.suffix]) for ref_path, ref_data in refs.items() if refs is not None), f"Refs {refs} are not a dict of {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys()}"
        assert isinstance(data, _type) or _type in self.DefaultMarkers.list(), f"Data {data} is not a {_type}"
        assert not path.is_file() or overwrite, f"File {key} already exists, and you chose to not overwrite"
        assert _type is not self.DefaultMarkers.ORIGINAL_MARKER or (key.suffix in REGISTRY_REFERENCE_SUFFIXES and isinstance(data, REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE[key.suffix])), \
            f"Original marker {key} must have a supported suffix: {REGISTRY_REFERENCE_SUFFIXES} and data must be a {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE[key.suffix]}. Received {type(data)} with suffix {key.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if _type is self.DefaultMarkers.ORIGINAL_MARKER: REGISTRY_REFERENCE_TYPES_MAPPING_TO_PATH[key.suffix](path, data)
        else: 
            assert is_dataclass(_type), f"Data {data} is not a dataclass, but a {type(data)}. Currently only dataclasses are supported."
            REGISTRY_REFERENCE_TYPES_MAPPING_TO_PATH[".json"](path, asdict(data))
            with open(path.with_suffix(".schema.json"), "w", encoding="utf-8") as f: json.dump({f.name: str(f.type) for f in fields(_type)}, f, indent=4)
        if refs is None: return
        for ref_path, data in refs.items(): REGISTRY_REFERENCE_TYPES_MAPPING_TO_PATH[ref_path.suffix](self._get_ref_path(key, ref_path), data)

    def DELETE(self, key: Path, _type: T) -> None:
        "Deletes the data excluding references. If _type is ORIGINAL_MARKER, then everything related to this key will be deleted including references."
        if _type is not self.DefaultMarkers.ORIGINAL_MARKER: os.remove(self._construct_path(key, _type))
        else:
            os.remove(self._construct_path(key, _type))
            results_dir = self.registry_path / key.with_suffix("")
            if results_dir.exists(): shutil.rmtree(results_dir)
        assert not self.EXISTS(key, _type), f"Data {key} still exists after deletion"


    def LIST(self, prefix: Optional[str] = "", suffix: Optional[List[str] | str] = REGISTRY_REFERENCE_SUFFIXES, cond: Optional[Callable[[str], bool]] = None, return_all: bool = False) -> list[Path]:
        files = [Path(os.path.join(root, file)) for root, _, files in tqdm(os.walk(Path(self.registry_path) / prefix), desc="Listing files") for file in files]
        files = [file for file in files if (file.suffix in list(suffix) if suffix else True) and not file.name.startswith("._")]
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

    def EXISTS(self, key: Path, _type: T) -> bool: return self._construct_path(key, _type).is_file()
    def EXPECT(self):
        # Potentially a EXPECT lazily generating missing results by returning list of bool, then using a specified func to generate results.
        pass

RegistryBase.DefaultTypes.RESULT_PEOPLE = make_dataclass("RESULT_PEOPLE", [("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA), ("frame_results", Dict[int, List["RegistryBase.BaseDefaultTypes.DEFAULT_DETECTION_DATACLASS"]])], bases=(RegistryBase.BaseDefaultTypes.BASECLASS,))
RegistryBase.DefaultTypes.RESULT_WASTE = make_dataclass("RESULT_WASTE", [("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA), ("frame_results", Dict[int, List["RegistryBase.BaseDefaultTypes.DEFAULT_WASTE_DATACLASS"]])], bases=(RegistryBase.BaseDefaultTypes.BASECLASS,))


Registry = RegistryBase(REGISTRY_PATH)

if __name__ == "__main__":
    Registry.SYNC()
    print(Registry.LIST("argo")[0])

