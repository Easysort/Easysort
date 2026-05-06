
# Improvements once storage / compute becomes an issue:
# - Use minikeyvalue with .host() option to pull data to multiple nodes
# - Store files in a compressed format
# - Remove supabase dependency
# - Data registry and compute on same machine. Once that is no longer true, GET should return bytes, not path.

from httpx import request
from easysort.helpers import T, REGISTRY_REFERENCE_SUFFIXES, REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES, REGISTRY_REFERENCE_TYPES_MAPPING_FROM_BYTES,\
    REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE, RefData

from pathlib import Path
from tqdm import tqdm
from typing import Optional, Callable, Dict, List, Any, Sequence, Literal, cast, overload
import json
from urllib.parse import quote
from dataclasses import make_dataclass, is_dataclass, asdict, fields, MISSING
from dacite import from_dict, Config
import time
from uuid import uuid4
import hashlib
import sys


class RegistryConnector:
    def __init__(self, base: str | None = None, *, timeout: float = 30):
        if base is None: print("Warning: REGISTRY_LOCAL_IP is not set, using default 'localhost:3099'")
        base = (base or "localhost:3099").rstrip("/")
        print("Registry Connector initialized with base:", base)
        if not base: raise ValueError("Missing REGISTRY_LOCAL_IP / base URL (e.g. http://localhost:3000)")
        self.base = base if base.startswith(("http://", "https://")) else "http://" + base
        self.timeout = timeout

    def _url(self, key: str | Path = "", query: str = "") -> str:
        k = str(key).lstrip("/")
        return (f"{self.base}/{quote(k, safe='/')}" if k else f"{self.base}/") + (query or "")

    def _req(self, method: str, key: str | Path = "", *, query: str = "", content: bytes | str | None = None, ignore_errors: bool = False):
        if isinstance(content, str): content = content.encode()
        r = request(method, self._url(key, query), content=content, follow_redirects=True, timeout=self.timeout)
        if ignore_errors: return r
        if r.status_code == 404: raise FileNotFoundError(str(key), self._url(key, query))
        if r.status_code == 403: raise PermissionError(str(key), self._url(key, query))
        if r.status_code >= 400: raise RuntimeError(f"{method} {key}{query} -> {r.status_code}: {r.text[:200]}")
        return r

    def POST(self, key: str | Path, data: bytes | str, allow_overwrite: bool = False, ignore_already_exists: bool = False) -> None:
        if allow_overwrite and self.EXISTS(key): self.DELETE(key)
        r = self._req("PUT", key, content=data, ignore_errors=ignore_already_exists)
        if ignore_already_exists and r.status_code == 403:
            print(key, "already exists, skipping")
            return
        if r.status_code != 201: raise RuntimeError(f"PUT {key} -> {r.status_code}: {r.text[:200]}")

    def GET_MULTIPLE(self, keys: Sequence[str | Path]) -> list[bytes]:
        listed_keys = self.LIST()
        return [self.GET(key) for key in keys if key in listed_keys]

    def GET(self, key: str | Path) -> bytes: return self._req("GET", key).content
    def DELETE(self, key: str | Path, ignore_errors: bool = False) -> None: self._req("DELETE", key, ignore_errors=ignore_errors)
    def UNLINK(self, key: str | Path) -> None: self._req("UNLINK", key)
    def LIST(self, prefix: str | Path = "") -> list[Path]: return self._keys(self._req("GET", prefix, query="?list"))
    def UNLINKED(self) -> list[Path]: return self._keys(self._req("GET", query="?unlinked"))
    def PUT_FILE(self, key: str | Path, local_path: str | Path) -> None: self.POST(key, Path(local_path).read_bytes())
    def GET_FILE(self, key: str | Path, local_path: str | Path) -> None: Path(local_path).write_bytes(self.GET(key))
    def EXISTS(self, key: str | Path) -> bool: return self._req("HEAD", key, ignore_errors=True).status_code == 200
    def URL(self, key: str | Path) -> str: return self._url(key)

    def EXISTS_MULTIPLE(self, keys: Sequence[str | Path]) -> list[bool]:
        listed_keys = self.LIST()
        return [Path(key) in listed_keys for key in tqdm(keys, desc="Checking if keys exist")]

    @staticmethod
    def _keys(r) -> list[Path]:
        try:
            j = r.json()
            if isinstance(j, dict) and isinstance(j.get("keys"), list): return [Path(str(k).lstrip("/")) for k in j["keys"]]
            if isinstance(j, list): return [Path(str(k).lstrip("/")) for k in j]
        except Exception: pass
        return [Path(s.lstrip("/")) for line in r.text.splitlines() if (s := line.strip())]


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
    In this way, you can trust all your dataclasses work as expected, no data are missing, and same code for all versions as you evolve.
    If you are not able to port, you can decide to force an update, meaning all your old data will be forgotten.
    If you want to save multiple version with the same intent, rename your dataclass and generate a new id.
    """
    class BaseDefaultTypes:
        BASECLASS = make_dataclass("BaseClass", [("id", str)])
        BASEMETADATA = make_dataclass("MetaClass", [("model", str), ("created_at", str)])
        DEFAULT_DETECTION_DATACLASS = make_dataclass("Detection", \
            [("x1", float), ("y1", float), ("x2", float), ("y2", float), ("conf", float), ("cls", str)])
        DEFAULT_WASTE_DATACLASS = make_dataclass("WasteDetection", \
            [("fraction", str), ("sub_fraction", str), ("purity", float), ("weight_kg", float), ("co2_kg", float)])

    class DefaultMarkers:
        ORIGINAL_MARKER: Any = make_dataclass("OriginalMarker", [("OriginalMarker", Any)]) # To interact with the original video/image/json/etc.
        REF_MARKER: Any = make_dataclass("RefMarker", [("RefMarker", Any)]) # To interact with the reference video/image/json/etc.

        @classmethod
        def list(cls): return [_type for _, _type in vars(cls).items() if is_dataclass(_type)]

    class DefaultTypes: # Or use your own dataclass by inheriting from BASECLASS and BASEMETADATA
        # Final Result types need BaseClass and BaseMetaClass, so they're populated below the RegistryBase class.
        RESULT_PEOPLE: Any = None
        RESULT_WASTE: Any = None

        @classmethod
        def list(cls): return [_type for _, _type in vars(cls).items() if is_dataclass(_type)]

    def __init__(self, registry_connector: RegistryConnector | None = None, base: str | None = None):
        if registry_connector is None and base is None: raise ValueError("Either registry_connector or base must be provided")
        if registry_connector is None: registry_connector = RegistryConnector(base)
        self.backend = registry_connector
        print("RegistryConnector created with base:", self.backend.base)
        self._hash_lookup_path = "hash_lookup.json"
        for _type in self.DefaultTypes.list() + self.DefaultMarkers.list(): self._update_hash_lookup(self.get_id(_type), self._hash(_type))

    def _get_hash_lookup(self) -> dict[str, str]:
        return json.loads(self.backend.GET(self._hash_lookup_path))

    def _delete_hash(self, _id: str, _hash: str) -> None:
        hash_lookup = self._get_hash_lookup()
        assert_statement = "The id you're trying to delete is not in the hash lookup. Make sure you the pair you are trying to delete is correct."
        assert _id in hash_lookup, assert_statement
        assert hash_lookup[_id] == _hash, assert_statement
        del hash_lookup[_id]
        self.backend.POST(self._hash_lookup_path, json.dumps(hash_lookup), allow_overwrite=True)

    def _update_hash_lookup(self, _id: str, _hash: str) -> str:
        hash_lookup = self._get_hash_lookup()
        hash_lookup[_id] = _hash
        self.backend.POST(self._hash_lookup_path, json.dumps(hash_lookup), allow_overwrite=True)
        return _id

    def _hash(self, _type: T) -> str:
        assert is_dataclass(_type), f"Type must be a dataclass, but is {type(_type)}"
        structure = str([(f.name, str(f.type)) for f in fields(_type)])
        return hashlib.sha256(structure.encode()).hexdigest()

    def _convert_int_keys(self, data):
        if isinstance(data, dict):
            return {int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k: self._convert_int_keys(v) for k, v in data.items()}
        if isinstance(data, list): return [self._convert_int_keys(item) for item in data]
        return data

    def _construct_path(self, key: Path | str, _type: T) -> Path:
        assert isinstance(key, Path) or isinstance(key, str), f"Key {key} is not a Path or str, but a {type(key)}"
        if isinstance(key, str): key = Path(key)
        assert not str(key).startswith("."), "Key cannot start with a dot"
        assert _type is self.DefaultMarkers.ORIGINAL_MARKER or self.backend.EXISTS(key), f"Original marker {key} does not exist"
        if _type is self.DefaultMarkers.ORIGINAL_MARKER: return key # TODO: Automatic Suffix?
        assert is_dataclass(_type), f"Type {_type} is not a dataclass, but a {type(_type)}"
        hash_lookup = self._get_hash_lookup()
        assert self._hash(_type) in hash_lookup.values(),\
            f"Hash {self._hash(_type)} not found in hash lookup. Check your id is correct using: .get_id(_type)"
        if _type in self.DefaultTypes.list() or _type in self.DefaultMarkers.list():
            id_value = next((k for k, v in hash_lookup.items() if v == self._hash(_type)), None)# Allow reverse ID detection for static default types
        else: id_value = next((f.default_factory() for f in fields(_type) if f.name == "id" and f.default_factory is not MISSING), None)
        assert id_value is not None and len(id_value) > 0, f"Type {_type} does not have a default id value or default id value is 0"
        return Path(key.with_suffix("") / hash_lookup[id_value]).with_suffix(".json")

    def _construct_many_paths(self, keys: Sequence[Path | str], _type: T) -> list[Path]:
        paths = [Path(k) for k in keys]
        assert not any(str(p).startswith(".") for p in paths), "Key cannot start with a dot"
        assert _type is self.DefaultMarkers.ORIGINAL_MARKER or self.backend.EXISTS(paths[0]), f"Original marker {paths[0]} does not exist"
        if _type is self.DefaultMarkers.ORIGINAL_MARKER: return paths # TODO: Automatic Suffix?
        assert is_dataclass(_type), f"Type {_type} is not a dataclass, but a {type(_type)}"
        hash_lookup = self._get_hash_lookup()
        assert self._hash(_type) in hash_lookup.values(), \
            f"Hash {self._hash(_type)} not found in hash lookup. Check your id is correct using: .get_id(_type)"
        if _type in self.DefaultTypes.list() or _type in self.DefaultMarkers.list():
            id_value = next((k for k, v in hash_lookup.items() if v == self._hash(_type)), None)# Allow reverse ID detection for static default types
        else: id_value = next((f.default_factory() for f in fields(_type) if f.name == "id" and f.default_factory is not MISSING), None)
        assert id_value is not None and len(id_value) > 0, f"Type {_type} does not have a default id value or default id value is 0"
        return [Path(path.with_suffix("") / hash_lookup[id_value]).with_suffix(".json") for path in paths]

    def get_id(self, t: T) -> str:
        assert is_dataclass(t), f"Type {t} is not a dataclass, but a {type(t)}"
        assert t in self.DefaultMarkers.list() or (any(f.name == "metadata" for f in fields(t)) and any(f.name == "id" for f in fields(t))), \
            f"Type {t} does not have a metaclass and/or id field, but the following fields: {fields(t)}"
        if (k := next((k for k, v in self._get_hash_lookup().items() if v == self._hash(t)), None)): return k
        # TODO: Check id not already exists
        return self._update_hash_lookup(str(uuid4()), self._hash(t))

    def port_ids(self, original_id: str, new_id: str, _porting_type: T, _new_type: T) -> None:
        pass

    def add_id(self, _type: T, _id: str) -> None:
        assert is_dataclass(_type), f"Type {_type} is not a dataclass, but a {type(_type)}"
        assert _type in self.DefaultMarkers.list() or (any(f.name == "metadata" for f in fields(_type)) and any(f.name == "id" \
            for f in fields(_type))), f"Type {_type} does not have a metaclass and/or id field, but the following fields: {fields(_type)}"
        assert _id is not None and len(_id) == 36 and _id.count("-") == 4, f"Id {_id} is not a valid uuid4"
        hash_lookup = self._get_hash_lookup()
        if _id in hash_lookup and hash_lookup[_id] == self._hash(_type): return
        assert _id not in hash_lookup, f"Id {_id} already exists"
        self._update_hash_lookup(_id, self._hash(_type))

    def GET(self, key: Path, _type: T, throw_error: bool = True, ref: Optional[Path] = None) -> Any:
        f"""
        If _type is REF_MARKER, ref must be provided and the reference will be returned.
        The supported reference types are: {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys()}.
        """
        assert is_dataclass(_type), f"Type {_type} is not a dataclass, but a {type(_type)}"
        if ref is not None or _type is self.DefaultMarkers.REF_MARKER:
            assert ref is not None, "ref must be provided when _type is REF_MARKER"
            return self._get_ref(key, ref)
        try: data = self.backend.GET(self._construct_path(key, _type))
        except FileNotFoundError:
            if throw_error: raise
            return None
        if _type is self.DefaultMarkers.ORIGINAL_MARKER: return REGISTRY_REFERENCE_TYPES_MAPPING_FROM_BYTES[key.suffix](data)
        data = REGISTRY_REFERENCE_TYPES_MAPPING_FROM_BYTES[".json"](data)
        return from_dict(data_class=cast(type, _type), data=self._convert_int_keys(data), config=Config(check_types=False))

    def _get_ref_path(self, key: Path, ref: Path) -> Path: return key.with_suffix("") / "refs" / ref

    def _get_ref(self, key: Path, ref: Path) -> RefData:
        assert ref.suffix in REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys(), \
            f"Ref {ref} must have a supported suffix: {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys()}"
        ref_path = self._get_ref_path(key, ref)
        if not self.backend.EXISTS(ref_path): raise FileNotFoundError(f"Ref {ref} not found")
        return REGISTRY_REFERENCE_TYPES_MAPPING_FROM_BYTES[ref.suffix](self.backend.GET(ref_path))

    def POST(self, key: Path, data: T, _type: T, overwrite: bool = False,
        refs: Optional[Dict[Path, RefData]] = None) -> None:
        original_key = key
        key = self._construct_path(key, _type)
        assert refs is None or all(isinstance(ref_data, REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE[ref_path.suffix])
            for ref_path, ref_data in refs.items()), f"Refs {refs} are not a dict of {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE.keys()}"
        assert isinstance(data, cast(type, _type)) or _type in self.DefaultMarkers.list(), f"Data {data} is not a {_type}"
        assert not self.backend.EXISTS(key) or overwrite, f"File {key} already exists, and you chose to not overwrite"
        assert _type is not self.DefaultMarkers.ORIGINAL_MARKER or (key.suffix in REGISTRY_REFERENCE_SUFFIXES \
            and isinstance(data, REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE[key.suffix])), \
            f"Original marker {key} must have a supported suffix: {REGISTRY_REFERENCE_SUFFIXES} and \
                data must be a {REGISTRY_REFERENCE_SUFFIXES_MAPPING_TO_TYPE[key.suffix]}. Received {type(data)} with suffix {key.suffix}"
        if _type is self.DefaultMarkers.ORIGINAL_MARKER:
            self.backend.POST(key, REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES[key.suffix](data), allow_overwrite=overwrite)
        else:
            assert is_dataclass(_type), f"Data {data} is not a dataclass, but a {type(data)}. Currently only dataclasses are supported."
            self.backend.POST(key, REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES[".json"](asdict(cast(Any, data))), allow_overwrite=overwrite)
            self.backend.POST(key.with_suffix(".schema.json"), REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES[".json"]\
                ({f.name: str(f.type) for f in fields(cast(type, _type))}), allow_overwrite=overwrite)
        if refs is None: return
        for ref_path, ref_data in refs.items(): self.backend.POST(self._get_ref_path(original_key, ref_path), \
            REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES[ref_path.suffix](ref_data), allow_overwrite=overwrite)

    def DELETE(self, key: Path, _type: T) -> None: # Maybe we shouldnt have this, only a periodic cleanup function?
        "Deletes the data excluding references. If _type is ORIGINAL_MARKER, then everything related to this key will be deleted including references"
        if _type is not self.DefaultMarkers.ORIGINAL_MARKER: self.backend.DELETE(self._construct_path(key, _type))
        else:
            self.backend.DELETE(key)
            # Delete all related keys (results, refs, etc.) by listing and deleting keys with this prefix
            prefix = str(key.with_suffix(""))
            for related_key in self.backend.LIST(prefix):
                try: self.backend.DELETE(related_key)
                except FileNotFoundError: pass
        assert not self.EXISTS(key, _type), f"Data {key} still exists after deletion"

    @overload
    def LIST(self, prefix: str = ..., suffix: Optional[Sequence[str] | str] = ..., cond: Optional[Callable[[str], bool]] = ...,
        return_all: Literal[False] = ..., check_exists_with_type: Optional[T] = ...) -> list[Path]: ...
    @overload
    def LIST(self, prefix: str = ..., suffix: Optional[Sequence[str] | str] = ..., cond: Optional[Callable[[str], bool]] = ...,
        *, return_all: Literal[True], check_exists_with_type: Optional[T] = ...) -> tuple[list[Path], list[Path]]: ...
    def LIST(self, prefix: str = "", suffix: Optional[Sequence[str] | str] = REGISTRY_REFERENCE_SUFFIXES,
        cond: Optional[Callable[[str], bool]] = None, return_all: bool = False,
        check_exists_with_type: Optional[T] = None) -> list[Path] | tuple[list[Path], list[Path]]:
        files = self.backend.LIST(prefix)
        print("Number of files: ", len(files))
        files = [file for file in tqdm(files) if (file.suffix in list(suffix) if suffix else True) and not file.name.startswith("._")]
        print("Number of files after suffix filter: ", len(files))
        cond_files = [file for file in tqdm(files, desc="Filtering files") if cond(file)] if cond else files
        print("Number of files after cond filter: ", len(cond_files))
        if check_exists_with_type:
            exists_bools = self.EXISTS_MULTIPLE(cond_files, check_exists_with_type)
            print("Number of files after check_exists_with_type filter: ", sum(exists_bools), "out of", len(cond_files))
            cond_files = [file for file, exists in zip(cond_files, exists_bools) if exists] # Files that exist
        return (files, cond_files) if return_all else cond_files

    def EXISTS(self, key: Path, _type: T) -> bool: return self.backend.EXISTS(self._construct_path(key, _type))

    def EXISTS_MULTIPLE(self, keys: list[Path], _type: T) -> list[bool]:
        if len(keys) == 0: return []
        set_file_list, paths = set(self.LIST()), self._construct_many_paths(keys, _type)
        return [path in set_file_list for path in tqdm(paths)]

    def BACKUP(self, local_path: Path) -> None:
        assert local_path.is_dir(), f"Local path {local_path} is not a directory"
        files = self.LIST()
        bools = self.backend.EXISTS_MULTIPLE(files)
        missing_files = [file for file, exists in zip(files, bools) if not exists]
        print(len(missing_files), "out of", len(files), "files are missing")
        for file in tqdm(missing_files, desc="Backing up files"):
            self.backend.GET_FILE(file, local_path / file)

    def EXPECT(self):
        # Potentially a EXPECT lazily generating missing results by returning list of bool, then using a specified func to generate results.
        pass

    def PUT_FOLDER(self, local_path: Path, prefix: str = "") -> None: # Super slow for some reason (at uploading)
        """Puts a folder into the registry. All files in the folder will have their relative path as the key.
        Example: /local/folder/my_data.jpg -> /prefix/my_data.jpg"""
        assert local_path.is_dir(), f"Local path {local_path} is not a directory"
        new_files = [f.relative_to(local_path) for f in tqdm(local_path.rglob("*")) if f.is_file() \
            and not f.name.startswith("._") and "hash_lookup" not in f.name]
        bools = self.backend.EXISTS_MULTIPLE(new_files)
        print("Files will look like this: ", *new_files[0:5], sep="\n")
        print("Waiting 10 seconds. If this is wrong, exit the program!")
        time.sleep(10)
        new_files = [prefix + str(file) for file in new_files]
        print(len(new_files) - sum(bools), "out of", len(new_files), "files are missing")
        for file, _bool in tqdm(zip(new_files, bools), total=len(new_files)):
            if _bool: continue
            self.backend.PUT_FILE(file, local_path / file)

    def PORT(self, _id: str, old_hash: str, new_type: T, porting_strategy: Optional[Callable[[T], T]] = None, allow_delete: bool = False) -> None:
        assert is_dataclass(new_type), f"New type {new_type} is not a dataclass, but a {type(new_type)}"
        assert new_type in self.DefaultMarkers.list() or \
             (any(f.name == "metadata" for f in fields(new_type)) and any(f.name == "id" for f in fields(new_type))),\
             f"New type {new_type} does not have a metaclass and/or id field, but the following fields: {fields(new_type)}"
        assert _id in self._get_hash_lookup(), f"Id {_id} not found in hash lookup"
        assert self._get_hash_lookup()[_id] == old_hash, f"Id {_id} does not match old_hash {old_hash}. \
            Fetch the current hash with Registry.get_id(your_type) or Registry._get_hash_lookup()[id]."
        files = [f for f in self.backend.LIST() if f.name == f"{old_hash}.json"]
        print(f"Found {len(files)} entries with old hash")
        if porting_strategy:
            new_hash = self._hash(new_type)
            for file in tqdm(files, desc="Porting files"):
                data = self._convert_int_keys(json.loads(self.backend.GET(file)))
                ported = porting_strategy(cast(T, from_dict(data_class=cast(type, new_type), data=data, config=Config(check_types=False))))
                new_file = file.parent / f"{new_hash}.json"
                self.backend.POST(new_file, REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES[".json"](asdict(cast(Any, ported))))
                self.backend.POST(new_file.with_suffix(".schema.json"), REGISTRY_REFERENCE_TYPES_MAPPING_TO_BYTES[".json"]\
                    ({f.name: str(f.type) for f in fields(cast(type, new_type))}))
        assert allow_delete, f"Refusing to delete {len(files)} old entries. Pass allow_delete=True to confirm."
        for file in tqdm(files, desc="Deleting old entries"):
            self.backend.DELETE(file)
            self.backend.DELETE(file.with_suffix(".schema.json"), ignore_errors=True)
        print(f"{'Ported' if porting_strategy else 'Deleted'} {len(files)} entries")
        self._update_hash_lookup(_id, self._hash(new_type))


RegistryBase.DefaultTypes.RESULT_PEOPLE = make_dataclass("RESULT_PEOPLE",
    [("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA),
     ("frame_results", Dict[int, List[RegistryBase.BaseDefaultTypes.DEFAULT_DETECTION_DATACLASS]])],  # type: ignore[valid-type]
    bases=(RegistryBase.BaseDefaultTypes.BASECLASS,))
RegistryBase.DefaultTypes.RESULT_WASTE = make_dataclass("RESULT_WASTE",
    [("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA),
     ("frame_results", Dict[int, List[RegistryBase.BaseDefaultTypes.DEFAULT_WASTE_DATACLASS]])],  # type: ignore[valid-type]
    bases=(RegistryBase.BaseDefaultTypes.BASECLASS,))

if __name__ == "__main__":
    from easysort.helpers import REGISTRY_LOCAL_IP
    Registry = RegistryBase(base=REGISTRY_LOCAL_IP)

    if len(sys.argv) < 2:
        print("Usage: uv run easysort.registry sync|explore|uuid|put_folder <folder>")
        sys.exit(1)
    command = sys.argv[1]
    if command == "explore": raise NotImplementedError("Explore not implemented")
    elif command == "uuid": print(uuid4())
    elif command == "put_folder": Registry.PUT_FOLDER(Path(sys.argv[2]))
