import unittest
from pathlib import Path
import os
import numpy as np
import pandas as pd
import shutil
from dataclasses import make_dataclass, asdict
from PIL import Image
from datetime import datetime

from easysort.helpers import current_timestamp


class TestRegistry(unittest.TestCase):

    def setUp(self):
        from easysort.registry import RegistryBase
        from easysort.helpers import REGISTRY_PATH
        test_registry_path = REGISTRY_PATH.parent / "test_registry"
        if os.path.exists(test_registry_path): shutil.rmtree(test_registry_path)
        os.makedirs(test_registry_path, exist_ok=True)
        self.registry = RegistryBase(test_registry_path)
        self.dummy_dataclass = make_dataclass("DummyDataclass", [("id", str), ("dummy", str), ("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA)])
        self.dummy_dataclass2 = make_dataclass("DummyDataclass2", [("id", str), ("dummy2", str), ("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA)])

    def tearDown(self) -> None:
        shutil.rmtree(self.registry.registry_path)

    def test_GET_default_type(self):
        data = {"test": "test"}
        self.registry.POST(Path("test.json"), data, self.registry.DefaultTypes.ORIGINAL_MARKER)
        loaded_data = self.registry.GET(Path("test.json"), self.registry.DefaultTypes.ORIGINAL_MARKER)
        assert loaded_data == data, "The loaded data should be the same as the posted data"

        result_people = self.registry.DefaultTypes.RESULT_PEOPLE(id=self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE), metadata=self.registry.BaseDefaultTypes.BASEMETADATA(model="test", created_at=current_timestamp()), \
            frame_results={0: [self.registry.BaseDefaultTypes.DEFAULT_DETECTION_DATACLASS(x1=0, y1=0, x2=100, y2=100, conf=0.9, cls="test")]})
        self.registry.POST(Path("test.json"), result_people, self.registry.DefaultTypes.RESULT_PEOPLE)
        loaded_result_people = self.registry.GET(Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE)
        assert loaded_result_people == result_people, "The loaded result people should be the same as the posted result people"

    def test_GET_custom_type(self): pass # Potentially support other, custom suffixes
    def test_POST_custom_type(self): pass # Potentially support other, custom suffixes
    
    def test_GET_bad_input_silent(self):
        self.registry.POST(Path("test.json"), {"test": "test"}, self.registry.DefaultTypes.ORIGINAL_MARKER)
        loaded_result_people = self.registry.GET(Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE, throw_error=False)
        assert loaded_result_people is None, "The loaded result people should be None"
        loaded_result_people = self.registry.GET(Path("test.json"), self.registry.DefaultTypes.RESULT_WASTE, throw_error=False)
        assert loaded_result_people is None, "The loaded result people should be None"

    def test_GET_bad_input(self): 
        self.assertRaises(AssertionError, self.registry.GET, Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE)
        self.assertRaises(AssertionError, self.registry.GET, Path("test.json"), {"test": "test"})
        self.assertRaises(AssertionError, self.registry.GET, Path("test.json"), "bad_type")

    def test_POST_GET_original_marker(self):
        data = {"test": "test"}
        self.registry.POST(Path("test.json"), data, self.registry.DefaultTypes.ORIGINAL_MARKER)
        loaded_data = self.registry.GET(Path("test.json"), self.registry.DefaultTypes.ORIGINAL_MARKER)
        assert loaded_data == data, "The loaded data should be the same as the posted data"

        dummy_image = Image.new("RGB", (100, 100), color = (255, 0, 0))
        self.registry.POST(Path("test.png"), dummy_image, self.registry.DefaultTypes.ORIGINAL_MARKER)
        loaded_image = self.registry.GET(Path("test.png"), self.registry.DefaultTypes.ORIGINAL_MARKER)
        assert np.array_equal(np.array(loaded_image), np.array(dummy_image)), "The loaded image should be the same as the posted image"

        dummy_array = np.array([1, 2, 3])
        self.registry.POST(Path("test.npy"), dummy_array, self.registry.DefaultTypes.ORIGINAL_MARKER)
        loaded_array = self.registry.GET(Path("test.npy"), self.registry.DefaultTypes.ORIGINAL_MARKER)
        assert np.array_equal(loaded_array, dummy_array), "The loaded array should be the same as the posted array"
        # Check all other supported types

    def test_LIST(self):
        self.registry.POST(Path("test.json"), {"test": "test"}, self.registry.DefaultTypes.ORIGINAL_MARKER)
        self.registry.POST(Path("test2.json"), {"test": "test"}, self.registry.DefaultTypes.ORIGINAL_MARKER)
        registry_list = self.registry.LIST()
        print(registry_list)
        assert len(registry_list) == 3, f"The list should have 3 items (2 objects + .hash_lookup.json), but has {len(registry_list)}"
        assert all(isinstance(item, Path) for item in registry_list), "The list should contain Path objects"

    def test_POST_reference_types(self):
        pass

    def test_POST_bad_reference_types(self):
        pass
        # self.assertRaises(AssertionError, self.registry.POST, Path("test.json"), {"test": "test"}, self.registry.DefaultTypes.ORIGINAL_MARKER)

    def test_POST_incorrect_reference_path(self):
        pass

    def test_GET_reference(self):
        pass

    def test_GET_incorrect_reference_path(self):
        pass

    def test_construct_path(self):
        id1 = self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE)
        self.registry.POST(Path("test.json"), {"test": "test"}, self.registry.DefaultTypes.ORIGINAL_MARKER)
        path = self.registry._construct_path(Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE)
        assert self.registry._hash_lookup[id1] in path.name, "The hash should be in the path"

    def test_no_dataclass_type(self):
        self.assertRaises(AssertionError, self.registry._construct_path, Path("test.json"), "bad_type")
        self.assertRaises(AssertionError, self.registry._construct_path, Path("test.json"), {"test": "test"})
        self.assertRaises(AssertionError, self.registry.get_id, {"test": "test"})
        self.assertRaises(AssertionError, self.registry._hash, {"test": "test"})

    def test_auto_generate_ids_for_defaulttypes(self):
        assert len(self.registry._hash_lookup) == len(self.registry.DefaultTypes.list())
        assert all(self.registry._hash(_type) in self.registry._hash_lookup.values() for _type in self.registry.DefaultTypes.list())

    def test_get_new_id(self):
        id1 = self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE)
        assert len(id1) == 36 and id1.count("-") == 4, "The id should be a uuid4"
        id2 = self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE)
        assert id1 == id2, "The two ids should be the same"
        id3 = self.registry.get_id(self.registry.DefaultTypes.RESULT_WASTE)
        assert id1 != id3, "The two ids should be different"
        id4 = self.registry.get_id(self.dummy_dataclass)
        assert id1 != id4, "The two ids should be different"
        assert len(self.registry._hash_lookup) == len(self.registry.DefaultTypes.list()) + 1
        assert id4 in self.registry._hash_lookup, "The hash lookup should have the id"
        assert self.registry._hash_lookup[id4] == self.registry._hash(self.dummy_dataclass), "The hash lookup should have the correct hash"
        self.assertRaises(AssertionError, self.registry._delete_hash, id1, self.registry._hash(self.dummy_dataclass))
        self.registry._delete_hash(id4, self.registry._hash(self.dummy_dataclass))
        assert len(self.registry._hash_lookup) == len(self.registry.DefaultTypes.list())
        assert id4 not in self.registry._hash_lookup, "The hash lookup should not have the deleted id"
        assert id1 in self.registry._hash_lookup, "The hash lookup should have the original id"
        self.assertRaises(AssertionError, self.registry._delete_hash, id4, self.registry._hash(self.dummy_dataclass2))

    def test_hashing(self):
        hash1 = self.registry._hash(self.registry.DefaultTypes.RESULT_PEOPLE)
        hash2 = self.registry._hash(self.registry.DefaultTypes.RESULT_PEOPLE)
        assert hash1 == hash2, "The two hashes should be the same"
        hash3 = self.registry._hash(self.registry.DefaultTypes.RESULT_WASTE)
        assert hash1 != hash3, "The two hashes should be different"
        assert len(hash1) == 64, "The hash should be a sha256 hash"

    def test_hash_lookup(self):
        id1 = self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE)
        hash1 = self.registry._hash(self.registry.DefaultTypes.RESULT_PEOPLE)
        assert id1 in self.registry._hash_lookup, "The hash lookup should have the id"
        assert self.registry._hash_lookup[id1] == hash1, "The hash lookup should have the correct hash"
        id2 = self.registry.get_id(self.registry.DefaultTypes.RESULT_WASTE)
        hash2 = self.registry._hash(self.registry.DefaultTypes.RESULT_WASTE)
        assert id2 in self.registry._hash_lookup, "The hash lookup should have the id"
        assert self.registry._hash_lookup[id2] == hash2, "The hash lookup should have the correct hash"
        assert id1 != id2, "The two ids should be different"
        assert hash1 != hash2, "The two hashes should be different"


if __name__ == '__main__':
  unittest.main()