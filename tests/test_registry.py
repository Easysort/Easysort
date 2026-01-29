import unittest
from pathlib import Path
import numpy as np
from dataclasses import make_dataclass
from PIL import Image

from easysort.helpers import current_timestamp
from easysort.registry import RegistryBase, RegistryConnector
from tests.helpers import minikeyvalue_server


class TestRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._srv = minikeyvalue_server()
        cls._base = cls._srv.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._srv.__exit__(None, None, None)

    def setUp(self):
        # Reset state: delete all keys, re-init hash_lookup
        connector = RegistryConnector(self._base)
        for key in connector.LIST():
            try: connector.DELETE(key)
            except: pass
        connector.POST("hash_lookup.json", b"{}")
        self.registry = RegistryBase(connector)
        self.dummy_dataclass = make_dataclass("DummyDataclass", [("id", str), ("dummy", str), ("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA)])
        self.dummy_dataclass2 = make_dataclass("DummyDataclass2", [("id", str), ("dummy2", str), ("metadata", RegistryBase.BaseDefaultTypes.BASEMETADATA)])

    def test_GET_default_type(self):
        data = {"test": "test"}
        self.registry.POST(Path("test.json"), data, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        loaded_data = self.registry.GET(Path("test.json"), self.registry.DefaultMarkers.ORIGINAL_MARKER)
        assert loaded_data == data, "The loaded data should be the same as the posted data"

        result_people = self.registry.DefaultTypes.RESULT_PEOPLE(id=self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE), metadata=self.registry.BaseDefaultTypes.BASEMETADATA(model="test", created_at=current_timestamp()), \
            frame_results={0: [self.registry.BaseDefaultTypes.DEFAULT_DETECTION_DATACLASS(x1=0, y1=0, x2=100, y2=100, conf=0.9, cls="test")]})
        self.registry.POST(Path("test.json"), result_people, self.registry.DefaultTypes.RESULT_PEOPLE)
        loaded_result_people = self.registry.GET(Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE)
        assert loaded_result_people == result_people, "The loaded result people should be the same as the posted result people"

    def test_GET_custom_type(self): pass # Potentially support other, custom suffixes
    def test_POST_custom_type(self): pass # Potentially support other, custom suffixes
    
    def test_GET_bad_input_silent(self):
        self.registry.POST(Path("test.json"), {"test": "test"}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
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
        self.registry.POST(Path("test.json"), data, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        loaded_data = self.registry.GET(Path("test.json"), self.registry.DefaultMarkers.ORIGINAL_MARKER)
        assert loaded_data == data, "The loaded data should be the same as the posted data"

        dummy_image = Image.new("RGB", (100, 100), color = (255, 0, 0))
        self.registry.POST(Path("test.png"), dummy_image, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        loaded_image = self.registry.GET(Path("test.png"), self.registry.DefaultMarkers.ORIGINAL_MARKER)
        assert np.array_equal(np.array(loaded_image), np.array(dummy_image)), "The loaded image should be the same as the posted image"

        dummy_array = np.array([1, 2, 3])
        self.registry.POST(Path("test.npy"), dummy_array, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        loaded_array = self.registry.GET(Path("test.npy"), self.registry.DefaultMarkers.ORIGINAL_MARKER)
        assert np.array_equal(loaded_array, dummy_array), "The loaded array should be the same as the posted array"
        # Check all other supported types

    def test_LIST(self):
        self.registry.POST(Path("test.json"), {"test": "test"}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        self.registry.POST(Path("test2.json"), {"test": "test"}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        registry_list = self.registry.LIST()
        print(registry_list)
        assert len(registry_list) == 3, f"The list should have 3 items (2 objects + .hash_lookup.json), but has {len(registry_list)}"
        assert all(isinstance(item, Path) for item in registry_list), "The list should contain Path objects"

    def test_POST_GET_with_refs(self):
        # Setup: create original marker first
        self.registry.POST(Path("video.json"), {"src": "test"}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        
        # Post result with refs (image + json metadata)
        ref_image = Image.new("RGB", (50, 50), color=(0, 255, 0))
        ref_json = {"extra": "metadata"}
        result = self.registry.DefaultTypes.RESULT_PEOPLE(
            id=self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE),
            metadata=self.registry.BaseDefaultTypes.BASEMETADATA(model="test", created_at=current_timestamp()),
            frame_results={0: []})
        self.registry.POST(Path("video.json"), result, self.registry.DefaultTypes.RESULT_PEOPLE, 
                          refs={Path("thumb.png"): ref_image, Path("meta.json"): ref_json})
        
        # GET refs back
        loaded_image = self.registry.GET(Path("video.json"), self.registry.DefaultMarkers.REF_MARKER, ref=Path("thumb.png"))
        loaded_json = self.registry.GET(Path("video.json"), self.registry.DefaultMarkers.REF_MARKER, ref=Path("meta.json"))
        assert np.array_equal(np.array(loaded_image), np.array(ref_image))
        assert loaded_json == ref_json

    def test_GET_ref_not_found(self):
        self.registry.POST(Path("test.json"), {"x": 1}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        self.assertRaises(FileNotFoundError, self.registry.GET, Path("test.json"), self.registry.DefaultMarkers.REF_MARKER, ref=Path("nonexistent.png"))

    def test_DELETE_type(self):
        # Create original + result
        self.registry.POST(Path("test.json"), {"x": 1}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        result = self.registry.DefaultTypes.RESULT_PEOPLE(
            id=self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE),
            metadata=self.registry.BaseDefaultTypes.BASEMETADATA(model="m", created_at="t"), frame_results={})
        self.registry.POST(Path("test.json"), result, self.registry.DefaultTypes.RESULT_PEOPLE)
        
        # Delete only the result type
        self.registry.DELETE(Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE)
        assert not self.registry.EXISTS(Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE)
        assert self.registry.EXISTS(Path("test.json"), self.registry.DefaultMarkers.ORIGINAL_MARKER)

    def test_DELETE_original_deletes_all(self):
        # Create original + result + refs
        self.registry.POST(Path("vid.json"), {"x": 1}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        result = self.registry.DefaultTypes.RESULT_PEOPLE(
            id=self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE),
            metadata=self.registry.BaseDefaultTypes.BASEMETADATA(model="m", created_at="t"), frame_results={})
        self.registry.POST(Path("vid.json"), result, self.registry.DefaultTypes.RESULT_PEOPLE,
                          refs={Path("thumb.png"): Image.new("RGB", (10, 10))})
        
        # Delete original marker - should delete everything
        self.registry.DELETE(Path("vid.json"), self.registry.DefaultMarkers.ORIGINAL_MARKER)
        assert not self.registry.EXISTS(Path("vid.json"), self.registry.DefaultMarkers.ORIGINAL_MARKER)
        self.assertRaises(AssertionError, self.registry.EXISTS, Path("vid.json"), self.registry.DefaultTypes.RESULT_PEOPLE) # Should fail as original marker is deleted
        assert not self.registry.backend.EXISTS(self.registry._get_ref_path(Path("vid.json"), Path("thumb.png")))

    def test_construct_path(self):
        id1 = self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE)
        self.registry.POST(Path("test.json"), {"test": "test"}, self.registry.DefaultMarkers.ORIGINAL_MARKER)
        path = self.registry._construct_path(Path("test.json"), self.registry.DefaultTypes.RESULT_PEOPLE)
        assert self.registry._get_hash_lookup()[id1] in path.name, "The hash should be in the path"

    def test_no_dataclass_type(self):
        self.assertRaises(AssertionError, self.registry._construct_path, Path("test.json"), "bad_type")
        self.assertRaises(AssertionError, self.registry._construct_path, Path("test.json"), {"test": "test"})
        self.assertRaises(AssertionError, self.registry.get_id, {"test": "test"})
        self.assertRaises(AssertionError, self.registry._hash, {"test": "test"})

    def test_auto_generate_ids_for_defaulttypes(self):
        hl = self.registry._get_hash_lookup()
        assert len(hl) == len(self.registry.DefaultTypes.list()) + len(self.registry.DefaultMarkers.list())
        assert all(self.registry._hash(_type) in hl.values() for _type in self.registry.DefaultTypes.list())
        assert all(self.registry._hash(_type) in hl.values() for _type in self.registry.DefaultMarkers.list())

    def test_get_new_id(self):
        id1 = self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE)
        assert len(id1) == 36 and id1.count("-") == 4, "The id should be a uuid4"
        id2 = self.registry.get_id(self.registry.DefaultTypes.RESULT_PEOPLE)
        assert id1 == id2, "The two ids should be the same"
        id3 = self.registry.get_id(self.registry.DefaultTypes.RESULT_WASTE)
        assert id1 != id3, "The two ids should be different"
        id4 = self.registry.get_id(self.dummy_dataclass)
        assert id1 != id4, "The two ids should be different"
        hl = self.registry._get_hash_lookup()
        assert len(hl) == len(self.registry.DefaultTypes.list()) + len(self.registry.DefaultMarkers.list()) + 1
        assert id4 in hl, "The hash lookup should have the id"
        assert hl[id4] == self.registry._hash(self.dummy_dataclass), "The hash lookup should have the correct hash"
        self.assertRaises(AssertionError, self.registry._delete_hash, id1, self.registry._hash(self.dummy_dataclass))
        self.registry._delete_hash(id4, self.registry._hash(self.dummy_dataclass))
        hl = self.registry._get_hash_lookup()
        assert len(hl) == len(self.registry.DefaultTypes.list()) + len(self.registry.DefaultMarkers.list())
        assert id4 not in hl, "The hash lookup should not have the deleted id"
        assert id1 in hl, "The hash lookup should have the original id"
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
        hl = self.registry._get_hash_lookup()
        assert id1 in hl, "The hash lookup should have the id"
        assert hl[id1] == hash1, "The hash lookup should have the correct hash"
        id2 = self.registry.get_id(self.registry.DefaultTypes.RESULT_WASTE)
        hash2 = self.registry._hash(self.registry.DefaultTypes.RESULT_WASTE)
        hl = self.registry._get_hash_lookup()
        assert id2 in hl, "The hash lookup should have the id"
        assert hl[id2] == hash2, "The hash lookup should have the correct hash"
        assert id1 != id2, "The two ids should be different"
        assert hash1 != hash2, "The two hashes should be different"


if __name__ == '__main__':
  unittest.main()