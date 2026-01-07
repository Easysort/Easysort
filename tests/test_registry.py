import unittest
from pathlib import Path
import os
import numpy as np

from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_PATH, TESTING 

class TestRegistry(unittest.TestCase):

    def setUp(self):
        test_registry_path = REGISTRY_PATH.parent / "test_registry"
        os.makedirs(test_registry_path, exist_ok=True)
        self.registry = RegistryBase(test_registry_path)
        self.registry._hash_lookup = {}
        assert TESTING > 0, "The following tests should only be run when TESTING is set to 1 to avoid messing up real registries."

    def test_GET_default_type(self):
        pass
        # self.registry.GET(Path("test.json"), RegistryBase.DefaultTypes.RESULT_PEOPLE)

    def test_GET_custom_type(self):
        pass
        # self.registry.GET(Path("test.json"), make_dataclass("CustomType", [("name", str), ("age", int)]))

    def test_GET_bad_input1(self): 
        pass
        # self.assertRaises(AssertionError, self.registry.GET, Path("test.json"), "bad_type")

    def test_POST(self):
        self.registry.POST(Path("test.json"), np.array([1, 2, 3]), RegistryBase.DefaultTypes.ORIGINAL_MARKER)
        pass

    def test_LIST(self):
        pass

    def test_POST_reference_types(self):
        pass

    def test_POST_bad_reference_types(self):
        pass

    def test_POST_incorrect_reference_path(self):
        pass

    def test_construct_deconstruct_path(self):
        pass

    def test_get_new_id(self):
        id1 = self.registry.get_new_id(RegistryBase.DefaultTypes.RESULT_PEOPLE)
        assert len(id1) == 36 and id1.count("-") == 4, "The id should be a uuid4"
        id2 = self.registry.get_new_id(RegistryBase.DefaultTypes.RESULT_PEOPLE)
        assert id1 == id2, "The two ids should be the same"
        id3 = self.registry.get_new_id(RegistryBase.DefaultTypes.RESULT_WASTE)
        assert id1 != id3, "The two ids should be different"
        assert len(self.registry._hash_lookup) == 2, "The hash lookup should have two entries"
        self.registry._delete_hash(id1, self.registry._hash(RegistryBase.DefaultTypes.RESULT_PEOPLE))
        assert len(self.registry._hash_lookup) == 1, "The hash lookup should have one entry"
        assert id1 not in self.registry._hash_lookup, "The hash lookup should not have the deleted id"
        assert id3 in self.registry._hash_lookup, "The hash lookup should a single entry"
        self.registry._delete_hash(id3, self.registry._hash(RegistryBase.DefaultTypes.RESULT_WASTE))
        assert len(self.registry._hash_lookup) == 0, "The hash lookup should have no entries"

    def test_hashing(self):
        hash1 = self.registry._hash(RegistryBase.DefaultTypes.RESULT_PEOPLE)
        hash2 = self.registry._hash(RegistryBase.DefaultTypes.RESULT_PEOPLE)
        assert hash1 == hash2, "The two hashes should be the same"
        hash3 = self.registry._hash(RegistryBase.DefaultTypes.RESULT_WASTE)
        assert hash1 != hash3, "The two hashes should be different"
        assert len(hash1) == 64, "The hash should be a sha256 hash"

    def test_hash_lookup(self):
        id1 = self.registry.get_new_id(RegistryBase.DefaultTypes.RESULT_PEOPLE)
        hash1 = self.registry._hash(RegistryBase.DefaultTypes.RESULT_PEOPLE)
        assert id1 in self.registry._hash_lookup, "The hash lookup should have the id"
        assert self.registry._hash_lookup[id1] == hash1, "The hash lookup should have the correct hash"
        id2 = self.registry.get_new_id(RegistryBase.DefaultTypes.RESULT_WASTE)
        hash2 = self.registry._hash(RegistryBase.DefaultTypes.RESULT_WASTE)
        assert id2 in self.registry._hash_lookup, "The hash lookup should have the id"
        assert self.registry._hash_lookup[id2] == hash2, "The hash lookup should have the correct hash"
        assert id1 != id2, "The two ids should be different"
        assert hash1 != hash2, "The two hashes should be different"


if __name__ == '__main__':
  unittest.main()