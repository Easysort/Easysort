import unittest
from dataclasses import make_dataclass
from pathlib import Path

from easysort.registry import RegistryBase
from easysort.helpers import REGISTRY_PATH

class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = RegistryBase(REGISTRY_PATH)

    def test_GET_default_type(self):
        self.registry.GET(Path("test.json"), RegistryBase.DefaultTypes.RESULT_PEOPLE)

    def test_GET_custom_type(self):
        self.registry.GET(Path("test.json"), make_dataclass("CustomType", [("name", str), ("age", int)]))

    def test_GET_bad_input1(self): 
        self.assertRaises(AssertionError, self.registry.GET, Path("test.json"), "bad_type")
        self.assertRaises(AssertionError, self.registry.GET, Path("test.json"), dict() )
        self.assertRaises(AssertionError, self.registry.GET, Path("test.json"), list(dict()))
        self.assertRaises(AssertionError, self.registry.GET, "test.json", self.registry.DefaultTypes.RESULT_PEOPLE)

    def test_POST(self):
        pass

    def test_get_new_id(self):
        pass

    def test_hashing(self):
        pass


if __name__ == '__main__':
  unittest.main()