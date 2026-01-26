import unittest
from pathlib import Path

from easysort.registry import RegistryConnector
from tests.helpers import minikeyvalue_server


class TestRegistryConnector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._srv = minikeyvalue_server()
        cls.base = cls._srv.__enter__()
        cls.r = RegistryConnector(cls.base)

    @classmethod
    def tearDownClass(cls):
        cls._srv.__exit__(None, None, None)

    def test_minikeyvalue_api(self):
        r = self.r
        r.POST("wehave", b"bigswag")
        self.assertEqual(r.GET("wehave"), b"bigswag")
        self.assertIn(Path("wehave"), r.LIST("we"))
        with self.assertRaises(PermissionError): r.POST("wehave", b"x")
        r.UNLINK("wehave")
        self.assertIn(Path("wehave"), r.UNLINKED())
        with self.assertRaises(FileNotFoundError): r.GET("wehave")
        r.DELETE("wehave")
        self.assertNotIn(Path("wehave"), r.UNLINKED())

