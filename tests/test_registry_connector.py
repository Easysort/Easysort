import unittest
from pathlib import Path

from easysort.registry import RegistryConnector
from tests.helpers import minikeyvalue_server
from easysort.helpers import ON_LOCAL_CLOUD, REGISTRY_LOCAL_IP

class TestRegistryConnector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._srv = minikeyvalue_server()
        cls.base = cls._srv.__enter__()
        cls.r = RegistryConnector(cls.base)
        cls.local_r = RegistryConnector(REGISTRY_LOCAL_IP)

    @classmethod
    def tearDownClass(cls):
        cls._srv.__exit__(None, None, None)
        if cls.local_r.EXISTS("wehave"): cls.local_r.DELETE("wehave")

    @unittest.skipUnless(ON_LOCAL_CLOUD > 0, "Skipping test: only runs when ON_LOCAL_CLOUD > 0")
    def test_on_local_cloud(self):
        self._run_connector_test(self.local_r)

    def test_minikeyvalue_api(self):
        self._run_connector_test(self.r)

    def _run_connector_test(self, r):
        r.POST("wehave", b"bigswag")
        self.assertEqual(r.GET("wehave"), b"bigswag")
        self.assertIn(Path("wehave"), r.LIST("we"))
        with self.assertRaises(PermissionError): r.POST("wehave", b"x")
        r.UNLINK("wehave")
        self.assertIn(Path("wehave"), r.UNLINKED())
        with self.assertRaises(FileNotFoundError): r.GET("wehave")
        r.DELETE("wehave")
        self.assertNotIn(Path("wehave"), r.UNLINKED())

