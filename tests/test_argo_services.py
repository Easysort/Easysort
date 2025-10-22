import unittest


class TestArgoServices(unittest.TestCase):
    def test_downloader(self):
        from easysort.services.argo.helper import Downloader, Locations
        downloader = Downloader()
        assert downloader.location == Locations.EASYSORT128
        assert downloader.date is not None

    def test_task_manager(self):
        pass 

    def test_supabase_locations(self):
        from easysort.services.argo.helper import SupabaseLocations
        assert SupabaseLocations.Argo.Roskilde01 == "ARGO-Roskilde-Entrance-01"
        assert isinstance(SupabaseLocations.Argo.ids, dict) and len(SupabaseLocations.Argo.ids.keys()) >= 1
        assert SupabaseLocations.Argo.bucket == "argo"

