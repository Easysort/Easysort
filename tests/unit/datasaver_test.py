import sys

sys.path.append("/Users/lucasvilsen/Documents/Documents/EasySort")

import tempfile
from dataclasses import asdict
import os

from src.helpers.datasaver import DataSaver


class TestDataSaver:
    def test_decode_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = os.path.join(temp_dir, "database.json")
            datasaver = DataSaver(database_path)
            print(datasaver.database)
            assert datasaver.decode("paper__success__none")
            assert datasaver.decode("plastic__fail__pickup_failure")
            assert not datasaver.decode("unknown__fail")
            datasaver.save()
            assert os.path.exists(database_path)
            print(asdict(datasaver.database))


if __name__ == "__main__":
    TestDataSaver().test_decode_decode()
