
from pathlib import Path
from dataclasses import dataclass, asdict, field, fields
from typing import Union
import json
import os


@dataclass
class SortType:
    name: str
    n_success: int = 0
    n_fail: int = 0

@dataclass
class Fails:
    pickup_failure: int = 0
    lost_while_moving: int = 0
    other: int = 0

@dataclass
class Database:
    fails: Fails = field(default_factory=lambda: Fails())
    paper: SortType = field(default_factory=lambda: SortType(name = "paper"))
    cardboard: SortType = field(default_factory=lambda: SortType(name = "cardboard"))
    plastic: SortType = field(default_factory=lambda: SortType(name = "plastic"))
    metal: SortType = field(default_factory=lambda: SortType(name = "metal"))
    glass: SortType = field(default_factory=lambda: SortType(name = "glass"))
    other: SortType = field(default_factory=lambda: SortType(name = "other"))
    unknown: SortType = field(default_factory=lambda: SortType(name = "unknown"))

APPROVED_STATUSES = ["fail", "success"]
APPROVED_MATERIALS = [field.name for field in fields(Database) if field.name not in ["path", "fails"]]
APPROVED_FAILS_REASONS = [field.name for field in fields(Fails)]

class DataSaver():
    """
    Takes in a response from arduino and logs the data to database

    TODO: Add information to each element: 
    """

    def __init__(self, database_path: Union[str, Path]):
        self.database_path = str(database_path)
        self.database = self._load_db()

    def _load_db(self) -> Database:
        if not os.path.exists(self.database_path):
            return Database()
        data = json.load(open(self.database_path))
        return Database(**data)

    def save(self) -> None:
        with open(self.database_path, 'w') as file:
            json.dump(asdict(self.database), file, indent=4)

    def decode_and_save(self, response: bytes) -> bool:
        """
        Decodes the response from arduino and updates the database and returns bool

        Responses could be:
            paper__success__none
            glass__fail__pickup_failure

        Generally the format is {material}__{success/fail}__{reason}
        """
        splits = response.split("__")
        if len(splits) != 3: return False
        material, status, reason = splits[0], splits[1], splits[2]
        if status in APPROVED_STATUSES and material in APPROVED_MATERIALS:
            self._decode_status_and_material(status, material)
            self._decode_reason(reason)
            return True
        return False # TODO: Handle this

    def _decode_status_and_material(self, status: str, material: str) -> None: 
        if status == "fail":
            decoded_status = "n_fail"
        else:
            decoded_status = "n_success"
        
        material_attribute = getattr(self.database, material)
        current_count = getattr(material_attribute, decoded_status)
        setattr(material_attribute, decoded_status, current_count + 1)
    
    def _decode_reason(self, reason: str) -> None:
        if reason not in APPROVED_FAILS_REASONS or reason == "none":
            return  # reason is none if success
        
        current_count = getattr(self.database.fails, reason)
        setattr(self.database.fails, reason, current_count + 1)

    def is_valid_movement_message(self, response: bytes) -> bool:
        if len(response.split("__")) != 3: return False
        if response.split("__")[1] not in APPROVED_STATUSES: return False
        if response.split("__")[0] not in APPROVED_MATERIALS: return False
        return True
    
    def quit(self):
        self.save()


