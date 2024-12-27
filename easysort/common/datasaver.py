
from pathlib import Path
from dataclasses import dataclass, asdict, field, fields
from typing import Union, Tuple
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

    Has 4 methods:
    - decode(msg, save: bool = false) -> Tuple[3 x str]: Decode the robot message and potentielly save to db
    - encode(status, container, reason) -> str: Encode the information into parsable message
    - is_valid_movement_message() -> bool: Checks if the message is one that could potentially be decoded and saved.
    - quit() -> None: Safely save and quit the database
    """

    def __init__(self, database_path: Union[str, Path]):
        self.database_path = str(database_path)
        self.database = self._load_db()

    def _load_db(self) -> Database: # TODO: load from online db?
        if not os.path.exists(self.database_path):
            return Database()
        data = json.load(open(self.database_path))
        return Database(**data)

    def save(self) -> None: # TODO: Save to online db?
        with open(self.database_path, 'w') as file:
            json.dump(asdict(self.database), file, indent=4)

    def decode(self, response: bytes, save: bool = False) -> Tuple[str, str, str]:
        """
        Decodes the response from robot. Has the option to save the reponse.
        Generally the format is {success/fail}__{material}__{reason}

        Responses could be:
            success__paper__none
            fail__glass__pickup_failure
        """
        splits = response.split("__")
        if len(splits) != 3: return False
        status, material, reason = splits[0], splits[1], splits[2]
        if status not in APPROVED_STATUSES or material not in APPROVED_MATERIALS: return ("", "", "")
        if save:    
            self._save_status_and_material(status, material, save=save)
            self._save_reason(reason, save=save)
        return status, material, reason
    
    def encode(self, status: str, container: str, reason: dict) -> str:
        """
        Translates Arduino response of "success"/"fail" into database representation:
        Generally the format is {success/fail}__{material}__{reason}
        """
        # Just want paper, not paper_position
        # container_str = get_matching_config_string(self.robot_config, container).replace("_position", "")
        return f"{status}__{container}__{reason}"

    def _save_status_and_material(self, status: str, material: str) -> None: 
        decoded_status = "n_fail" if status == "fail" else "n_success"
        material_attribute = getattr(self.database, material)
        current_count = getattr(material_attribute, decoded_status)
        setattr(material_attribute, decoded_status, current_count + 1)
    
    def _save_reason(self, reason: str) -> None:
        if reason not in APPROVED_FAILS_REASONS or reason == "none": return
        current_count = getattr(self.database.fails, reason)
        setattr(self.database.fails, reason, current_count + 1)

    def is_valid_movement_message(self, response: bytes) -> bool:
        if len(response.split("__")) != 3: return False
        if response.split("__")[1] not in APPROVED_STATUSES: return False
        if response.split("__")[0] not in APPROVED_MATERIALS: return False
        return True
    
    def quit(self): self.save()


