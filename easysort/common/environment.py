from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Environment:
    # Generel:
    DEBUG: int = int(os.getenv("DEBUG", "0"))

    # Supabase:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_AI_IMAGES_BUCKET: str = "ai-images"
    IMAGE_REGISTRY_PATH: str = "image_registry"

    SEQUENCE_TABLE_NAME: str = "sequences"
    ORGANIZATION_TABLE_NAME: str = "organizations"
    SEQUENCE_ARTIFACTS_FOLDER: str = "sequence-artifacts"
    SUPABASE_TRACKING_BUCKET: str = "tracking"

    # Gantry:
    GANTRY_PORT: str = os.getenv("GANTRY_PORT", "")

    # Robot:
    CURRENT_ROBOT_ID: str = os.getenv("CURRENT_ROBOT_ID", "")  # 0001 = Gantry Sorting, 0101 = Delta Sorting
