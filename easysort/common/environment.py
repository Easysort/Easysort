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

    # Gantry:
    GANTRY_PORT: str = os.getenv("GANTRY_PORT", "")
