from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Environment:
    # Supabase:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL") or ""
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY") or ""
    SUPABASE_AI_IMAGES_BUCKET: str = "ai-images"

    # Gantry:
    GANTRY_PORT: str = os.getenv("GANTRY_PORT") or ""
