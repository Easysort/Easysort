import supabase
from dataclasses import dataclass
from easysort.utils.image_sample import DetectionSample
from easysort.common.environment import Environment

@dataclass
class BaseMetadata:
    id: str
    date: str
    robot_id: str


class SupabaseHelper:
    def __init__(self, bucket_name: str):
        self.bucket_name: str = bucket_name
        self.client = supabase.create_client(Environment.SUPABASE_URL or "", Environment.SUPABASE_KEY or "")

    def upload_sample(self, sample: DetectionSample) -> None:
        file_obj = sample.to_json().encode('utf-8')
        self.client.storage.from_(self.bucket_name).upload(
            path=f"{sample.metadata.uuid}.json",
            file=file_obj,
            file_options={
                "content-type": "application/json",
                "cache-control": "3600",
                "upsert": "false"  # False = Error if file already exists
            }
        )

    def get(self, uuid: str) -> DetectionSample:
        response = self.client.storage.from_(self.bucket_name).download(f"{uuid}.json")
        json_str = response.decode('utf-8')
        return DetectionSample.from_json(json_str)

    def delete(self, uuid: str) -> None:
        self.client.storage.from_(self.bucket_name).remove(f"{uuid}.json")