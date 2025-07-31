
from supabase import create_client, Client
from easysort.common.environment import Environment

from typing import Optional, Union, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import uuid

ORGANIZATION_HANDLES = {
    "AffaldPlus": "#AffaldPlus",
    "Din Organisation": "#DitID",
}

StatusType = Literal['ok', 'warning', 'danger']

@dataclass
class SequenceData:
    organization_id: str
    sequence_id: str
    delivery_company_name: str
    analysis_stov_status: StatusType
    analysis_stov_resource_name: str
    analysis_stov_images: list
    analysis_sorte_poser_status: StatusType
    analysis_sorte_poser_resource_name: str
    analysis_sorte_poser_images: list
    analysis_genbrug_status: StatusType
    analysis_genbrug_resource_name: str
    analysis_genbrug_images: list
    video_url: str
    problem_description: Optional[str] = None
    is_checked: Optional[bool] = False
    user_report: Optional[bool] = False

    def __post_init__(self):
        if self.problem_description is None:
            self.problem_description = SupabaseHelper.generate_description(self)

# class Video:
#     def __init__(self, local_path: Union[Path, str]):
#         self.data = self._load_video_data(local_path)

#     def _load_video_data(self, local_path: Union[Path, str]):
#         if isinstance(local_path, str):
#             local_path = Path(local_path)
#         if not local_path.exists():
#             raise FileNotFoundError(f"Video file {local_path} does not exist.")
#         self.data = 


class SupabaseHelper:
    def __init__(self):
        self.client: Client = create_client(Environment.SUPABASE_URL, Environment.SUPABASE_KEY)
        if not self.test_connection(): raise ConnectionError("❌ Failed to connect to Supabase")
        self.verbose = Environment.DEBUG > 0
        self.load_organizations()
    
    def load_organizations(self): 
        organizations_data = self.client.table(Environment.ORGANIZATION_TABLE_NAME).select("id, name, description").execute().data
        self.organizations_name_to_id = {org['name']: org['id'] for org in organizations_data}
        self.organizations_id_to_data = {org['id']: org for org in organizations_data}
        if self.verbose: print("Organizations loaded:", *list(self.organizations_id_to_data.items()), sep="\n")

    def test_connection(self) -> bool:
        try: return self.client.table(Environment.ORGANIZATION_TABLE_NAME).select("id").execute() is not None
        except Exception as e: print("❌ Connection failed:", e)
        return False
    
    def get_organization_id(self, org_name: str) -> Optional[str]:
        if self.verbose: print(f"Organization ID for '{org_name}' is {self.organizations_name_to_id.get(org_name, None)}")
        return self.organizations_name_to_id.get(org_name, None)

    def upload_sequence(self, data: SequenceData) -> None:
        response = self.client.table(Environment.SEQUENCE_TABLE_NAME).insert(asdict(data)).execute()
        if self.verbose and response.data: print("✅ Upload successfully:", data.sequence_id, "at", datetime.now())
        else: print(f"❌ Upload failed for {data.sequence_id}: {data.organization_id} at {datetime.now()}")
        return None
    
    def upload_sequence_artifact(self, local_path: Union[Path, str]) -> Optional[Union[Path, str]]:
        if isinstance(local_path, str): local_path = Path(local_path)
        if not local_path.exists(): raise FileNotFoundError(f"Artifact file '{local_path}' does not exist.")
        assert local_path.suffix.lower() in [".png", ".mp4"], f"Unsupported file type: {local_path.suffix}. Only .png and .mp4 are supported."
        try:
            file_content = open(local_path, "rb").read()
            mime_type = "image/png" if local_path.suffix.lower() == ".png" else "video/mp4"
            unique_filename = f"{uuid.uuid4()}{local_path.suffix.lower()}"
            self.client.storage.from_(Environment.SUPABASE_TRACKING_BUCKET).upload(str(unique_filename), file_content, file_options={"content-type": mime_type, "cache-control": "3600", "upsert": False})
            if self.verbose: print(f"✅ Uploaded artifact '{local_path}' with MIME type '{mime_type}'")
            return unique_filename
        except Exception as e:
            print(f"❌ Failed to upload artifact '{local_path}': {e}")
            return None

    @staticmethod
    def get_organization_handle(organization_name: str) -> str: return ORGANIZATION_HANDLES.get(organization_name, organization_name)

    @staticmethod
    def generate_description(data: SequenceData) -> str:
        description = []
        if data.analysis_stov_status != "ok": description.append(data.analysis_stov_resource_name)
        if data.analysis_sorte_poser_status != "ok": description.append(data.analysis_sorte_poser_resource_name)
        if data.analysis_genbrug_status != "ok": description.append(data.analysis_genbrug_resource_name)
        if len(description) > 0: return " ".join(description)
        return "Ingen problemer fundet."
        
    @staticmethod
    def default_id(organization_handle: str) -> str: return f"{organization_handle}-{datetime.now().strftime('%Y%m%d')}-AA"

    @staticmethod
    def next_id(id_num: str) -> str:
        """
        Give the next ID based on the current ID. 
        Current ID can be: AA, BA, BD, etc. The following increment is done by the last two characters:
        - AA -> AB, AB -> AC, ..., AZ -> BA
        """
        assert len(id_num) == 2 and id_num.isalpha() and id_num.isupper(), f"ID must be exactly 2 characters long, capitalized letters only: {id_num=}."
        assert id_num != "ZZ", "ID cannot be 'ZZ', as this is the last possible ID. Consider upgrading your system to support more IDs (like AAA)."
        next_last_two = chr(ord(id_num[0]) + (ord(id_num[1]) - ord('A') + 1) // 26) + chr((ord(id_num[1]) - ord('A') + 1) % 26 + ord('A'))
        return next_last_two

    def get_next_id(self, organization_handle: str) -> str:
        """
        Get the next ID for the organization. If no sequences exist, return the default ID. 
        Organization Handle either handle or name of organization.
        """
        if organization_handle[0] != "#": organization_handle = self.get_organization_handle(organization_handle)
        sequences = [str(d["sequence_id"]) for d in self.client.table(Environment.SEQUENCE_TABLE_NAME).select("sequence_id").execute().data]
        organization_sequences = sorted([seq for seq in sequences if seq.startswith(organization_handle)])
        if self.verbose: print(f"Sequences for {organization_handle}: {organization_sequences}")
        if not organization_sequences: return self.default_id(organization_handle)
        return "-".join([*organization_sequences[-1].split("-")[:-1], SupabaseHelper.next_id(organization_sequences[-1].split("-")[-1])])


if __name__ == "__main__":
    helper = SupabaseHelper()
    organization = "Din Organisation"
    stov_images = [helper.upload_sequence_artifact()]
    sorte_poser_images = [helper.upload_sequence_artifact()]
    genbrug_images = [helper.upload_sequence_artifact()]
    video_url = helper.upload_sequence_artifact()

    data = SequenceData(
        organization_id=helper.get_organization_id("Din Organisation"),
        sequence_id=helper.get_next_id(organization),
        delivery_company_name="Københavns Kommune", # "Odense Renovation", "Århus Affald"
        analysis_stov_status="completed",
        analysis_stov_resource_name="stov_analysis_001",
        analysis_stov_images=stov_images,
        analysis_sorte_poser_status="completed",
        analysis_sorte_poser_resource_name="sorte_poser_analysis_001",
        analysis_sorte_poser_images=sorte_poser_images,
        analysis_genbrug_status="completed",
        analysis_genbrug_resource_name="",
        analysis_genbrug_images=genbrug_images,
        video_url=video_url,
    )

    helper.upload_sequence(data)

        
