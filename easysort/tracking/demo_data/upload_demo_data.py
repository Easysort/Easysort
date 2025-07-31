from easysort.tracking.supabase_helper import SupabaseHelper, SequenceData

from typing import List

if __name__ == "__main__":
    helper = SupabaseHelper()
    organization = "Din Organisation"

    stov_images: List[str] = []
    sorte_poser_images: List[str] = []
    genbrug_images = [helper.upload_sequence_artifact("easysort/tracking/demo_data/1/Genbrug1.png")]
    video_url = helper.upload_sequence_artifact("easysort/tracking/demo_data/1/DemoData2.mp4")

    data = SequenceData(
        organization_id=helper.get_organization_id("Din Organisation"),
        sequence_id=helper.get_next_id(organization),
        delivery_company_name="Københavns Kommune",  # "Odense Renovation", "Århus Affald"
        analysis_stov_status="ok",
        analysis_stov_resource_name="Ingen betydelige mængde støv fundet.",
        analysis_stov_images=stov_images,
        analysis_sorte_poser_status="ok",
        analysis_sorte_poser_resource_name="Ingen sorte poser ved denne aflæsning.",
        analysis_sorte_poser_images=sorte_poser_images,
        analysis_genbrug_status="warning",
        analysis_genbrug_resource_name="Nogle genbrugelige materialer fundet.",
        analysis_genbrug_images=genbrug_images,
        video_url=video_url,
    )

    helper.upload_sequence(data)

    stov_images = []
    sorte_poser_images = []
    genbrug_images = [
        helper.upload_sequence_artifact("easysort/tracking/demo_data/2/Genbrug1.png"),
        helper.upload_sequence_artifact("easysort/tracking/demo_data/2/Genbrug2.png"),
    ]
    video_url = helper.upload_sequence_artifact("easysort/tracking/demo_data/2/DemoData1.mp4")

    data = SequenceData(
        organization_id=helper.get_organization_id("Din Organisation"),
        sequence_id=helper.get_next_id(organization),
        delivery_company_name="Københavns Kommune",  # "Odense Renovation", "Århus Affald"
        analysis_stov_status="ok",
        analysis_stov_resource_name="Ingen betydelige mængde støv fundet.",
        analysis_stov_images=stov_images,
        analysis_sorte_poser_status="ok",
        analysis_sorte_poser_resource_name="Ingen sorte poser ved denne aflæsning.",
        analysis_sorte_poser_images=sorte_poser_images,
        analysis_genbrug_status="danger",
        analysis_genbrug_resource_name="Betydelige mængder af genbrugelige materialer fundet.",
        analysis_genbrug_images=genbrug_images,
        video_url=video_url,
    )

    helper.upload_sequence(data)

    stov_images = []
    sorte_poser_images = []
    genbrug_images = [helper.upload_sequence_artifact("easysort/tracking/demo_data/3/Genbrug1.png")]
    video_url = helper.upload_sequence_artifact("easysort/tracking/demo_data/3/DemoData3.mp4")

    data = SequenceData(
        organization_id=helper.get_organization_id("Din Organisation"),
        sequence_id=helper.get_next_id(organization),
        delivery_company_name="Odense Renovation",  # "Odense Renovation", "Århus Affald"
        analysis_stov_status="ok",
        analysis_stov_resource_name="Ingen betydelige mængde støv fundet.",
        analysis_stov_images=stov_images,
        analysis_sorte_poser_status="ok",
        analysis_sorte_poser_resource_name="Ingen sorte poser ved denne aflæsning.",
        analysis_sorte_poser_images=sorte_poser_images,
        analysis_genbrug_status="ok",
        analysis_genbrug_resource_name="Meget få genbrugelige materialer fundet.",
        analysis_genbrug_images=genbrug_images,
        video_url=video_url,
    )

    helper.upload_sequence(data)

    stov_images = [
        helper.upload_sequence_artifact("easysort/tracking/demo_data/4/dust.png"),
        helper.upload_sequence_artifact("easysort/tracking/demo_data/4/dust2.png"),
    ]
    sorte_poser_images = []
    genbrug_images = []
    video_url = helper.upload_sequence_artifact("easysort/tracking/demo_data/4/DemoData4.mp4")

    data = SequenceData(
        organization_id=helper.get_organization_id("Din Organisation"),
        sequence_id=helper.get_next_id(organization),
        delivery_company_name="Århus Affald",  # "Odense Renovation", "Århus Affald"
        analysis_stov_status="danger",
        analysis_stov_resource_name="Klar overskridelse af støvmængde.",
        analysis_stov_images=stov_images,
        analysis_sorte_poser_status="ok",
        analysis_sorte_poser_resource_name="Ingen sorte poser ved denne aflæsning.",
        analysis_sorte_poser_images=sorte_poser_images,
        analysis_genbrug_status="warning",
        analysis_genbrug_resource_name="Få genbrugelige materialer fundet.",
        analysis_genbrug_images=genbrug_images,
        video_url=video_url,
    )

    helper.upload_sequence(data)
