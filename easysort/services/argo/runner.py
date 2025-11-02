from google import genai
from easysort.common.environment import Env
import os

print(os.getcwd())

client = genai.Client(api_key = Env.GOOGLE_API_KEY)

image_path = "easysort/services/argo/stats/photo1.jpg"
image = open(image_path, "rb").read()

response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Is there a person (more than 50%) in this image. Answer only Yes/No",
                  genai.types.Part.from_bytes(mime_type = "image/jpeg", data = image)]
        )

print(response.text)

image_path = "easysort/services/argo/stats/photo2.jpg"
image = open(image_path, "rb").read()

response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Return a JSON with the following keys: 'number_of_people' (int), 'description_of_people' (list[str]) and 'list_of_items_people_are_carrying' (list[str]).",
                  genai.types.Part.from_bytes(mime_type = "image/jpeg", data = image)]
        )

print(response.text)
