from dataclasses import dataclass

ALLOWED_CATEGORIES = [
    "Cardboard",
    "Paper",
    "Residual",
    "Plastics",
    "Empty",
]

CATEGORIES_DESCRIPTION = (
    "Cardboard — belt mostly brown corrugated sheets/boxes.\n"
    "Paper — belt full of small bright white/light 2D paper scraps.\n"
    "Residual — big soft bags/film blobs (lots of white, bits of black/blue), shiny, amorphous.\n"
    "Plastics — few large rigid objects (hard plastics, e-waste, bins), strong edges, darker colors.\n"
    "Empty — Belt is more than 95% empty. Usually looks pretty white, sometimes with brown bits on it.\n"
)

waste_type_belt_prompt = f"""
You are classifying what is on a conveyor belt from a single still image.\n
Choose EXACTLY one category from: {ALLOWED_CATEGORIES}.\n
Use these definitions:\n
{CATEGORIES_DESCRIPTION}\n
Rule: If any material covers over 5% of the belt area, it is NOT 'Empty'.\n
Rule: You need to classify the material in the highlighted region of the image.\n
Return STRICT JSON ONLY (no extra text): {{ "category": string }}
"""

@dataclass 
class WasteTypeBeltJsonSchema:
    category: str

if __name__ == "__main__":
    print(waste_type_belt_prompt)