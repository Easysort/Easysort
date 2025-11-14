from dataclasses import dataclass

ALLOWED_CATEGORIES = [
    "Plastics",
    "Hard plastics",
    "Tubes",
    "Cardboard",
    "Paper",
    "Folie",
    "Empty",
]

CATEGORIES_DESCRIPTION = (
    "Plastics — Mixed plastic waste. Mostly white small bags. Can look like general residual waste.\n"
    "Hard plastics — Few but clearly visible large rigid plastic objects with strong edges and clear 3D shapes (crates, bins, boxes, containers, buckets, toys, hard shells). These objects stand out individually on or next to the belt.\n"
    "Tubes — Long cylindrical objects such as pipes, hoses, cables or similar. At least one clearly visible tube/pipe dominates the highlighted region.\n"
    "Cardboard — Belt mostly covered by brown corrugated sheets/boxes and flat brown paperboard. Large 2D brown pieces dominate the view.\n"
    "Paper — Belt covered mainly by many small light/white or printed 2D paper scraps (flyers, office paper, newspapers, magazines). Looks like a colourful paper confetti layer.\n"
    "Folie — Dominated by large bags and wraps: large shiny, bags. Very white, very large, thin and flexible materials. Usually close to no colors, transparent or very light grey. Crumpled white/transparent/coloured film, often in loose heaps or big bags. Usually bales of plastic film on the side of the belt."
    "Empty — The highlighted belt area is more than 95% empty grey floor/belt, with only a few very small pieces of material.\n"
)

waste_type_belt_prompt = f"""
You are classifying what is on a conveyor belt area from a single still image.

Choose EXACTLY one category from: {ALLOWED_CATEGORIES}.

Use these definitions:
{CATEGORIES_DESCRIPTION}

The highlighted polygon and the belt inside it define the area of interest. You are allowed to look beside the belt for any materials to get hints.
Ignore machines, forklifts and walls outside this highlighted region.

Reference images will be provided after the main image. Compare the main image (first image) with the reference images to help identify the category.

Rules:
- If any material type (of any category) covers over 5% of the highlighted area,
  the image is NOT 'Empty'.
- There is always only one category present. If you think there are multiple categories present, choose the one that visually covers the largest area inside the highlighted region.
- Use the reference images to understand what each category looks like in practice.
- Be careful with the "Folie" category. Make sure it's large bags and wraps, not just smaller bags and plastics.
- The Plastics category is mostly white, non coloured and looks like general residual waste/bags, while the hard plastics category is mostly dark grey, coloured and actually looks like plastic.

Return STRICT JSON ONLY (no extra text):
{{ "category": string }}
"""

@dataclass 
class WasteTypeBeltJsonSchema:
    category: str