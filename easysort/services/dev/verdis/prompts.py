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
    "Plastics — Mixed plastic waste. Many small or medium plastic items and bags "
    "along the belt or floor, often coloured. Can look like general residual "
    "waste, but should visually be mostly plastic (packaging, bottles, trays, "
    "bags, etc.). Not dominated by only thin film, not dominated by big rigid items.\n"
    "Hard plastics — Few but clearly visible large rigid plastic objects with "
    "strong edges and clear 3D shapes (crates, bins, boxes, containers, buckets, "
    "toys, hard shells). These objects stand out individually on or next to the belt.\n"
    "Tubes — Long cylindrical objects such as pipes, hoses, cables or similar. "
    "At least one clearly visible tube/pipe dominates the highlighted region.\n"
    "Cardboard — Belt mostly covered by brown corrugated sheets/boxes and flat "
    "brown paperboard. Large 2D brown pieces dominate the view.\n"
    "Paper — Belt covered mainly by many small light/white or printed 2D paper "
    "scraps (flyers, office paper, newspapers, magazines). Looks like a colourful "
    "paper confetti layer.\n"
    "Folie — Dominated by soft plastic film, bags and wraps: large shiny, "
    "crumpled white/transparent/coloured film, often in loose heaps or big bags. "
    "Looks soft, flexible and thin rather than rigid.\n"
    "Empty — The highlighted belt area is more than 95% empty grey floor/belt, "
    "with only a few very small pieces of material.\n"
)

waste_type_belt_prompt = f"""
You are classifying what is on a conveyor belt area from a single still image.

Choose EXACTLY one category from: {ALLOWED_CATEGORIES}.

Use these definitions:
{CATEGORIES_DESCRIPTION}

The highlighted polygon and the belt inside it define the area of interest.
Ignore machines, forklifts, walls and stacked bales outside this highlighted region.

Rules:
- If any material type (of any category) covers over 5% of the highlighted area,
  the image is NOT 'Empty'.
- If several categories are present, choose the one that visually covers the
  largest area inside the highlighted region.
- If both 'Folie' (soft film/bags) and general 'Plastics' are present, choose:
    * 'Folie' if thin film/bags clearly dominate,
    * 'Plastics' if other plastic objects clearly dominate.
- Use 'Hard plastics' only when large rigid plastic objects are clearly the
  main visible material.
- Use 'Tubes' only when one or more long pipe-/hose-like items clearly dominate
  the view.

Return STRICT JSON ONLY (no extra text):
{{ "category": string }}
"""