"""Small, deterministic request composer for the interactive agent UI.

The composer writes a natural-language request; it does not select tools or
override the planner. Users can edit the resulting request before running it.
"""

TASKS = {
    "Object detection": {
        "description": "Find or count discrete objects such as vehicles, ships and aircraft (LAE-DINO).",
        "subject_label": "Objects to look for",
        "subject_default": "ships",
        "outputs": ("Locate", "Count", "Binary mask"),
    },
    "Land-cover segmentation": {
        "description": "Map land-cover classes such as water, roads and vegetation (SegEarth-OV3).",
        "subject_label": "Classes, separated by commas",
        "subject_default": "water, vegetation, roads",
        "outputs": ("Classification map", "Binary mask", "Coverage"),
    },
    "Buildings": {
        "description": "Find building and roof pixels or building instances (UniBuilding).",
        "subject_label": "Building subject",
        "subject_default": "buildings",
        "outputs": ("Locate", "Count", "Binary mask"),
    },
    "Agricultural fields": {
        "description": "Delineate individual field polygons using Delineate and land-cover evidence.",
        "subject_label": "Field subject",
        "subject_default": "agricultural fields",
        "outputs": ("Field polygons", "Binary mask"),
    },
    "Spatial reasoning": {
        "description": "Ask about a relationship or condition; the planner can combine tools and VLM reasoning.",
        "subject_label": "Objects or areas of interest",
        "subject_default": "buildings",
        "outputs": ("Locate", "Count", "Binary mask"),
    },
}


def compose_request(task, subject, context, output):
    """Build an editable request from the current controls, without invoking a model."""
    if task not in TASKS or output not in TASKS[task]["outputs"]:
        raise ValueError("Unsupported task or output")
    subject = subject.strip()
    context = context.strip()
    if not subject:
        raise ValueError("Enter a subject or class list first.")
    condition = " " + context.rstrip(".?!") if context else ""
    if task == "Land-cover segmentation":
        prefixes = {
            "Classification map": "Create a land-cover classification map for ",
            "Binary mask": "Create a binary mask for the land-cover classes ",
            "Coverage": "Estimate the pixel coverage of the land-cover classes ",
        }
    elif task == "Agricultural fields":
        prefixes = {
            "Field polygons": "Delineate individual agricultural field boundaries as polygons for ",
            "Binary mask": "Delineate individual agricultural field boundaries and create a binary mask for ",
        }
    else:
        prefixes = {
            "Locate": "Locate ",
            "Count": "Count ",
            "Binary mask": "Create a binary mask for ",
        }
    return prefixes[output] + subject.rstrip(".?!") + condition + "."
