"""
Semantic-segmentation worker -- runs inside the SAM conda env.

Thin shell around segearth_tool.SegEarthOV3Segmenter (your Streamlit_UI class,
copied here unchanged). The class-id mask is written as a PNG next to the input
image; everything else goes back as JSON.

run args:    {"image_path": str, "classes": ["background", "building,roof,house", ...],
              "out_path": str}
run result:  {"mask_path": str, "class_names": [str], "colors": [[r,g,b]],
              "coverage": {name: percent}, "seconds": float}
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import serve  # noqa: E402

SEGMENTER = None


def init(settings):
    global SEGMENTER
    from segearth_tool import SegEarthOV3Segmenter

    t0 = time.time()
    SEGMENTER = SegEarthOV3Segmenter(
        repo_dir=settings["segearth_root"],
        checkpoint_path=settings.get("checkpoint", ""),
        confidence_threshold=settings.get("confidence_thr", 0.1),
        prob_threshold=settings.get("prob_thr", 0.1),
        slide_crop=settings.get("slide_crop", 0),
        slide_stride=settings.get("slide_stride", 512),
        single_pass_max_side=settings.get("single_pass_max_side", 1024),
        device=settings.get("device") or None,
    )
    return {"model": "SegEarth-OV3", "device": SEGMENTER.device,
            "load_seconds": round(time.time() - t0, 1)}


def run(args):
    from PIL import Image

    image = Image.open(args["image_path"]).convert("RGB")
    t0 = time.time()
    result = SEGMENTER.segment(image, args["classes"])
    Image.fromarray(result.mask).save(args["out_path"])
    return {
        "mask_path": args["out_path"],
        "class_names": list(result.class_names),
        "colors": [[int(v) for v in c] for c in result.colors],
        "coverage": {k: round(float(v), 3) for k, v in result.coverage.items()},
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    serve(init, run)
