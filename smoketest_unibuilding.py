"""Run the REAL UniBuilding checkpoint through V3 without an LLM/API call.

The VLM transport is stubbed to keep all building proposals. Image construction,
tool evidence, alignment, masks, orchestrator and worker use their real code.
Defaults reproduce the local Capture06 sample. Optional --image / --reference-mask.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import config
import drawing
import orchestrator
import planner
import vlm
from schemas import Plan
from tools import unibuilding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(Path(config.OBJECT_DETECTION_DIR) / "Google_Earth" / "Capture06.PNG"))
    parser.add_argument("--reference-mask", default="")
    parser.add_argument("--output-dir", default=str(Path(config.AGENT_DIR) / "validation" / "unibuilding_smoke"))
    args = parser.parse_args()
    image_path = Path(args.image).resolve()
    image = Image.open(image_path).convert("RGB")
    out_dir = Path(args.output_dir).resolve()
    plan = planner.apply_rules(Plan(use_unibuilding=True, output_kind="mask"), "show me the buildings as a mask")
    messages = []

    def keep_all(model, prompt, schema, **kwargs):
        messages.append(prompt)
        return vlm.VLMReply(answer="Smoke test: retain every UniBuilding proposal.",
                            keep_all_classes=["building"]), ""

    real_chat = vlm.chat_json
    vlm.chat_json = keep_all
    try:
        result = orchestrator.run_agent(image, "show me the buildings as a mask", plan=plan,
                                        run_dir=str(out_dir), notify=print)
        errors = [s.detail for s in result.steps if s.status == "error"]
        assert not errors, errors
        uni = result.unibuilding
        assert uni is not None and uni.mask.shape == (image.height, image.width)
        assert uni.probability_path and Path(uni.probability_path).is_file()
        prob = np.load(uni.probability_path, allow_pickle=False)
        assert np.array_equal(uni.mask, prob >= uni.threshold)
        assert len(result.vlm_objects) == len(uni.buildings)
        assert all(o["origin"] == "unibuilding" for o in result.vlm_objects)
        assert all(0 <= v <= 1 for b in uni.buildings for v in b["bbox"])
        selected_labels = list(uni.component_id_to_label.values())
        assert np.array_equal(result.final_mask, np.isin(uni.component_labels, selected_labels))
        assert unibuilding.worker().alive and unibuilding.worker().load_info
        assert '"unibuilding"' in messages[0][1]["content"][-1]["text"]
        assert not any(s.name.startswith(("LAE-DINO", "SegEarth")) for s in result.steps)
        reference = Path(args.reference_mask) if args.reference_mask else None
        default_ref = Path(config.UNIBUILD_ROOT) / "outputs" / "Capture06_mask.png"
        if reference is None and image_path.name.lower() == "capture06.png" and default_ref.is_file():
            reference = default_ref
        identical = None
        if reference is not None:
            expected = np.asarray(Image.open(reference)) > 0
            identical = bool(np.array_equal(uni.mask, expected))
            assert identical, "Worker mask differs from the existing UniBuild sample result"
        drawing.draw_vlm_objects(image, result.vlm_objects).save(out_dir / "building_boxes.png")
        drawing.seg_overlay(image, uni.mask.astype("uint8"), [[0, 0, 0], [255, 149, 0]]).save(
            out_dir / "building_overlay.png")
        summary = {"image": str(image_path), "size_px": list(image.size),
                   "worker": unibuilding.worker().load_info,
                   "building_pixels": int(uni.mask.sum()), "components": len(uni.buildings),
                   "coverage_percent": uni.coverage, "reference_mask_identical": identical,
                   "vlm": "transport stubbed; no API call", "plan": plan.model_dump(),
                   "steps": [s.__dict__ for s in result.steps]}
        (out_dir / "smoke_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        vlm.chat_json = real_chat
        unibuilding.worker().close()


if __name__ == "__main__":
    main()
