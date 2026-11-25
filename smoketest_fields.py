"""Real Delineate checkpoint smoke; optionally real SegEarth and live VLM.

Default uses controlled cropland semantics and a VLM transport stub, explicitly
recorded in smoke_result.json. --real-segearth exercises the learned LULC model.
--live-vlm also calls the configured endpoint. No smoke is an accuracy benchmark.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.windows import Window

import config
import drawing
import field_geometry as geo
import orchestrator
import planner
import vlm
from schemas import SegmentationOutput
from tools import delineate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(Path(config.DELINEATE_UPSTREAM)/"data/images/Sample/part_1.tif"))
    parser.add_argument("--output", default="validation/field_smoke")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--real-segearth", action="store_true")
    parser.add_argument("--live-vlm", action="store_true")
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(args.image).resolve()
    if source.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(source) as src:
            w, h = min(src.width,args.crop_size), min(src.height,args.crop_size)
            window = Window((src.width-w)//2,(src.height-h)//2,w,h)
            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(width=w,height=h,transform=src.window_transform(window))
            crop = folder/"source_crop.tif"
            with rasterio.open(crop,"w",**profile) as dst:
                dst.write(data)
        source = crop
    image = geo.read_image(source)
    question = "Delineate all agricultural fields and give a binary mask. Distinguish uncertain candidates."
    plan = planner.apply_rules(planner.Plan(output_kind="mask"), question)

    def semantic(*a, **k):
        classes = ["background"] + list(config.SEMANTIC_CLASSES.values())
        names = [entry.split(",")[0] for entry in classes]
        mask = np.full((image.height,image.width), names.index("cropland"),np.uint8)
        return SegmentationOutput(classes,names,
            [[i*19%256,i*79%256,i*41%256] for i in range(len(names))],
            {n:100.0 if n=="cropland" else 0.0 for n in names},mask,"")

    def stub(model, messages, schema, **kwargs):
        if schema is vlm.FieldReviewPolicy:
            return vlm.FieldReviewPolicy(), {}
        if schema is vlm.FieldBatchReply:
            payload=json.loads(messages[1]["content"][-1]["text"])
            return vlm.FieldBatchReply(judgments=[vlm.FieldJudgment(id=f["id"],decision="keep")
                for f in payload["fields"]]), {}
        if schema is vlm.FieldSummaryReply:
            return vlm.FieldSummaryReply(answer="Smoke: reviewed eligible polygons; not an accuracy evaluation."), {}
        return vlm.VLMReply(answer="Smoke test: retained eligible candidate polygons; "
            "not an accuracy evaluation.",keep_all_classes=["agricultural field"]), {}

    from contextlib import ExitStack
    with ExitStack() as stack:
        if not args.real_segearth:
            stack.enter_context(patch.object(orchestrator.segmentation,"segment",side_effect=semantic))
        if not args.live_vlm:
            stack.enter_context(patch.object(vlm,"chat_json",side_effect=stub))
        result = orchestrator.run_agent(image,question,plan=plan,run_dir=str(folder),
            source_path=str(source),notify=lambda s: print(s,flush=True))
    errors = [s.__dict__ for s in result.steps if s.status=="error"]
    if result.fields is not None:
        drawing.draw_fields(image,result.fields.raw_fields).save(folder/"native_overlay.png")
        drawing.draw_fields(image,result.fields.fields).save(folder/"semantic_overlay.png")
        drawing.draw_vlm_objects(image,result.vlm_objects).save(folder/"selected_overlay.png")
        drawing.seg_color_map(result.segmentation.mask,result.segmentation.colors).save(folder/"classification.png")
    info = {"image":str(source),"size_px":image.size,"real_delineate":True,
            "segearth":"real model" if args.real_segearth else "controlled cropland fixture",
            "vlm":"live endpoint" if args.live_vlm else "transport stub",
            "native_candidates":len(result.fields.raw_fields) if result.fields else None,
            "eligible":len(result.fields.fields) if result.fields else None,
            "selected":len(result.vlm_objects), "errors":errors,
            "worker":delineate.worker().load_info,"plan":plan.model_dump()}
    (folder/"smoke_result.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2),flush=True)
    delineate.worker().close()
    if orchestrator.segmentation._worker:
        orchestrator.segmentation._worker.close()
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
