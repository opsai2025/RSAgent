"""SegEarth evidence -> aligned LCLU -> native Delineate -> semantic verification."""

import json
import os
import time

from PIL import Image

import config
import field_geometry as geometry
from schemas import FieldOutput
from tools.worker_client import ToolWorker

_worker = None


def worker():
    global _worker
    if _worker is None:
        _worker = ToolWorker("Delineate-V2", config.DELINEATE_PYTHON,
            os.path.join(config.WORKERS_DIR, "delineate_worker.py"),
            {"upstream": config.DELINEATE_UPSTREAM, "checkpoint": config.DELINEATE_CHECKPOINT,
             "cache_dir": os.path.join(config.LOGS_DIR, "ultralytics"),
             "confidence": config.DELINEATE_CONFIDENCE, "min_area_px": config.DELINEATE_MIN_AREA_PX,
             "batch_size": config.DELINEATE_BATCH_SIZE, "post_workers": config.DELINEATE_POST_WORKERS},
            config.LOGS_DIR, timeout=config.DELINEATE_TIMEOUT)
    return _worker


def delineate(image_path, seg, run_dir, source_path=None):
    if seg is None:
        raise ValueError("Delineate requires successful SegEarth semantic evidence")
    if not 0 < config.DELINEATE_REMAINING_AREA <= 1:
        raise ValueError("DELINEATE_REMAINING_AREA must be in (0,1]")
    if not (0 < config.FIELD_NONAG_REJECT_FRACTION <= 1 and
            0 < config.FIELD_CROPLAND_SUPPORT_FRACTION <= 1):
        raise ValueError("Field semantic thresholds must be in (0,1]")
    if config.DELINEATE_BATCH_SIZE < 1 or config.DELINEATE_POST_WORKERS < 1:
        raise ValueError("Delineate batch size and post workers must be positive")
    t0 = time.time()
    with Image.open(image_path) as image:
        meta = geometry.source_metadata(image.size, source_path)
    cfg = geometry.native_filter_config(seg)
    lclu_path = geometry.write_lclu(seg, os.path.join(run_dir, "segearth_lclu.tif"), meta)
    out = worker().call({"image_path": image_path, "lclu_path": lclu_path,
                         "run_dir": run_dir, "filter_config": cfg})
    with open(out["fields_path"], encoding="utf-8") as fh:
        raw = json.load(fh)
    ids = [f["id"] for f in raw]
    if len(ids) != len(set(ids)) or any(not i.startswith("f") or not i[1:].isdigit() for i in ids):
        raise ValueError("Delineate returned invalid or duplicate field IDs")
    checked, kept, rejected = geometry.verify_fields(raw, seg)
    result = FieldOutput(checked, kept, rejected, lclu_path, out["native_gpkg_path"],
                         metadata=meta, filter_config=cfg, seconds=round(time.time()-t0, 2))
    export(result, [], run_dir)
    return result


def selected_fields(objects, fields):
    by_id = {f["id"]: f for f in fields.fields}
    selected, seen = [], set()
    for obj in objects:
        pid = obj.get("source_id")
        if obj.get("origin") == "delineate" and pid in by_id and pid not in seen:
            seen.add(pid)
            selected.append(dict(by_id[pid], selection_status=obj.get("selection_status", "uncertain"),
                                 field_classification=obj.get("field_classification", "")))
    return selected


def export(fields, objects, run_dir):
    selected = selected_fields(objects, fields)
    payload = {"metadata": fields.metadata, "layers": {
        "native_candidates": fields.raw_fields, "semantic_candidates": fields.fields,
        "semantic_rejected": fields.rejected, "vlm_selected": selected}}
    payload_path = os.path.join(run_dir, "field_proposals.json")
    with open(payload_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    fields.gpkg_path = worker().call({"action": "export", "payload_path": payload_path,
                                      "output_path": os.path.join(run_dir, "fields.gpkg")})["gpkg_path"]
    size = (fields.metadata["width"], fields.metadata["height"])
    for name, proposals in (("field_candidates", fields.fields), ("fields_selected", selected)):
        instances = geometry.field_mask(proposals, size, instance=True)
        geometry.save_raster(instances, os.path.join(run_dir, name + "_instances.tif"), fields.metadata)
        geometry.save_raster((instances > 0).astype("uint8"), os.path.join(run_dir, name + "_binary.tif"),
                             fields.metadata)
    return fields.gpkg_path
