"""
Semantic-segmentation tool (SegEarth-OV3), as the agent sees it.

    segment(image_path, classes, run_dir) -> SegmentationOutput

The model runs in the SAM env via workers/segearth_worker.py. On top of the raw
class map this module adds one cheap, deterministic product the VLM cannot make
itself: the map split into individual areas (connected components) with
normalized boxes and pixel areas -- buildings (b0, b1, ...) and every other
class (r0, r1, ...: each parking lot, pond, field). The VLM selects from these,
so "show the parking lots" rests on measured pixels. Legacy building components
remain available for general LULC maps; explicit buildings use tools/unibuilding.py.
"""

import os

import numpy as np
from PIL import Image

import config
from schemas import SegmentationOutput
from tools.worker_client import ToolWorker

_worker = None

def worker() -> ToolWorker:
    global _worker
    if _worker is None:
        _worker = ToolWorker(
            name="SegEarth-OV3",
            python_exe=config.SAM_PYTHON,
            script=os.path.join(config.WORKERS_DIR, "segearth_worker.py"),
            settings={
                "segearth_root": config.SEGEARTH_ROOT,
                "checkpoint": config.SAM3_CHECKPOINT,
                "confidence_thr": config.SEG_CONFIDENCE_THR,
                "prob_thr": config.SEG_PROB_THR,
                "slide_crop": config.SEG_SLIDE_CROP,
                "slide_stride": config.SEG_SLIDE_STRIDE,
                "single_pass_max_side": config.SEG_SINGLE_PASS_MAX_SIDE,
                "device": config.SEG_DEVICE,
            },
            log_dir=config.LOGS_DIR,
        )
    return _worker


def is_building_class(name: str) -> bool:
    import re
    return bool(re.search(r"\b(buildings?|houses?|homes?|roofs?|rooftops?|residential)\b"
                          r"|ساختمان|خانه|منزل|بام|سقف", name, re.IGNORECASE))


def segment(image_path, classes, run_dir, out_name="seg_mask.png") -> SegmentationOutput:
    """`out_name` lets the mask step's extra pass write its own file instead of
    overwriting the main segmentation map."""
    if not classes:
        raise ValueError("SegEarth requires explicit resolved classes; only the planner can authorize a default scheme")
    if classes[0] != "background" or len(classes) > 256:
        raise ValueError("SegEarth needs background at ID 0 and at most 256 class IDs")
    if len(classes) - 1 > config.SEG_MAX_CLASSES:
        raise ValueError("SegEarth vocabulary exceeds SEG_MAX_CLASSES; refusing to truncate")
    out = worker().call({
        "image_path": image_path,
        "classes": classes,
        "out_path": os.path.join(run_dir, out_name),
    })
    mask = np.array(Image.open(out["mask_path"]))
    expected = [entry.split(",")[0].strip() for entry in classes]
    if out["class_names"] != expected:
        raise ValueError("SegEarth output class-ID mapping differs from the requested vocabulary")
    with Image.open(image_path) as image:
        if mask.shape != (image.height, image.width):
            raise ValueError("SegEarth class-ID raster does not match the original image grid")
    if mask.ndim != 2 or mask.max() >= len(expected):
        raise ValueError("SegEarth returned invalid class IDs")
    building_idx = next((i for i, c in enumerate(classes) if i > 0 and is_building_class(c)), None)
    buildings, regions, labels, id_to_label = split_into_regions(
        mask, out["class_names"], building_idx)

    return SegmentationOutput(
        classes=list(classes),
        class_names=out["class_names"],
        colors=out["colors"],
        coverage=out["coverage"],
        mask=mask,
        mask_path=out["mask_path"],
        buildings=buildings,
        regions=regions,
        building_class=out["class_names"][building_idx] if building_idx is not None else None,
        component_labels=labels,
        component_id_to_label=id_to_label,
        seconds=out["seconds"],
    )


def split_into_regions(mask: np.ndarray, class_names, building_idx):
    """
    Connected components of EVERY segmented class, so the VLM can point at
    individual areas, not just buildings:
      buildings  ids b0, b1, ... (building class, >= MIN_BUILDING_AREA_PX)
      regions    ids r0, r1, ... (every other class: each parking lot, pond,
                 field ...; >= MIN_REGION_AREA_FRAC of the image), with "class"
    One int32 label image holds all of them (classes never overlap in the
    argmax map), and id_to_label maps each id to its label value there.
    """
    labels_all = np.zeros(mask.shape, np.int32)
    id_to_label, buildings, regions = {}, [], []
    offset, region_no = 0, 0
    min_region = max(config.MIN_BUILDING_AREA_PX, int(config.MIN_REGION_AREA_FRAC * mask.size))
    for idx in range(1, len(class_names)):
        binary = mask == idx
        if not binary.any():
            continue
        is_building = idx == building_idx
        comps, labels, ids = label_components(
            binary, config.MIN_BUILDING_AREA_PX if is_building else min_region)
        if labels is None or not comps:
            continue
        labels_all[binary] = labels[binary] + offset
        for comp in comps:
            raw_label = ids[comp["id"]]
            if is_building:
                new_id = comp["id"]                       # b0, b1, ...
                buildings.append(comp)
            else:
                new_id = "r%d" % region_no
                region_no += 1
                comp = dict(comp, id=new_id, **{"class": class_names[idx]})
                regions.append(comp)
            id_to_label[new_id] = raw_label + offset
        offset += int(labels.max())
    return buildings, regions, labels_all, id_to_label


def label_components(binary: np.ndarray, min_area: int, connectivity: int = 4):
    """
    Connected components of a binary mask.
    Returns (components, label_image, id_to_label):
      components   [{"id": "b0", "bbox": normalized, "area_px": int}], largest first
      label_image  int array, 0 = background, k = k-th raw component
      id_to_label  {"b0": k, ...} -- to rebuild the pixels of any component
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    h, w = binary.shape
    raw = []  # (label, x1, y1, x2, y2, area) in pixels
    try:
        from scipy import ndimage
        labels, _ = ndimage.label(binary, structure=ndimage.generate_binary_structure(
            2, 1 if connectivity == 4 else 2))
        areas = np.bincount(labels.ravel())
        for label, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is not None:
                raw.append((label, sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, int(areas[label])))
    except ImportError:
        try:
            import cv2
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=connectivity)
            for label, (x, y, bw, bh, area) in enumerate(stats[1:], start=1):  # row 0 = background
                raw.append((label, int(x), int(y), int(x + bw), int(y + bh), int(area)))
        except ImportError:
            return [], None, {}   # neither scipy nor opencv installed: no instance stats

    raw = sorted((r for r in raw if r[5] >= min_area), key=lambda r: r[5], reverse=True)
    comps, id_to_label = [], {}
    for i, (label, x1, y1, x2, y2, area) in enumerate(raw):
        comp_id = "b%d" % i
        comps.append({"id": comp_id, "bbox": [round(x1 / w, 4), round(y1 / h, 4),
                                               round(x2 / w, 4), round(y2 / h, 4)],
                      "area_px": area})
        id_to_label[comp_id] = label
    return comps, labels, id_to_label


def connected_components(binary: np.ndarray, min_area: int):
    """Just the component list (see label_components)."""
    return label_components(binary, min_area)[0]


def component_mask(seg: SegmentationOutput, ids) -> np.ndarray:
    """Binary mask (H, W) with only the given components (b* or r*), e.g. the ones
    the VLM selected -- a map built from the TOOL's pixels, chosen by the VLM."""
    if seg.component_labels is None:
        return np.zeros(seg.mask.shape, bool)
    wanted = [seg.component_id_to_label[i] for i in ids if i in seg.component_id_to_label]
    return np.isin(seg.component_labels, wanted)
