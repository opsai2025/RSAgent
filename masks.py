"""
The mask OUTPUT -- one of the two ways a result is shown (plan.output_kind:
"boxes" or "mask", chosen by the router). It is built from the tools' pixels for
exactly what was decided; building it never changes the plan.

    from_decision(objects, target_classes, seg, size, object_seg)   VLM path
    from_plain_map(seg)                                             plain map path
    object_seg_classes(objects, seg)    vocabulary for the extra segmentation pass
                                        the orchestrator runs AFTER the VLM, only
                                        for a mask, only for the kept objects

For each object the VLM kept:
  UniBuilding b* (origin "unibuilding")       -> its building pixels, clipped
      to a refined/split box if the VLM supplied one; no SegEarth pass
  segmentation area b*/r* (origin "segmentation") -> the area's own pixels
  detection (origin "detector")               -> pixels of the matching
      segmentation class INSIDE the detection box (e.g. "car" pixels inside the
      box of a yellow car), taken from the mask-only pass (`object_seg`) or the
      main segmentation; a filled box if neither found enough there
  object added by the VLM (origin "vlm")      -> a filled box
Plus every land-cover class the VLM listed in `target_classes` (e.g. "bare soil"
for "a mask of the empty land").
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from schemas import SegmentationOutput, UniBuildingOutput
from tools.segmentation import component_mask

MIN_FILL = 0.10   # segmentation must cover >= 10 % of a box to be used as its mask

# detector class -> segmentation vocabulary entry used to cut that object's mask.
# Only compact objects SAM 3 segments well; courts / fields keep a box mask.
MASK_SEG_CLASS: Dict[str, str] = {
    "small vehicle": "car,vehicle", "large vehicle": "truck,bus,lorry",
    "ship": "ship,boat", "plane": "airplane,plane,aircraft", "helicopter": "helicopter",
    "storage tank": "storage tank,tank", "swimming pool": "swimming pool,pool",
}


def _px(box, w, h):
    x1, y1, x2, y2 = box
    return (int(x1 * w), int(y1 * h), max(int(x1 * w) + 1, int(round(x2 * w))),
            max(int(y1 * h) + 1, int(round(y2 * h))))


def _class_index(seg: SegmentationOutput, name: str) -> Optional[int]:
    name = name.strip().lower()
    for i, entry in enumerate(seg.classes):
        if i == 0:
            continue
        synonyms = [s.strip().lower() for s in entry.split(",")]
        if name == entry.lower() or name in synonyms:
            return i
    return None


def object_seg_classes(objects: List[dict], seg: Optional[SegmentationOutput]) -> List[str]:
    """Segmentation vocabulary still needed to cut the kept DETECTIONS from real
    pixels: their classes' MASK_SEG_CLASS entries, minus any the main
    segmentation already has. Empty -> no extra pass needed."""
    needed: List[str] = []
    for obj in objects:
        if obj.get("origin") != "detector":
            continue
        entry = MASK_SEG_CLASS.get(obj.get("class", ""))
        if not entry or entry in needed:
            continue
        if seg is not None and _class_index(seg, entry) is not None:
            continue
        needed.append(entry)
    return needed


def from_decision(objects: List[dict], target_classes: List[str],
                  seg: Optional[SegmentationOutput], size: Tuple[int, int],
                  object_seg: Optional[SegmentationOutput] = None,
                  unibuilding: Optional[UniBuildingOutput] = None):
    w, h = size
    out = np.zeros((h, w), bool)
    used = {"pixels": 0, "boxes": 0, "components": 0, "buildings": 0, "classes": []}

    # where a detection's pixels can come from: the mask-only pass first, then
    # the main segmentation (if it happened to segment the same objects)
    sources = [s for s in (object_seg, seg) if s is not None]

    comp_ids = [o["source_id"] for o in objects if o["origin"] == "segmentation"]
    if seg is not None and comp_ids:
        out |= component_mask(seg, comp_ids)
        used["components"] = len(comp_ids)

    for obj in objects:
        if obj["origin"] == "segmentation":
            continue
        if obj["origin"] == "delineate":
            from field_geometry import polygon_pixels
            window, pixels = polygon_pixels(obj["geometry"], size)
            out[window] |= pixels
            used.setdefault("fields", 0)
            used["fields"] += 1
            continue
        box = _px(obj["box"], w, h)
        if obj["origin"] == "unibuilding" and unibuilding is not None:
            label = unibuilding.component_id_to_label.get(obj.get("source_id"))
            if label is not None:
                x1, y1, x2, y2 = box
                out[y1:y2, x1:x2] |= unibuilding.component_labels[y1:y2, x1:x2] == label
                used["buildings"] += 1
                continue
        if obj["origin"] == "detector" and _cut_from_pixels(obj.get("class", ""), sources, out, box):
            used["pixels"] += 1
            continue
        x1, y1, x2, y2 = box
        out[y1:y2, x1:x2] = True
        used["boxes"] += 1

    if unibuilding is not None and "building" in target_classes:
        out |= unibuilding.mask
        used["classes"].append("building (UniBuilding)")
    if seg is not None:
        for name in target_classes:
            if name == "building" and unibuilding is not None:
                continue
            idx = _class_index(seg, name)
            if idx is not None:
                out |= seg.mask == idx
                used["classes"].append(seg.class_names[idx])

    parts = []
    if used.get("fields"):
        parts.append("%d original field polygon(s), including holes" % used["fields"])
    if used["buildings"]:
        parts.append("%d building(s) from UniBuilding mask pixels" % used["buildings"])
    if used["components"]:
        parts.append("%d area(s) from the segmentation mask" % used["components"])
    if used["pixels"]:
        parts.append("%d object(s) cut from segmentation pixels inside their box" % used["pixels"])
    if used["boxes"]:
        parts.append("%d object(s) as filled boxes" % used["boxes"])
    if used["classes"]:
        parts.append("land-cover class(es): " + ", ".join(used["classes"]))
    return out, "; ".join(parts) or "empty -- nothing selected"


def _cut_from_pixels(cls: str, sources, out: np.ndarray, box) -> bool:
    """Paint the object's segmentation pixels inside its box into `out`. False if
    no source covers enough of the box (the caller then fills the box)."""
    x1, y1, x2, y2 = box
    for s in sources:
        own = _class_index(s, MASK_SEG_CLASS.get(cls, cls))
        any_object = [i for i in (_class_index(s, c) for c in MASK_SEG_CLASS.values())
                      if i is not None]
        # own class first; then any object class (a car labelled "truck" is still the object)
        for idxs in ([own], any_object):
            idxs = [i for i in idxs if i is not None]
            if not idxs:
                continue
            inside = np.isin(s.mask[y1:y2, x1:x2], idxs)
            if inside.size and inside.mean() >= MIN_FILL:
                out[y1:y2, x1:x2] |= inside
                return True
    return False


def from_plain_map(seg: SegmentationOutput):
    return seg.mask != 0, "all segmented classes (%s)" % ", ".join(seg.class_names[1:])
