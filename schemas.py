"""
Data passed between the agent's parts.

Box convention everywhere: bbox = [x1, y1, x2, y2] normalized 0..1 against the
analysed image -- in V02 that is the TILE the agent ran on, not the full scene.
Tool adapters normalize pixel boxes; drawing.py converts them back to pixels.
Tile.to_scene_px() maps a tile box back to the scene.
V4 fields additionally carry original Polygon/MultiPolygon geometry in image
pixels. Their bounding boxes never replace their geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# The router's decision
# ---------------------------------------------------------------------------
class RequestTarget(BaseModel):
    """A subject or contextual dependency, before any tool is selected."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["land_cover", "object", "building", "agricultural_field"]
    category: str = Field(default="", description="The subject category named by the user.")
    classes: List[str] = Field(default_factory=list, description=(
        "Explicit requested classes, or meaningful subclasses of the requested category. "
        "English lowercase; comma-separated synonyms are one class. Do not copy the LULC defaults."))
    role: Literal["target", "context"] = "target"


class RequestIntent(BaseModel):
    """Semantic request contract. Presentation never selects a model."""

    model_config = ConfigDict(extra="forbid")
    operation: Literal["describe", "locate", "classify", "count", "coverage", "delineate", "select", "analyze"]
    targets: List[RequestTarget] = Field(default_factory=list)
    land_cover_scope: Literal["none", "requested_classes", "general_scene"] = Field(default="none",
        description="general_scene for whole-image classification with no narrower subject; uses configured scene classes.")
    presentation: Literal["boxes", "mask", "classification_map"] = "boxes"
    requires_reasoning: bool = Field(default=False, description=(
        "Attributes, relationships, comparison, uncertainty or selection beyond the tool's raw class map."))
    constraints: List[str] = Field(default_factory=list, description="Selection/spatial/attribute conditions, not class names.")
    classification_basis: str = Field(default="", description="What distinguishes classes: type, colour, material, damage, land cover, etc.")
    output_classes: List[str] = Field(default_factory=list, description="Requested result categories; attribute labels are not detector prompts.")
    reasoning: str = ""


class Plan(BaseModel):
    request: Optional[RequestIntent] = Field(default=None, description="Resolved subject/operation/scope, before tool compilation.")
    use_detection: bool = Field(
        default=False,
        description="Run LAE-DINO object detection (discrete, countable NON-BUILDING objects: "
        "vehicles, ships, planes, storage tanks, sports fields, bridges, ...).",
    )
    detection_classes: List[str] = Field(
        default_factory=list,
        description="Bare singular lowercase nouns for the detector.",
    )
    use_segmentation: bool = Field(
        default=False,
        description="Run SegEarth-OV3 semantic segmentation (land "
        "cover, areas, percentages, maps).",
    )
    segmentation_classes: List[str] = Field(
        default_factory=list,
        description="One entry per class; comma-separated synonyms allowed, "
        "e.g. 'water,river,lake'. Targeted segmentation must have explicit classes; "
        "the planner materializes the full scheme only for a general-scene LULC request.",
    )
    use_unibuilding: bool = Field(
        default=False,
        description="Run UniBuilding for buildings, houses, homes, roofs and rooftops. "
        "Combine with detection for non-building objects or segmentation for land cover.",
    )
    protect_semantic_buildings: bool = Field(default=False, description=(
        "Compiler policy: run UniBuilding before semantic inference, exclude its pixels from "
        "non-building classes and use only UniBuilding for the building class."))
    use_delineate: bool = Field(
        default=False,
        description="Agricultural field instances and boundaries ONLY. Always requires "
        "SegEarth full LULC evidence and VLM reasoning; never use for farmland percentages, "
        "urban parks or sports facilities.",
    )
    output: Literal["vlm_answer", "tool_output"] = Field(
        default="vlm_answer",
        description="WHO DECIDES. 'tool_output' for native semantic/building class identities "
        "with nothing to interpret or select; 'vlm_answer' (the VLM decides) "
        "for everything else.",
    )
    output_kind: Literal["boxes", "mask", "classification_map"] = Field(
        default="boxes",
        description="HOW THE RESULT IS SHOWN -- independent of the tools and of `output`. "
        "'boxes' (default): the selected objects/areas as bounding boxes. 'mask': ONLY when "
        "the user explicitly wants a pixel mask / mask /binary mask / binary image as the result. "
        "'classification_map': one dense categorical pixel mask of requested classes. "
        "Use measured pixels or native polygons, never boxes or blended image overlays. "
        "Object pixels may be fetched after instance selection; no default vocabulary is implied.",
    )
    reasoning: str = Field(default="", description="One short sentence. Shown in the UI.")

    # filled in by planner.apply_rules() -- never by the LLM
    rule_notes: List[str] = Field(default_factory=list, description="Always leave empty.")

    @field_validator("output_kind", mode="before")
    @classmethod
    def _normalize_output_kind(cls, value):
        """A near-miss from the router ("binary mask", "bounding boxes", null) must
        not invalidate the whole plan: anything naming a mask/binary -> "mask",
        everything else -> "boxes"."""
        text = str(value or "").strip().lower()
        if "mask" in text or "binary" in text:
            return "mask"
        if text in ("map", "classification_map", "classification map", "class map", "multiclass map"):
            return "classification_map"
        return "boxes"

    @property
    def tools(self) -> List[str]:
        names = []
        if self.use_detection:
            names.append("detection")
        if self.use_segmentation:
            names.append("segmentation")
        if self.use_unibuilding:
            names.append("unibuilding")
        if self.use_delineate:
            names.append("delineate")
        return names


# ---------------------------------------------------------------------------
# Tool outputs
# ---------------------------------------------------------------------------
@dataclass
class DetectionOutput:
    """`raw` is for the debug panel and the trace only. Everything the VLM is
    shown is built from `nms` (see vlm.vlm_detections); nothing in this pipeline
    passes `raw` to the model."""

    classes: List[str]
    raw: List[dict]            # thresholded, before NMS -- ids raw0, raw1, ... DEBUG/UI ONLY
    nms: List[dict]            # after NMS -- ids d0, d1, ... the ONLY detections the VLM sees
    seconds: float

    def counts(self) -> Dict[str, int]:
        """Counts AFTER NMS: what the VLM is told, before its own rejections and
        additions -- not the final answer's count."""
        out: Dict[str, int] = {}
        for det in self.nms:
            out[det["label"]] = out.get(det["label"], 0) + 1
        return out


@dataclass
class SegmentationOutput:
    classes: List[str]         # vocabulary as sent (with synonyms)
    class_names: List[str]     # display name per class id
    colors: List[List[int]]    # RGB per class id
    coverage: Dict[str, float] # display name -> % of pixels
    mask: np.ndarray           # (H, W) uint8 class-id map
    mask_path: str
    buildings: List[dict] = field(default_factory=list)  # legacy LULC components, not VLM building proposals
    regions: List[dict] = field(default_factory=list)    # other classes' areas r0, r1, ...
    building_class: Optional[str] = None
    component_labels: Optional[np.ndarray] = None       # one label image for all b*/r* ids
    component_id_to_label: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class UniBuildingOutput:
    """UniBuild binary mask and connected-component proposals on the input grid.

    The model supplies pixels, not instance boxes. b* proposals are derived with
    the repository's default 8-connectivity and minimum instance area (9 px).
    Touching roofs may form one component; no detector confidence is invented.
    """

    mask: np.ndarray                    # (H, W) bool, native image/tile grid
    mask_path: str                     # visible 0/255 PNG
    buildings: List[dict]               # id b*, normalized bbox, area_px
    component_labels: np.ndarray        # int32, 0 = background
    component_id_to_label: Dict[str, int]
    coverage: float                    # building pixels as % of all pixels
    threshold: float
    min_area_px: int
    connectivity: int
    seconds: float = 0.0
    probability_path: str = ""         # float32 .npy, same grid


# ---------------------------------------------------------------------------
@dataclass
class FieldOutput:
    """Polygons in original-image pixel coordinates, with holes/multipart intact.

    raw_fields is the native LCLU-filtered output before the extra semantic check.
    Raw here does not mean an additional unfiltered model inference.
    """

    raw_fields: List[dict]
    fields: List[dict]
    rejected: List[dict]
    lclu_path: str
    native_gpkg_path: str
    gpkg_path: str = ""
    metadata: dict = field(default_factory=dict)
    filter_config: dict = field(default_factory=dict)
    seconds: float = 0.0


# What the orchestrator returns to the UI
# ---------------------------------------------------------------------------
@dataclass
class Step:
    name: str                  # "Router", "LAE-DINO", "SegEarth-OV3", "UniBuilding", "VLM"
    detail: str
    seconds: float = 0.0
    status: str = "ok"         # ok | error | skipped
    parent: str = ""           # nested review timings are already in the parent total


@dataclass(frozen=True)
class Tile:
    """One window of the input scene (see tiling.py). Pixel bounds are half-open,
    [x0, x1) x [y0, y1), and always lie INSIDE the scene -- a tile is never padded."""

    index: int                 # 1-based, row-major -> "Tile 1", "Tile 2", ...
    count: int                 # tiles in the scene
    row: int                   # 1-based grid position
    col: int
    rows: int
    cols: int
    x0: int
    y0: int
    x1: int
    y1: int
    scene_w: int
    scene_h: int

    @property
    def name(self) -> str:
        return "Tile %d" % self.index

    @property
    def box(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def size(self):
        return (self.x1 - self.x0, self.y1 - self.y0)

    @property
    def is_whole_scene(self) -> bool:
        return self.count == 1

    def describe(self) -> str:
        return "row %d / %d · column %d / %d · scene pixels x %d–%d, y %d–%d · %d × %d px" % (
            self.row, self.rows, self.col, self.cols,
            self.x0, self.x1 - 1, self.y0, self.y1 - 1, self.size[0], self.size[1])

    def to_scene_px(self, bbox) -> List[int]:
        """A normalized box of this tile -> pixel box in the full scene (for a
        later merging stage; V02 itself does not merge tiles)."""
        w, h = self.size
        return [round(self.x0 + bbox[0] * w), round(self.y0 + bbox[1] * h),
                round(self.x0 + bbox[2] * w), round(self.y0 + bbox[3] * h)]

    def as_dict(self) -> dict:
        return {"name": self.name, "index": self.index, "count": self.count,
                "row": self.row, "col": self.col, "grid": [self.rows, self.cols],
                "box_px": list(self.box), "size_px": list(self.size),
                "scene_size_px": [self.scene_w, self.scene_h]}


@dataclass
class AgentResult:
    question: str
    plan: Plan
    steps: List[Step] = field(default_factory=list)
    tile: Optional[Tile] = None  # the window this result was computed on (None = called directly)
    detection: Optional[DetectionOutput] = None
    segmentation: Optional[SegmentationOutput] = None
    unibuilding: Optional[UniBuildingOutput] = None
    fields: Optional[FieldOutput] = None
    source_path: str = ""       # original upload, retained for geospatial metadata
    answer: str = ""
    answered_by: str = ""      # "VLM (<model>)" or "Tool output — VLM not used"
    vlm_objects: List[dict] = field(default_factory=list)
    vlm_rejected: List[dict] = field(default_factory=list)
    final_mask: Optional[np.ndarray] = None   # (H, W) bool -- only when plan.output_kind == "mask"
    final_mask_note: str = ""                 # how it was built (or why it could not be)
    run_dir: str = ""
    output_image_path: str = "" # orchestrator-rendered presentation, independent of tool choice
    output_caption: str = ""
    resume_source: str = ""
    semantic_building_guard: Dict = field(default_factory=dict)


@dataclass
class TiledRun:
    """One question on one scene: the tiles it was cut into and one independent
    AgentResult per tile (same order). Results are NOT merged at this stage."""

    question: str
    scene_size: tuple          # (w, h) of the input image
    tile_size: int             # max tile side requested (config.TILE_SIZE)
    strategy: str              # "balanced" | "fixed" (see tiling.py)
    tiles: List[Tile] = field(default_factory=list)
    results: List[AgentResult] = field(default_factory=list)
    run_dir: str = ""
    seconds: float = 0.0

    @property
    def tiled(self) -> bool:
        return len(self.tiles) > 1
