"""
Every path, model name and threshold of the agent lives here.
Section 1 is the only part you normally edit per machine.

Anything can also be overridden from `.env` (same variable name).
"""

import os
import sys

from dotenv import load_dotenv

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(AGENT_DIR, ".env"))


def _env(name, default):
    value = os.getenv(name)
    return value if value not in (None, "") else default


# ---------------------------------------------------------------------------
# 1. MACHINE PATHS
# ---------------------------------------------------------------------------
# Model repositories can live beside this repository, or anywhere configured
# through .env. Paths and interpreter binaries must belong to the running OS.
OBJECT_DETECTION_DIR = _env("OBJECT_DETECTION_DIR", os.path.dirname(AGENT_DIR))

LAE_DINO_ROOT = _env("LAE_DINO_ROOT", os.path.join(OBJECT_DETECTION_DIR, "LAE-DINO-main"))
SEGEARTH_ROOT = _env(
    "SEGEARTH_ROOT",
    os.path.join(OBJECT_DETECTION_DIR, "SegEarth-OV-3", "SegEarth-OV-3-main"),
)

LAE_CHECKPOINT = _env(
    "LAE_CHECKPOINT", os.path.join(OBJECT_DETECTION_DIR, "lae_dino_swint_lae1m-28ca3a15.pth")
)
LAE_CONFIG_FILE = _env("LAE_CONFIG_FILE", os.path.join(
    LAE_DINO_ROOT, "mmdetection_lae", "configs", "lae_dino", "lae_dino_swin-t_pretrain_LAE-1M.py"
))
LAE_BERT_DIR = _env("LAE_BERT_DIR", os.path.join(LAE_DINO_ROOT, "weights", "bert-base-uncased"))

# Local sam3.pt; empty -> SegEarth tool looks in <repo>/weights/sam3/sam3.pt, then
# downloads facebook/sam3 from Hugging Face (needs HF_TOKEN in .env).
SAM3_CHECKPOINT = _env("SAM3_CHECKPOINT", "")

# ---------------------------------------------------------------------------
# 2. CONDA ENVIRONMENTS (one per tool -- they cannot share a Python process)
# ---------------------------------------------------------------------------
# LAE-DINO needs Python 3.8 + torch 1.10 + mmdet; SegEarth-OV3 needs a modern
# torch. Each tool therefore runs as a worker process inside its own env.
LAE_ENV_NAME = _env("LAE_ENV_NAME", "LAE")
SAM_ENV_NAME = _env("SAM_ENV_NAME", "SAM")
# Existing environment used by UniBuild/outputs/*_run.json on this machine:
# Python 3.10, torch 2.7.1+cpu, NumPy 2.2.6, rasterio 1.4.3, OpenCV 4.12.
UNIBUILD_ENV_NAME = _env("UNIBUILD_ENV_NAME", "geopython")


def _find_env_python(env_name):
    """
    python.exe of a conda env, found WITHOUT hardcoding your user folder:
      * if the agent itself runs in that env -> the current interpreter;
      * otherwise the sibling env next to the current one (...\\envs\\<name>);
      * otherwise the usual conda install locations under your home folder.
    Set LAE_PYTHON / SAM_PYTHON / UNIBUILD_PYTHON in .env to skip the search.
    """
    if os.path.basename(sys.prefix).lower() == env_name.lower():
        return sys.executable

    exe = "python.exe" if os.name == "nt" else os.path.join("bin", "python")
    home = os.path.expanduser("~")
    env_dirs = [os.path.dirname(sys.prefix), os.path.join(sys.prefix, "envs")]
    for base in ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3", ".conda"):
        env_dirs.append(os.path.join(home, base, "envs"))
        env_dirs.append(os.path.join(home, "AppData", "Local", base, "envs"))
    for env_dir in env_dirs:
        candidate = os.path.join(env_dir, env_name, exe)
        if os.path.isfile(candidate):
            return candidate
    return ""  # tools/worker_client.py raises a clear error if still empty


LAE_PYTHON = _env("LAE_PYTHON", "") or _find_env_python(LAE_ENV_NAME)
SAM_PYTHON = _env("SAM_PYTHON", "") or _find_env_python(SAM_ENV_NAME)
UNIBUILD_PYTHON = _env("UNIBUILD_PYTHON", "") or _find_env_python(UNIBUILD_ENV_NAME)

WORKERS_DIR = os.path.join(AGENT_DIR, "workers")
RUNS_DIR = os.path.join(AGENT_DIR, "runs")        # per-question input/outputs
LOGS_DIR = os.path.join(AGENT_DIR, "logs")        # worker stderr (model-loading logs, tracebacks)

# ---------------------------------------------------------------------------
# 3. OBJECT DETECTION (LAE-DINO)
# ---------------------------------------------------------------------------
DET_DEVICE = _env("DET_DEVICE", "cpu")
DET_SCORE_THR = float(_env("DET_SCORE_THR", "0.08"))
DET_NMS_IOU = float(_env("DET_NMS_IOU", "0.3"))
DET_CLASS_AGNOSTIC_NMS = True
DET_MAX_CLASSES = 8           # classes per LAE text prompt

# ---------------------------------------------------------------------------
# 4. SEMANTIC SEGMENTATION (SegEarth-OV3 / SAM 3)
# ---------------------------------------------------------------------------
SEG_DEVICE = _env("SEG_DEVICE", "")            # "" = auto (cuda if available)
SEG_CONFIDENCE_THR = 0.1
SEG_PROB_THR = 0.1
SEG_SLIDE_CROP = int(_env("SEG_SLIDE_CROP", "0"))   # 0 = one pass (fast on CPU)
SEG_SLIDE_STRIDE = 512
SEG_SINGLE_PASS_MAX_SIDE = 1024
SEG_MAX_CLASSES = 16  # supports the default map and the richer field evidence

# One semantic catalog: default presentation and task evidence select from this
# source. Field verification needs agricultural/exclusion evidence, while an
# ordinary scene map must contain exactly the seven user-defined classes below.
SEMANTIC_CLASSES = {
    "buildings": "buildings,building,roof,house",
    "grass": "grass,lawn",
    "tree": "tree,forest,woodland",
    "meadow": "meadow,grassland,pasture",
    "road": "road,pavement,street",
    "water": "water,river,lake,pond",
    "bareland": "bareland,bare soil,barren,fallow land",
    "cropland": "cropland,farmland,cultivated land,arable land,agricultural field",
    "orchard": "orchard,vineyard",
    "urban green space": "urban green space,urban park",
    "sports field": "sports field,football pitch,tennis court,golf course",
    "industrial area": "industrial area,factory yard,industrial yard",
}
DEFAULT_LULC_CLASSES = ["background"] + [SEMANTIC_CLASSES[name] for name in
    ("buildings", "grass", "tree", "meadow", "road", "water", "bareland")]
BUILDING_SEG_CLASS = SEMANTIC_CLASSES["buildings"]

# Delineate uses the upstream full-scene pipeline, never independent agent tiles.
DELINEATE_ROOT = _env("DELINEATE_ROOT", os.path.join(OBJECT_DETECTION_DIR, "Delineate_AnythingV2"))
DELINEATE_UPSTREAM = _env("DELINEATE_UPSTREAM", os.path.join(DELINEATE_ROOT, "upstream"))
DELINEATE_PYTHON = _env("DELINEATE_PYTHON", os.path.join(
    AGENT_DIR, ".venv-delineate", *(('Scripts', 'python.exe') if os.name == 'nt' else ('bin', 'python'))))
DELINEATE_CHECKPOINT = _env("DELINEATE_CHECKPOINT", os.path.join(
    DELINEATE_ROOT, "weights", "DelineateAnythingv2.pt"))
DELINEATE_CONFIDENCE = float(_env("DELINEATE_CONFIDENCE", "0.15"))
# Native valid_area / original_area; smaller is more permissive. No LCLU clipping.
DELINEATE_REMAINING_AREA = float(_env("DELINEATE_REMAINING_AREA", "0.35"))
DELINEATE_MIN_AREA_PX = int(_env("DELINEATE_MIN_AREA_PX", "128"))
DELINEATE_BATCH_SIZE = int(_env("DELINEATE_BATCH_SIZE", "2"))
DELINEATE_POST_WORKERS = int(_env("DELINEATE_POST_WORKERS", "1"))
DELINEATE_TIMEOUT = float(_env("DELINEATE_TIMEOUT", "900"))
FIELD_NONAG_REJECT_FRACTION = float(_env("FIELD_NONAG_REJECT_FRACTION", "0.70"))
FIELD_CROPLAND_SUPPORT_FRACTION = float(_env("FIELD_CROPLAND_SUPPORT_FRACTION", "0.40"))
RGB_BANDS = tuple(int(x) for x in _env("RGB_BANDS", "1,2,3").split(","))

# Legacy LULC building components only; explicit building tasks use UniBuilding.
MIN_BUILDING_AREA_PX = int(_env("MIN_BUILDING_AREA_PX", "40"))

# Areas of every OTHER segmented class (each parking lot, pond, field) become
# selectable regions r0, r1, ... when they cover at least this fraction of the image.
MIN_REGION_AREA_FRAC = float(_env("MIN_REGION_AREA_FRAC", "0.001"))

# UniBuilding (UniBuild DINOv3-Base + HR-DPT), isolated worker environment.
# Actual model/preprocessing helpers live in this local repository.
UNIBUILD_ROOT = _env("UNIBUILD_ROOT", os.path.join(OBJECT_DETECTION_DIR, "UniBuild"))
UNIBUILD_REPO = _env("UNIBUILD_REPO", os.path.join(UNIBUILD_ROOT, "unibuild_repo"))
UNIBUILD_CHECKPOINT = _env("UNIBUILD_CHECKPOINT", os.path.join(
    UNIBUILD_ROOT, "checkpoints", "unibuild_dinov3_base_hrdpt.pth"))
UNIBUILD_DEVICE = _env("UNIBUILD_DEVICE", "")  # auto: CUDA if available, else CPU
UNIBUILD_THRESHOLD = float(_env("UNIBUILD_THRESHOLD", "0.5"))
UNIBUILD_BATCH_SIZE = int(_env("UNIBUILD_BATCH_SIZE", "0"))  # auto: GPU 16 / CPU 2
UNIBUILD_MIN_AREA_PX = int(_env("UNIBUILD_MIN_AREA_PX", "9"))
UNIBUILD_CONNECTIVITY = int(_env("UNIBUILD_CONNECTIVITY", "8"))

# Mask output (plan.output_kind == "mask"): detections the VLM kept are cut from
# SegEarth-OV3 pixels inside their box -- one extra segmentation pass AFTER the
# VLM, on the kept objects' classes only. Off -> each detection is a filled box.
MASK_FROM_SEGMENTATION = _env("MASK_FROM_SEGMENTATION", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# 5. LANGUAGE MODELS (OpenAI-compatible endpoint, key + base URL in .env)
# ---------------------------------------------------------------------------
ROUTER_MODEL = _env("ROUTER_MODEL", "gpt-5.2")   # small, cheap, text-only
VLM_MODEL = _env("VLM_MODEL", "gpt-5.6-luna")          # the vision model

ROUTER_TIMEOUT = 60.0
VLM_CONNECT_TIMEOUT = 60.0
VLM_READ_TIMEOUT = float(_env("VLM_READ_TIMEOUT", "300"))
VLM_RETRIES = 2
# The VLM sees the input image at its ORIGINAL size (lossless PNG). Only an image
# over the API's per-image patch limit is reduced, just enough to fit.
VLM_IMAGE_DETAIL = _env("VLM_IMAGE_DETAIL", "original")
VLM_PATCH_PX = 32               # the API counts images in 32x32 px patches
VLM_MAX_PATCHES = 30000         # per-image limit (~30 MP)
VLM_MAX_DETECTIONS = 500        # highest-score detections sent to the VLM
VLM_MAX_BUILDINGS = 200     # building components listed individually
VLM_MAX_REGIONS_PER_CLASS = 25  # largest r* areas listed per segmentation class
# Bounded field review. These are payload limits, not a cap on scene candidates.
VLM_FIELD_BATCH_SIZE = int(_env("VLM_FIELD_BATCH_SIZE", "64"))
VLM_FIELD_TEXT_BUDGET = int(_env("VLM_FIELD_TEXT_BUDGET", "32000"))
VLM_FIELD_IMAGE_MAX_SIDE = int(_env("VLM_FIELD_IMAGE_MAX_SIDE", "1024"))
VLM_FIELD_MIN_IMAGE_SIDE = int(_env("VLM_FIELD_MIN_IMAGE_SIDE", "256"))
VLM_FIELD_CROP_PADDING = int(_env("VLM_FIELD_CROP_PADDING", "96"))

# ---------------------------------------------------------------------------
# 6. TILING (V02) -- large scenes are cut into tiles; each tile runs the whole
#    agent (tools + VLM) independently. Tile results are NOT merged.
# ---------------------------------------------------------------------------
# Largest tile side. An image whose sides are both <= TILE_SIZE is processed whole.
TILE_SIZE = int(_env("TILE_SIZE", "2000"))
# How a side longer than TILE_SIZE is cut (both strategies NEVER pad a tile;
# see tiling.py):
#   "balanced" (default) ceil(side / TILE_SIZE) equal tiles, no overlap:
#                        7000 x 3000 -> 4 x 2 tiles of 1750 x 1500 px
#   "fixed"              exactly TILE_SIZE tiles; the last row / column is
#                        shifted back inside the image and overlaps its neighbour
TILE_STRATEGY = _env("TILE_STRATEGY", "balanced")
# The router reads only the question, so one plan serves every tile (cheaper and
# consistent). False -> the router runs again for each tile.
TILE_SHARED_PLAN = _env("TILE_SHARED_PLAN", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# 7. USER INTERFACE (V02)
# ---------------------------------------------------------------------------
# Pillow refuses images above ~179 MP as a "decompression bomb"; scenes are big.
MAX_IMAGE_PIXELS = int(_env("MAX_IMAGE_PIXELS", "1000000000"))
TILE_TABS_MAX = 12        # up to this many tiles -> tabs; more -> a tile picker
DISPLAY_MAX_SIDE = 1400   # pictures in the UI are shrunk to this; downloads stay full size
PREVIEW_MAX_SIDE = 1600   # the tile-grid overview of the whole scene
