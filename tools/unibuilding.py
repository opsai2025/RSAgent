"""UniBuilding adapter: binary building pixels -> b* proposals and exact masks.

Model loading/inference is in its own persistent worker, like the other tools.
Connected components match the repository's inference defaults (8-connected,
>=9 pixels); boxes use the agent's normalized half-open coordinate convention.
"""

import os

import numpy as np
from PIL import Image

import config
from schemas import UniBuildingOutput
from tools.segmentation import label_components
from tools.worker_client import ToolWorker

_worker = None


def worker() -> ToolWorker:
    global _worker
    if _worker is None:
        _worker = ToolWorker(
            name="UniBuilding", python_exe=config.UNIBUILD_PYTHON,
            script=os.path.join(config.WORKERS_DIR, "unibuilding_worker.py"),
            settings={"root": config.UNIBUILD_ROOT, "repo": config.UNIBUILD_REPO,
                      "checkpoint": config.UNIBUILD_CHECKPOINT, "device": config.UNIBUILD_DEVICE,
                      "threshold": config.UNIBUILD_THRESHOLD, "batch_size": config.UNIBUILD_BATCH_SIZE},
            log_dir=config.LOGS_DIR,
        )
    return _worker


def from_mask(mask, mask_path="", *, seconds=0.0, probability_path="", threshold=None,
              min_area=None, connectivity=None) -> UniBuildingOutput:
    """Adapt real model pixels; also usable offline on an existing UniBuild mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("UniBuilding mask must be a nonempty 2D array")
    min_area = config.UNIBUILD_MIN_AREA_PX if min_area is None else min_area
    connectivity = config.UNIBUILD_CONNECTIVITY if connectivity is None else connectivity
    if min_area < 1:
        raise ValueError("UNIBUILD_MIN_AREA_PX must be positive")
    buildings, labels, ids = label_components(mask, min_area, connectivity=connectivity)
    if labels is None:
        raise RuntimeError("UniBuilding proposals require scipy or opencv-python in the agent environment")
    return UniBuildingOutput(
        mask=mask, mask_path=mask_path, buildings=buildings, component_labels=labels,
        component_id_to_label=ids, coverage=float(mask.mean() * 100),
        threshold=config.UNIBUILD_THRESHOLD if threshold is None else threshold,
        min_area_px=min_area, connectivity=connectivity, seconds=seconds,
        probability_path=probability_path)


def detect(image_path, run_dir) -> UniBuildingOutput:
    out = worker().call({"image_path": image_path,
                         "mask_path": os.path.join(run_dir, "unibuilding_mask.png"),
                         "probability_path": os.path.join(run_dir, "unibuilding_probability.npy")})
    with Image.open(out["mask_path"]) as im:
        mask = np.asarray(im).copy() > 0
    with Image.open(image_path) as im:
        if mask.shape != (im.height, im.width):
            raise ValueError("UniBuilding mask does not match the input image grid")
    return from_mask(mask, out["mask_path"], seconds=out["seconds"],
                     threshold=out["threshold"], probability_path=out["probability_path"])
