"""UniBuilding worker; uses the local adapter and upstream UniBuild model code.

Only this process imports torch/rasterio/model modules. The CPU-capable adapter's
load_model / predict_probability reuse official infer_geotiff helpers: RGB /255,
ImageNet normalization, 512 px windows, stride 384, Hann blending, class-1 softmax.
The agent supplies RGB PNG tiles without georeferencing, so inference uses their
native grid, just as the sample does for PNG. No downloads or GIS resampling.
"""

import os
import sys
import time
from pathlib import Path

from protocol import serve

S = {}
MODEL = UB = SAMPLE = DEVICE = None
BATCH_SIZE = None


def init(settings):
    global MODEL, UB, SAMPLE, DEVICE, BATCH_SIZE
    S.clear()
    S.update(settings)
    repo = Path(S["repo"]).resolve()
    checkpoint = Path(S["checkpoint"]).resolve()
    for path in (repo / "infer_geotiff.py", checkpoint):
        if not path.is_file():
            raise FileNotFoundError("UniBuilding required file missing: %s" % path)
    if not 0 < S["threshold"] < 1:
        raise ValueError("UNIBUILD_THRESHOLD must be in (0, 1)")
    if S["batch_size"] < 0:
        raise ValueError("UNIBUILD_BATCH_SIZE must be 0 (auto) or positive")

    # Keep all external source/model files unchanged, including bytecode caches.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    import torch
    import infer_geotiff

    import unibuild_inference
    SAMPLE = unibuild_inference
    UB = infer_geotiff
    DEVICE = torch.device(S.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    BATCH_SIZE = S["batch_size"] or (16 if DEVICE.type == "cuda" else 2)
    t0 = time.time()
    MODEL = SAMPLE.load_model(UB, checkpoint, DEVICE)
    return {"model": "UniBuilding (UniBuild DINOv3-Base HR-DPT)", "device": str(DEVICE),
            "checkpoint": str(checkpoint), "load_seconds": round(time.time() - t0, 1)}


def run(args):
    import numpy as np
    import rasterio
    from PIL import Image
    import warnings
    from rasterio.errors import NotGeoreferencedWarning

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(args["image_path"]) as src:
            if src.count != 3 or any(dtype != "uint8" for dtype in src.dtypes):
                raise ValueError("UniBuilding agent input must be an 8-bit RGB image")
            probability = SAMPLE.predict_probability(
                UB, MODEL, src, [1, 2, 3], None, DEVICE, BATCH_SIZE)
    if not np.isfinite(probability).all():
        raise ValueError("UniBuilding returned non-finite probabilities")
    mask = probability >= S["threshold"]
    Image.fromarray(mask.astype(np.uint8) * 255).save(args["mask_path"])
    np.save(args["probability_path"], probability.astype(np.float32), allow_pickle=False)
    return {"mask_path": args["mask_path"], "probability_path": args["probability_path"],
            "threshold": S["threshold"], "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    serve(init, run)
