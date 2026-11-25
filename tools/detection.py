"""
Object-detection tool (LAE-DINO), as the agent sees it.

    detect(image_path, classes) -> DetectionOutput

The model itself runs in the LAE env via workers/lae_worker.py; this module only
owns the worker handle (one per Python process, so the model loads once).
"""

import os

import config
from schemas import DetectionOutput
from tools.worker_client import ToolWorker

_worker = None


def worker() -> ToolWorker:
    global _worker
    if _worker is None:
        _worker = ToolWorker(
            name="LAE-DINO",
            python_exe=config.LAE_PYTHON,
            script=os.path.join(config.WORKERS_DIR, "lae_worker.py"),
            settings={
                "lae_root": config.LAE_DINO_ROOT,
                "config_file": config.LAE_CONFIG_FILE,
                "checkpoint": config.LAE_CHECKPOINT,
                "bert_dir": config.LAE_BERT_DIR,
                "device": config.DET_DEVICE,
                "cache_dir": os.path.join(config.LOGS_DIR, "lae_cache"),
            },
            log_dir=config.LOGS_DIR,
        )
    return _worker


def detect(image_path, classes, score_thr=config.DET_SCORE_THR, nms_iou=config.DET_NMS_IOU):
    out = worker().call({
        "image_path": image_path,
        "classes": classes,
        "score_thr": score_thr,
        "nms_iou": nms_iou,
        "class_agnostic": config.DET_CLASS_AGNOSTIC_NMS,
    })
    return DetectionOutput(classes=list(classes), raw=out["raw"], nms=out["nms"],
                           seconds=out["seconds"])
