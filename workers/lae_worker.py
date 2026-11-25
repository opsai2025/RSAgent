"""
Object-detection worker -- runs inside the LAE conda env (Python 3.8, mmdet).

Same logic as reflection_scenario/lae_detector.py (itself pr03.py): LAE-DINO
open-vocabulary detection -> score threshold -> greedy NMS. The differences are
only in packaging: settings arrive in the "init" message instead of config.py,
and results go back as JSON.

run args:    {"image_path": str, "classes": [str], "score_thr": float,
              "nms_iou": float, "class_agnostic": bool}
run result:  {"raw": [Detection], "nms": [Detection], "seconds": float}
Detection:   {"id": str, "label": str, "bbox": [x1,y1,x2,y2] normalized 0..1, "score": float}

"raw" is the score-thresholded list BEFORE NMS and exists only for the debug
panel in the UI; its ids are "raw0", "raw1", ... The agent sends only "nms" to
the VLM, and those rows are renumbered "d0", "d1", ... AFTER suppression, so the
ids carry no gaps and say nothing about the boxes that were dropped.
"""

import contextlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import serve  # noqa: E402

S = {}          # settings from the init message
MODEL = None
PIPELINE = None


@contextlib.contextmanager
def in_mmdet_cwd():
    """The LAE config resolves some paths against cwd -- scope it to each call."""
    previous = os.getcwd()
    os.chdir(S["mmdet_dir"])
    try:
        yield
    finally:
        os.chdir(previous)


def init(settings):
    global MODEL, PIPELINE
    S.update(settings)
    # YAPF (imported by mmengine) otherwise writes grammar pickles to the user's
    # AppData during model startup. Keep this worker's cache inside V4 instead.
    if S.get("cache_dir"):
        import platformdirs
        original_cache_dir = platformdirs.user_cache_dir
        cache_dir = os.path.abspath(S["cache_dir"])
        os.makedirs(cache_dir, exist_ok=True)
        def worker_cache_dir(appname=None, *args, **kwargs):
            if appname == "YAPF":
                return cache_dir
            return original_cache_dir(appname, *args, **kwargs)
        platformdirs.user_cache_dir = worker_cache_dir
    S["mmdet_dir"] = os.path.join(S["lae_root"], "mmdetection_lae")
    if S["mmdet_dir"] not in sys.path:
        sys.path.insert(0, S["mmdet_dir"])

    from mmcv.transforms import Compose
    from mmdet.apis import init_detector
    from mmdet.utils import get_test_pipeline_cfg

    t0 = time.time()
    with in_mmdet_cwd():
        MODEL = init_detector(
            S["config_file"],
            S["checkpoint"],
            device=S.get("device", "cpu"),
            palette="random",
            cfg_options=dict(
                model=dict(
                    language_model=dict(name=S["bert_dir"]),
                    backbone=dict(init_cfg=None),   # checkpoint overwrites the backbone
                )
            ),
        )
        pipeline_cfg = get_test_pipeline_cfg(MODEL.cfg.copy())
        pipeline_cfg = [t for t in pipeline_cfg if t["type"] != "LoadAnnotations"]
        PIPELINE = Compose(pipeline_cfg)
    return {"model": "LAE-DINO", "load_seconds": round(time.time() - t0, 1)}


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(detections, iou_thr, class_agnostic):
    kept = []
    for det in sorted(detections, key=lambda d: d["score"], reverse=True):
        clash = any(
            (class_agnostic or k["label"] == det["label"]) and iou(k["bbox"], det["bbox"]) > iou_thr
            for k in kept
        )
        if not clash:
            kept.append(det)
    return kept


def number_after_nms(detections):
    """Renumber the survivors d0..dn-1, on COPIES so the pre-NMS rows keep their
    own "raw*" ids. Numbering after suppression is what keeps the raw list out of
    the VLM's view: gapless ids reveal nothing about what NMS removed."""
    out = []
    for i, det in enumerate(detections):
        row = dict(det)
        row["id"] = "d%d" % i
        out.append(row)
    return out


def run(args):
    from mmdet.apis import inference_detector
    from PIL import Image

    classes = [c for c in args.get("classes", []) if c]
    if not classes:
        return {"raw": [], "nms": [], "seconds": 0.0}

    image_path = args["image_path"]
    with Image.open(image_path) as im:
        width, height = im.size
    t0 = time.time()
    with in_mmdet_cwd():
        result = inference_detector(
            MODEL,
            image_path,
            test_pipeline=PIPELINE,
            text_prompt=" . ".join(classes) + " .",
            custom_entities=True,
        )
    pred = result.pred_instances.cpu().numpy()

    score_thr = float(args.get("score_thr", 0.08))
    rows = []
    for box, score, label_idx in zip(pred.bboxes, pred.scores, pred.labels):
        if float(score) < score_thr:
            continue
        idx = int(label_idx)
        x1, y1, x2, y2 = [float(v) for v in box]
        rows.append({
            "label": classes[idx] if 0 <= idx < len(classes) else "class_%d" % idx,
            "bbox": [
                round(max(0.0, min(1.0, x1 / width)), 4),
                round(max(0.0, min(1.0, y1 / height)), 4),
                round(max(0.0, min(1.0, x2 / width)), 4),
                round(max(0.0, min(1.0, y2 / height)), 4),
            ],
            "score": round(float(score), 4),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, row in enumerate(rows):
        row["id"] = "raw%d" % i          # debug / UI only -- never reaches the VLM

    kept = number_after_nms(
        nms(rows, float(args.get("nms_iou", 0.3)), bool(args.get("class_agnostic", True))))
    return {"raw": rows, "nms": kept, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    serve(init, run)
