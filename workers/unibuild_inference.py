"""Local UniBuild adapter for the worker's RGB, native-resolution inference.

Model architecture and tile helpers come from the separately installed upstream
UniBuild repository; checkpoint weights are not bundled here.
"""
import time
from types import SimpleNamespace
import numpy as np

TILE_SIZE = 512
STRIDE = 384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_model(ub, checkpoint_path, device):
    import torch
    from models.segmentation_model import SegmentationModel

    # Same model config as the official script: DINOv3-Base backbone + HR-DPT decoder, 2 classes.
    cfg = ub.build_model_cfg(SimpleNamespace(dinov3_variant="base", backbone_checkpoint=None))
    model = SegmentationModel(cfg, nclass=2)
    # weights_only=True refuses arbitrary pickled code; the checkpoint's training metadata
    # (e.g. best_val_iou) is stored as numpy float64 scalars, so allow exactly those types.
    numpy_scalar_types = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                          np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(numpy_scalar_types):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trained_cfg = checkpoint["model_cfg"]
    for part in ("backbone", "decoder"):
        for key in ("name", "variant"):
            if key in cfg[part] and key in trained_cfg[part] and str(cfg[part][key]).lower() != str(trained_cfg[part][key]).lower():
                raise ValueError(f"Checkpoint was trained with {part}.{key}={trained_cfg[part][key]}, "
                                 f"this script builds {cfg[part][key]}")
    state_dict = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # The official script only prints missing keys. A missing key means random weights, so stop.
    if missing:
        raise RuntimeError(f"Checkpoint lacks {len(missing)} model weights, e.g. {missing[:5]}")
    if unexpected:
        print(f"Note: {len(unexpected)} checkpoint entries are not used by the model, e.g. {unexpected[:5]}")
    return model.to(device).eval()


def to_0_255(raw, stretch):
    """raw: float32 (3, h, w). The model expects 8-bit RGB values (official scale_rgb divides by 255)."""
    if stretch is None:
        if raw.min() < 0 or raw.max() > 255:
            raise ValueError(f"Pixel values range {raw.min():g}..{raw.max():g}, but UniBuild expects 8-bit RGB "
                             f"(0-255). For 16-bit / reflectance data pass --stretch LOW_PCT HIGH_PCT, "
                             f"e.g. --stretch 2 98, after checking it suits your data.")
        return raw
    lo, hi = stretch
    return (raw - lo[:, None, None]) / (hi - lo)[:, None, None] * 255.0


def read_patch(ub, src, row, col, bands, stretch):
    """Mirror of the official read_patch, plus the 8-bit check / optional stretch."""
    import rasterio
    import torch

    h = min(TILE_SIZE, src.height - row)
    w = min(TILE_SIZE, src.width - col)
    raw = src.read(bands, window=rasterio.windows.Window(col, row, w, h)).astype(np.float32)
    rgb = ub.scale_rgb(to_0_255(raw, stretch))  # /255 and clip to [0, 1]
    if h != TILE_SIZE or w != TILE_SIZE:  # zero-pad edge tiles like the official code
        padded = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
        padded[:, :h, :w] = rgb
        rgb = padded
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return (torch.from_numpy(rgb) - mean) / std, h, w


def predict_probability(ub, model, src, bands, stretch, device, batch_size):
    """Sliding-window building probability (H, W) in [0, 1], Hann-weighted blending as in the official code."""
    import torch

    y_starts = ub.compute_starts(src.height, TILE_SIZE, STRIDE)
    x_starts = ub.compute_starts(src.width, TILE_SIZE, STRIDE)
    weight = ub.blend_weight(TILE_SIZE)
    prob_sum = np.zeros((src.height, src.width), dtype=np.float32)
    weight_sum = np.zeros_like(prob_sum)
    windows = [(r, c) for r in y_starts for c in x_starts]
    print(f"Inference grid {src.width} x {src.height} px -> {len(windows)} tiles of {TILE_SIZE} px "
          f"on {device} (batch {batch_size})")

    t0 = time.time()
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        patches, sizes = zip(*[(p, (h, w)) for p, h, w in
                               (read_patch(ub, src, r, c, bands, stretch) for r, c in chunk)])
        probs = ub.predict_batch(model, torch.stack(patches).to(device)).cpu().numpy()
        for prob, (row, col), (h, w) in zip(probs, chunk, sizes):
            prob_sum[row:row + h, col:col + w] += prob[:h, :w] * weight[:h, :w]
            weight_sum[row:row + h, col:col + w] += weight[:h, :w]
        done = i + len(chunk)
        eta = (time.time() - t0) / done * (len(windows) - done)
        print(f"  {done}/{len(windows)} tiles, ~{eta:.0f} s left", flush=True)

    assert np.all(weight_sum > 0), "Some pixels were not covered by any tile."
    prob = prob_sum / weight_sum
    assert np.all((prob >= 0) & (prob <= 1 + 1e-6)), "Blended probability left [0, 1]."
    return prob
