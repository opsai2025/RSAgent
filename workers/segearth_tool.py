"""
segearth_tool.py
=================
Unified, self-contained wrapper around SegEarth-OV3 (SAM 3-based) open-vocabulary
semantic segmentation.

Refactored out of `segearth_ov3_infer.py` (not `segearthov3_segmentor.py`): the
`_infer` script is the one that runs standalone -- no mmcv / mmsegmentation needed,
CPU or GPU -- so it's a much better base for a reusable class than the mmseg
`BaseSegmentor` subclass, which hardcodes `device="cuda"` and needs the full
OpenMMLab stack just to hold a `PixelData`/`SegDataSample`.

Everything here (model build, CPU-safety patches, sliding-window vs single-pass,
instance/semantic/presence fusion, synonym merging, coloring) is the same logic
as `segearth_ov3_infer.py`, just packaged as one class with `segment()` /
`plot_original()` / `plot_segmentation()` methods instead of a top-to-bottom
script, so it can be dropped into any future app.

Typical use:

    from segearth_tool import SegEarthOV3Segmenter, DEFAULT_LULC_CLASSES

    segmenter = SegEarthOV3Segmenter(repo_dir=r"D:\\for GIS-DB\\Golkar\\Object_Detection\\SegEarth-OV-3\\SegEarth-OV-3-main")
    result = segmenter.segment(image, DEFAULT_LULC_CLASSES)   # image: PIL.Image
    fig1 = segmenter.plot_original(image)
    fig2 = segmenter.plot_segmentation(image, result)
"""

from __future__ import annotations

import io
import os
import sys
import types
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # safe for a server-side Streamlit process
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Used whenever the caller asks for "a land use / land cover map" without naming
# classes. Index 0 ("background") is always added automatically by the segmenter,
# so this list itself only needs to carry the six requested classes.
DEFAULT_LULC_CLASSES: List[str] = [
    "background",
    "building,roof,house",
    "road,pavement,street",
    "bare soil,bareland,barren",
    "grass,green space,lawn,meadow",
    "forest,tree,woodland",
    "water,river,lake,pond",
]

DEFAULT_PALETTE: List[Tuple[int, int, int]] = [
    (0, 0, 0), (166, 124, 60), (88, 184, 72), (128, 128, 128),
    (255, 64, 64), (22, 110, 45), (40, 110, 220), (240, 200, 60),
    (230, 90, 200), (120, 60, 200), (60, 220, 220), (255, 150, 40),
    (150, 255, 150), (200, 200, 255), (140, 30, 30), (30, 140, 140),
    (255, 255, 120), (90, 90, 90), (200, 120, 255), (10, 60, 100),
]


@dataclass
class SegmentationResult:
    mask: np.ndarray                       # (H, W) uint8 class-id map, full input resolution
    color_mask: np.ndarray                 # (H, W, 3) uint8
    overlay: np.ndarray                    # (H, W, 3) uint8, mask blended over the input image
    class_names: List[str]                 # display name per class id (first synonym only)
    colors: np.ndarray                     # (num_classes, 3) uint8, palette actually used
    coverage: Dict[str, float] = field(default_factory=dict)   # class name -> % of pixels

    @property
    def legend(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """[(class name, RGB in 0..1)], ready for matplotlib.patches.Patch."""
        return [(name, tuple(c / 255.0 for c in self.colors[i]))
                for i, name in enumerate(self.class_names)]


def _stub_module(name: str, **attributes):
    """Install a placeholder module so `import sam3` survives missing optional
    deps (triton, pycocotools) that only sam3's video / training code needs.
    Image inference never touches either of them."""
    module = types.ModuleType(name)

    def __getattr__(attribute):
        # Dunders must stay missing: torch/inspect walk sys.modules and read
        # __file__ / __spec__ off every module; a fake value crashes them.
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        return object

    module.__getattr__ = __getattr__
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class SegEarthOV3Segmenter:
    """Loads SAM 3 / SegEarth-OV3 once and runs open-vocabulary semantic
    segmentation on PIL images with an arbitrary, per-call class vocabulary.

    Instantiation is expensive (model load); create ONE instance and reuse it
    for every image / call (e.g. cache it with `st.cache_resource` in Streamlit).
    """

    _FACTORIES = ("zeros", "ones", "empty", "full", "arange", "linspace", "tensor", "eye")

    def __init__(
        self,
        repo_dir: str,
        checkpoint_path: str = "",
        hf_token: Optional[str] = None,
        confidence_threshold: float = 0.1,
        prob_threshold: float = 0.1,
        background_index: int = 0,
        slide_crop: int = 0,
        slide_stride: int = 512,
        single_pass_max_side: int = 1024,
        alpha: float = 0.55,
        palette: Optional[Sequence[Tuple[int, int, int]]] = None,
        device: Optional[str] = None,
    ):
        """
        repo_dir: path to the cloned SegEarth-OV-3 repo (needs `sam3/` inside it).
        checkpoint_path: local path to sam3.pt. If empty, tries
            `<repo_dir>/weights/sam3/sam3.pt`, then falls back to downloading
            `facebook/sam3` from the Hugging Face Hub (gated repo -> needs a
            token that accepted the licence; pass `hf_token=` or set env HF_TOKEN).
        slide_crop=0 means "whole image in one pass" (recommended on CPU / for
            scenes up to a couple thousand px). Set e.g. 512 for very large
            scenes / small objects, with slide_stride = slide_crop // 2 for
            overlap.
        device: "cuda" or "cpu". Auto-detected if not given.
        """
        self.repo_dir = os.path.abspath(repo_dir)
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        self.confidence_threshold = confidence_threshold
        self.prob_threshold = prob_threshold
        self.background_index = background_index
        self.slide_crop = slide_crop
        self.slide_stride = slide_stride
        self.single_pass_max_side = single_pass_max_side
        self.alpha = alpha
        self.palette = list(palette) if palette else DEFAULT_PALETTE

        # NOTE: do not hardcode a real token here -- set HF_TOKEN in your
        # environment / .env file instead.
        self.hf_token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN", "")
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.processor = None
        self._load_model(checkpoint_path)

    # -- model loading -------------------------------------------------

    def _load_model(self, checkpoint_path: str):
        bpe_path = os.path.join(self.repo_dir, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

        try:
            import triton  # noqa: F401
        except ImportError:
            def edt_triton(*args, **kwargs):
                raise RuntimeError("edt_triton requires Triton; only the SAM 3 video tracker uses it")
            _stub_module("sam3.model.edt", edt_triton=edt_triton)

        try:
            import pycocotools.mask  # noqa: F401
        except ImportError:
            _stub_module("pycocotools", mask=_stub_module("pycocotools.mask"))

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if checkpoint_path and os.path.isfile(checkpoint_path):
            resolved_checkpoint = checkpoint_path
        else:
            default_local = os.path.join(self.repo_dir, "weights", "sam3", "sam3.pt")
            if os.path.isfile(default_local):
                resolved_checkpoint = default_local
            else:
                from huggingface_hub import hf_hub_download
                resolved_checkpoint = hf_hub_download(
                    "facebook/sam3", "sam3.pt", token=self.hf_token or None
                )

        # Two constructors in sam3 hardcode device="cuda" while precomputing caches
        # (position_encoding.py, decoder.py's boxRPB coords). On CPU-only torch that
        # raises, so the tensor factories are redirected for the duration of the
        # build only. Everything at inference time already follows the tensors'
        # own device.
        original_factories = {name: getattr(torch, name) for name in self._FACTORIES}

        def _redirect_cuda(original):
            def factory(*args, **kwargs):
                if str(kwargs.get("device", "")).startswith("cuda"):
                    kwargs["device"] = self.device
                return original(*args, **kwargs)
            return factory

        if self.device != "cuda":
            for name, original in original_factories.items():
                setattr(torch, name, _redirect_cuda(original))
            warnings.filterwarnings("ignore", message="CUDA is not available")
            # geometry_encoders.py calls scale.pin_memory() at inference time; pinned
            # memory is meaningless without an accelerator and raises instead of
            # being ignored. This patch is intentionally permanent (unlike the
            # factory redirect above), since the call happens inside forward().
            torch.Tensor.pin_memory = lambda self_, *a, **kw: self_

        try:
            self.model = build_sam3_image_model(
                bpe_path=bpe_path,             # must be passed: the built-in default path is wrong
                checkpoint_path=resolved_checkpoint,
                device=self.device,
                eval_mode=True,
                load_from_HF=False,
                enable_segmentation=True,      # the semantic head is required by this pipeline
            )
        finally:
            if self.device != "cuda":
                for name, original in original_factories.items():
                    setattr(torch, name, original)

        self.processor = Sam3Processor(
            self.model, confidence_threshold=self.confidence_threshold, device=self.device
        )

    # -- inference -------------------------------------------------

    @staticmethod
    def _split_classes(class_names: Sequence[str]):
        """"a,b" -> two text queries that both map to the same class index."""
        query_words, query_index = [], []
        for class_id, line in enumerate(class_names):
            for word in line.split(","):
                word = word.strip()
                if word:
                    query_words.append(word)
                    query_index.append(class_id)
        return query_words, query_index

    def _segment_crop(self, crop: Image.Image, query_words: List[str]) -> torch.Tensor:
        """Return per-text-query score maps, shape (num_queries, h, w), values in [0, 1]."""
        w, h = crop.size
        logits = torch.zeros((len(query_words), h, w), device=self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == "cuda")
        ):
            state = self.processor.set_image(crop)                  # image encoded once...

            for q, word in enumerate(query_words):                  # ...text prompts reused
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(state=state, prompt=word)

                # (a) instance head: element-wise max over the detected objects,
                #     each mask weighted by its object score
                instance_masks = state["masks_logits"]              # (n, 1, h, w)
                for i in range(instance_masks.shape[0]):
                    mask = instance_masks[i].squeeze()
                    score = state["object_score"][i]
                    if mask.shape != (h, w):
                        mask = F.interpolate(
                            mask.view(1, 1, *mask.shape), size=(h, w),
                            mode="bilinear", align_corners=False
                        ).squeeze()
                    logits[q] = torch.max(logits[q], mask * score)

                # (b) semantic head: full-coverage map, fused with the same max
                semantic = state["semantic_mask_logits"]            # (1, 1, h, w)
                if semantic.shape != (h, w):
                    semantic = F.interpolate(
                        semantic, size=(h, w), mode="bilinear", align_corners=False
                    ).squeeze()
                logits[q] = torch.max(logits[q], semantic)

                # (c) presence head: suppress classes absent from the scene
                logits[q] = logits[q] * state["presence_score"]

        return logits

    def segment(self, image: Image.Image, class_names: Sequence[str]) -> SegmentationResult:
        """Run open-vocabulary segmentation on `image` with the given vocabulary.

        `class_names` follows the demo.py / segearth_ov3_infer.py convention: one
        entry per output class, comma-separated synonyms are merged into the same
        class, and index 0 is conventionally "background". Use `DEFAULT_LULC_CLASSES`
        for the generic 6-class land-use/land-cover scheme.
        """
        image = self._to_rgb(image)
        width, height = image.size

        query_words, query_index = self._split_classes(class_names)
        num_classes = len(class_names)

        if self.slide_crop > 0 and (self.slide_crop < width or self.slide_crop < height):
            # accumulators stay on CPU so very large scenes don't fill the GPU
            scores = torch.zeros((len(query_words), height, width))
            counts = torch.zeros((1, height, width))
            rows = max(height - self.slide_crop + self.slide_stride - 1, 0) // self.slide_stride + 1
            cols = max(width - self.slide_crop + self.slide_stride - 1, 0) // self.slide_stride + 1

            for r in range(rows):
                for c in range(cols):
                    y2 = min(r * self.slide_stride + self.slide_crop, height)
                    x2 = min(c * self.slide_stride + self.slide_crop, width)
                    y1 = max(y2 - self.slide_crop, 0)
                    x1 = max(x2 - self.slide_crop, 0)
                    crop_logits = self._segment_crop(image.crop((x1, y1, x2, y2)), query_words)
                    scores[:, y1:y2, x1:x2] += crop_logits.float().cpu()
                    counts[:, y1:y2, x1:x2] += 1
            assert (counts == 0).sum() == 0, "sliding window did not cover the whole image"
            scores = scores / counts
        else:
            single_pass_image = image
            if self.single_pass_max_side and max(width, height) > self.single_pass_max_side:
                ratio = self.single_pass_max_side / max(width, height)
                single_pass_image = image.resize(
                    (round(width * ratio), round(height * ratio)), Image.BILINEAR
                )
            scores = self._segment_crop(single_pass_image, query_words).float().cpu()

        # synonyms -> classes
        score_height, score_width = scores.shape[-2:]  # may be the downscaled single pass
        class_scores = torch.zeros((num_classes, score_height, score_width))
        for q, class_id in enumerate(query_index):
            class_scores[class_id] = torch.max(class_scores[class_id], scores[q])

        best_score, prediction = class_scores.max(dim=0)
        prediction[best_score < self.prob_threshold] = self.background_index

        mask = prediction.numpy().astype(np.uint8)
        if mask.shape != (height, width):           # back to the input resolution
            mask = np.array(Image.fromarray(mask).resize((width, height), Image.NEAREST))

        colors = np.array(
            [self.palette[i % len(self.palette)] for i in range(num_classes)], dtype=np.uint8
        )
        color_mask = colors[mask]
        rgb = np.array(image, dtype=np.float32)
        overlay = ((1 - self.alpha) * rgb + self.alpha * color_mask.astype(np.float32)).astype(np.uint8)

        display_names = [name.split(",")[0].strip() for name in class_names]
        coverage = {
            display_names[i]: float((mask == i).sum()) / mask.size * 100 for i in range(num_classes)
        }

        return SegmentationResult(
            mask=mask, color_mask=color_mask, overlay=overlay,
            class_names=display_names, colors=colors, coverage=coverage,
        )

    # -- visualization -------------------------------------------------

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        if image.mode == "RGB":
            return image
        band = np.array(image)
        if band.ndim == 2:                       # single band / panchromatic
            band = np.stack([band] * 3, axis=-1)
        band = band[:, :, :3]                    # RGBA / RGBN -> keep R,G,B
        if band.dtype != np.uint8:               # 12/16-bit imagery -> 2-98% stretch
            low, high = np.percentile(band, (2, 98))
            band = (np.clip((band - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(band, mode="RGB")

    def plot_original(self, image: Image.Image):
        """Figure 1: the plain input image."""
        image = self._to_rgb(image)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.set_title("Original image")
        ax.axis("off")
        fig.tight_layout()
        return fig

    def plot_segmentation(self, image: Image.Image, result: SegmentationResult):
        """Figure 2: the segmentation map (overlay) with a proper legend."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(result.overlay)
        ax.set_title("SegEarth-OV3 segmentation")
        ax.axis("off")
        handles = [mpatches.Patch(color=color, label=name) for name, color in result.legend]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
        fig.tight_layout()
        return fig

    def render_side_by_side_png(self, image: Image.Image, result: SegmentationResult) -> bytes:
        """Input + segmentation-with-legend combined into one PNG. This is what the
        VLM is shown when it "sees" the tool output -- it never runs segmentation
        itself, it only describes this image."""
        image = self._to_rgb(image)
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        ax[0].imshow(image)
        ax[0].set_title("input")
        ax[0].axis("off")
        ax[1].imshow(result.overlay)
        ax[1].set_title("SegEarth-OV3")
        ax[1].axis("off")
        handles = [mpatches.Patch(color=color, label=name) for name, color in result.legend]
        ax[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()