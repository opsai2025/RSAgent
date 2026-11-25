"""
All pictures the agent makes. Pure functions, always on a copy of the image.

- draw_boxes(...)          detection boxes (colour by class) or VLM objects
                           (green = LAE box kept, cyan = building component kept,
                           magenta = VLM added)
- seg_overlay(...)         class map blended over the image
- add_legend(...)          legend strip on the right (what the VLM and the user see)
- seg_color_map(...)       pure class-colour map (the "classification map" product)
- draw_tile_grid(...)      V02: the scene shrunk, with the tile windows numbered
                           (one tile highlighted = the "where am I" locator)
- fit(...)                 V02: shrink a picture for on-screen display only
"""

from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_PALETTE = [(255, 59, 48), (0, 122, 255), (52, 199, 89), (255, 149, 0),
                 (175, 82, 222), (0, 199, 190), (255, 214, 10), (255, 45, 145)]
ORIGIN_COLORS = {"detector": (52, 199, 89),      # green  -- LAE box the VLM kept
                 "unibuilding": (255, 149, 0), # orange -- UniBuilding component kept
                 "segmentation": (0, 199, 255), # cyan   -- building component the VLM kept
                 "vlm": (255, 45, 145)}         # magenta -- found by the VLM itself


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def class_colors(labels: Sequence[str]) -> Dict[str, tuple]:
    unique = list(dict.fromkeys(labels))
    return {label: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i, label in enumerate(unique)}


def draw_boxes(image: Image.Image, items: List[dict]) -> Image.Image:
    """items: [{"bbox": normalized, "text": str, "color": (r,g,b)}]"""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _font(max(12, int(min(w, h) / 50)))
    width = max(2, int(min(w, h) / 400))
    for it in items:
        x1, y1, x2, y2 = (int(it["bbox"][0] * w), int(it["bbox"][1] * h),
                          int(it["bbox"][2] * w), int(it["bbox"][3] * h))
        draw.rectangle([x1, y1, x2, y2], outline=it["color"], width=width)
        if it.get("text"):
            left, top, right, bottom = draw.textbbox((0, 0), it["text"], font=font)
            ty = y1 - (bottom - top) - 4 if y1 - (bottom - top) - 4 >= 0 else y1 + 2
            draw.rectangle([x1, ty, x1 + right - left + 6, ty + bottom - top + 4], fill=it["color"])
            draw.text((x1 + 3, ty + 2), it["text"], fill=(255, 255, 255), font=font)
    return img


def draw_detections(image, detections: List[dict]) -> Image.Image:
    colors = class_colors([d["label"] for d in detections])
    return draw_boxes(image, [
        {"bbox": d["bbox"], "text": "%s %s %.2f" % (d["id"], d["label"], d["score"]),
         "color": colors[d["label"]]} for d in detections
    ])


def draw_vlm_objects(image, objects: List[dict]) -> Image.Image:
    picture = draw_boxes(image, [
        {"bbox": o["box"], "text": o.get("label", ""),
         "color": ORIGIN_COLORS.get(o.get("origin"), ORIGIN_COLORS["vlm"])} for o in objects
        if o.get("origin") != "delineate"
    ])
    return draw_fields(picture, [o for o in objects if o.get("origin") == "delineate"])


def draw_fields(image, fields, by_class=False):
    """Draw exact polygon rings (including holes); never replace them with boxes."""
    from field_geometry import geometry_bounds, polygon_pixels
    rgb = np.asarray(image.convert("RGB")).copy()
    field_colors = class_colors([f.get("field_classification") or f.get("label") or "unknown" for f in fields])
    for f in fields:
        evidence = f.get("selection_status", f.get("agricultural_evidence", {}).get("status", "unverified"))
        color = (255, 214, 10) if evidence == "supported" else (
            (255, 59, 48) if evidence.startswith("rejected") else (255, 149, 0))
        if by_class:
            color = field_colors[f.get("field_classification") or f.get("label") or "unknown"]
        window, pixels = polygon_pixels(f["geometry"], image.size)
        target = rgb[window]
        target[pixels] = (target[pixels] * 0.78 + np.array(color) * 0.22).astype("uint8")
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = _font(max(12, int(min(image.size) / 65)))
    for f in fields:
        geometry = f["geometry"]
        color = field_colors[f.get("field_classification") or f.get("label") or "unknown"] if by_class else (255, 214, 10)
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            for ring in polygon:
                draw.line([tuple(p[:2]) for p in ring], fill=color,
                          width=max(1, int(min(image.size) / 400)))
        x, y, _, _ = geometry_bounds(geometry)
        label = f.get("id", f.get("source_id", ""))
        draw.text((x+2, y+2), label, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return canvas


def seg_color_map(mask: np.ndarray, colors: List[List[int]]) -> Image.Image:
    palette = np.array(colors, dtype=np.uint8)
    return Image.fromarray(palette[np.clip(mask, 0, len(palette) - 1)])


def seg_overlay(image: Image.Image, mask: np.ndarray, colors: List[List[int]], alpha=0.55) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.asarray(seg_color_map(mask, colors), dtype=np.float32)
    if color.shape[:2] != rgb.shape[:2]:
        color = np.asarray(Image.fromarray(color.astype(np.uint8)).resize(image.size, Image.NEAREST),
                           dtype=np.float32)
    return Image.fromarray(((1 - alpha) * rgb + alpha * color).astype(np.uint8))


def add_legend(image: Image.Image, names: List[str], colors: List[List[int]],
               coverage: Dict[str, float] = None) -> Image.Image:
    """Legend strip to the right: colour swatch + class name (+ % coverage)."""
    h = image.size[1]
    font = _font(max(14, int(h / 45)))
    row = int(font.size * 1.6) if hasattr(font, "size") else 22
    lines = ["%s  %.1f%%" % (n, coverage.get(n, 0.0)) if coverage else n for n in names]
    probe = ImageDraw.Draw(image)
    text_w = max(probe.textbbox((0, 0), t, font=font)[2] for t in lines)
    strip_w = text_w + row * 2
    canvas = Image.new("RGB", (image.size[0] + strip_w, max(h, row * (len(lines) + 1))), "white")
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    x0 = image.size[0] + row // 2
    for i, (text, color) in enumerate(zip(lines, colors)):
        y = row // 2 + i * row
        draw.rectangle([x0, y, x0 + row - 6, y + row - 6], fill=tuple(color), outline=(0, 0, 0))
        draw.text((x0 + row + 2, y), text, fill=(0, 0, 0), font=font)
    return canvas


def classification_comparison(original: Image.Image, map_with_legend: Image.Image) -> Image.Image:
    """Presentation only: equal native pixel scale for original and class map."""
    font = _font(max(14, original.height // 45))
    header = max(28, int(getattr(font,"size",16)*2))
    gap = max(12, original.width // 100)
    canvas = Image.new("RGB", (original.width + gap + map_with_legend.width,
        max(original.height,map_with_legend.height) + header), "white")
    canvas.paste(original.convert("RGB"),(0,header))
    canvas.paste(map_with_legend.convert("RGB"),(original.width+gap,header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8,4),"Original image",fill="black",font=font)
    draw.text((original.width+gap+8,4),"Classification map",fill="black",font=font)
    return canvas


def fit(image: Image.Image, max_side: int) -> Image.Image:
    """A copy no larger than max_side (display only -- never sent to a model)."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def draw_tile_grid(image: Image.Image, tiles, highlight: int = None, max_side: int = 1600,
                   labels: Dict[int, str] = None) -> Image.Image:
    """The scene shrunk to max_side with every tile window outlined and numbered.
    `image` may be the scene or any smaller copy of it (tile windows are in scene
    pixels and are scaled to whatever is passed). `highlight` (a tile index): that
    tile is outlined in yellow and the rest is dimmed. `labels`: optional extra
    text per tile index (e.g. a status icon)."""
    small = fit(image.convert("RGB"), max_side)
    scene_w, scene_h = (tiles[0].scene_w, tiles[0].scene_h) if tiles else image.size
    sx, sy = small.size[0] / scene_w, small.size[1] / scene_h
    canvas = small.convert("RGBA")
    shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shade)
    font = _font(max(12, int(min(small.size) / 18)))
    width = max(2, int(min(small.size) / 250))

    def px(t):
        return (round(t.x0 * sx), round(t.y0 * sy), round(t.x1 * sx) - 1, round(t.y1 * sy) - 1)

    if highlight is not None:
        draw.rectangle([0, 0, canvas.size[0], canvas.size[1]], fill=(0, 0, 0, 140))
        for t in tiles:
            if t.index == highlight:
                draw.rectangle(px(t), fill=(0, 0, 0, 0))          # punch the tile back out
    for t in tiles:
        on = t.index == highlight
        draw.rectangle(px(t), outline=(255, 214, 10, 255) if on else (255, 255, 255, 230),
                       width=width * 2 if on else width)
    for t in tiles:
        x1, y1, x2, y2 = px(t)
        text = str(t.index) + (" " + labels[t.index] if labels and t.index in labels else "")
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left, bottom - top
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pad = max(4, th // 3)
        draw.rounded_rectangle([cx - tw // 2 - pad, cy - th // 2 - pad, cx + tw // 2 + pad,
                                cy + th // 2 + pad], radius=pad,
                               fill=(255, 214, 10, 235) if t.index == highlight else (0, 0, 0, 170))
        draw.text((cx - tw // 2 - left, cy - th // 2 - top), text, font=font,
                  fill=(0, 0, 0, 255) if t.index == highlight else (255, 255, 255, 255))
    return Image.alpha_composite(canvas, shade).convert("RGB")
