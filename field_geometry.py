"""Field geometry and semantic evidence on the original, unresampled pixel grid.

Pixel polygons use x right/y down, pixel edges, including holes and MultiPolygons.
Only export applies the source affine. Non-georeferenced imagery has no world CRS.
"""

import json
import math
import os

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.features import rasterize


def read_image(path):
    """RGB view of original pixels; TIFF retains CRS/affine through source_path.

    Non-byte rasters use a valid-pixel 1/99 percentile stretch, as Delineate does.
    """
    import config
    if str(path).lower().endswith((".tif", ".tiff")):
        with rasterio.open(path) as src:
            bands = src.read(config.RGB_BANDS, masked=True)
        valid = ~np.ma.getmaskarray(bands).any(axis=0)
        valid &= np.isfinite(bands.data).all(axis=0)
        rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
        for i, band in enumerate(bands.data):
            if band.dtype == np.uint8:
                rgb[..., i] = band
            elif valid.any():
                lo, hi = np.percentile(band[valid], (1, 99))
                rgb[..., i] = (np.clip((band - lo) / max(float(hi - lo), 1), 0, 1) * 255).astype("uint8")
        rgb[~valid] = 0
        image = Image.fromarray(rgb)
    else:
        with Image.open(path) as src:
            image = src.convert("RGB")
    image.info["source_path"] = os.path.abspath(path)
    return image


def source_metadata(image_size, source_path=None):
    w, h = image_size
    meta = {"width": w, "height": h, "coordinate_reference": "pixel",
            "crs_wkt": None, "transform": list(Affine.identity())[:6]}
    if source_path and str(source_path).lower().endswith((".tif", ".tiff")):
        with rasterio.open(source_path) as src:
            if (src.width, src.height) != (w, h):
                raise ValueError("Original source and analysed image dimensions differ")
            meta.update(transform=list(src.transform)[:6],
                        crs_wkt=src.crs.to_wkt() if src.crs else None,
                        coordinate_reference="geographic" if src.crs else "pixel")
    return meta


def transform_geometry(geometry, transform):
    """Affine-only coordinate conversion; no simplification, clipping or redraw."""
    if geometry["type"] not in ("Polygon", "MultiPolygon"):
        raise ValueError("Only polygon and multipolygon field geometries are supported")
    def convert(coords):
        if len(coords) >= 2 and isinstance(coords[0], (int, float)):
            return list(transform * (coords[0], coords[1]))
        return [convert(c) for c in coords]
    return {"type": geometry["type"], "coordinates": convert(geometry["coordinates"])}


def geometry_bounds(geometry):
    pts = []
    def walk(coords):
        if len(coords) >= 2 and isinstance(coords[0], (int, float)):
            if not all(math.isfinite(float(n)) for n in coords[:2]):
                raise ValueError("Non-finite field coordinates")
            pts.append(coords[:2])
        else:
            for c in coords:
                walk(c)
    if geometry.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("Field geometry must be Polygon or MultiPolygon")
    walk(geometry["coordinates"])
    if not pts:
        raise ValueError("Empty field geometry")
    x, y = zip(*pts)
    return [min(x), min(y), max(x), max(y)]


def polygon_pixels(geometry, size):
    """Window + bool mask, using pixel centres. Holes and multipart survive."""
    w, h = size
    x1, y1, x2, y2 = geometry_bounds(geometry)
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
    if x2 <= x1 or y2 <= y1:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), bool)
    pixels = rasterize([(geometry, 1)], out_shape=(y2-y1, x2-x1),
                       transform=Affine.translation(x1, y1), dtype="uint8").astype(bool)
    return (slice(y1, y2), slice(x1, x2)), pixels


def field_mask(fields, size, instance=False):
    out = np.zeros((size[1], size[0]), dtype="uint32" if instance else bool)
    for field in fields:
        window, pixels = polygon_pixels(field["geometry"], size)
        out[window][pixels] = int(field["id"][1:]) + 1 if instance else True
    return out


def validate_class_mapping(seg):
    expected = [c.split(",")[0].strip() for c in seg.classes]
    if seg.class_names != expected or len(set(expected)) != len(expected):
        raise ValueError("Ambiguous or inconsistent SegEarth output class-ID mapping")
    if seg.mask.ndim != 2 or not np.issubdtype(seg.mask.dtype, np.integer):
        raise ValueError("LCLU must be a 2D integer class-ID raster")
    if seg.mask.min() < 0 or seg.mask.max() >= len(expected):
        raise ValueError("LCLU values lie outside the verified SegEarth mapping")
    return {name: i for i, name in enumerate(seg.class_names)}


def native_filter_config(seg):
    import config
    mapping = validate_class_mapping(seg)
    # These are semantic categories, NOT assumed class numbers. Verify every one.
    names = ("buildings", "road", "urban green space", "sports field", "industrial area", "water")
    if any(n not in mapping for n in names):
        raise ValueError("Field delineation requires the complete V4 LULC vocabulary")
    return {"range": len(seg.class_names),
            "filter_classes": [mapping[n] for n in names],
            "class_id_mapping": mapping,
            "remaining_area_threshold": config.DELINEATE_REMAINING_AREA}


def verify_fields(raw_fields, seg):
    import config
    cfg = native_filter_config(seg)
    mapping = cfg["class_id_mapping"]
    w, h = seg.mask.shape[1], seg.mask.shape[0]
    checked, kept, rejected = [], [], []
    for field in raw_fields:
        item = dict(field)
        window, inside = polygon_pixels(item["geometry"], (w, h))
        values = seg.mask[window][inside]
        counts = np.bincount(values, minlength=len(seg.class_names))
        total = int(counts.sum())
        fractions = counts / max(total, 1)
        nonag = float(sum(fractions[i] for i in cfg["filter_classes"]))
        crop = float(fractions[mapping["cropland"]])
        agricultural_cover = crop + sum(float(fractions[mapping[n]]) for n in ("orchard", "vineyard") if n in mapping)
        possible = sum(float(fractions[mapping[n]]) for n in
                       ("bareland", "grass", "meadow", "tree") if n in mapping)
        status = ("rejected_nonagricultural" if nonag >= config.FIELD_NONAG_REJECT_FRACTION
                  else "supported" if agricultural_cover >= config.FIELD_CROPLAND_SUPPORT_FRACTION
                  else "uncertain")
        if not total:
            status = "rejected_empty"
        bounds = geometry_bounds(item["geometry"])
        item.update(bbox=[bounds[0]/w, bounds[1]/h, bounds[2]/w, bounds[3]/h],
                    bbox_px=bounds, area_px=total,
                    semantic_composition={n: {"class_id": i, "pixels": int(counts[i]),
                                              "fraction": float(fractions[i])}
                                          for i, n in enumerate(seg.class_names)},
                    agricultural_evidence={"status": status, "cropland_fraction": crop,
                        "agricultural_cover_fraction": agricultural_cover,
                        "possible_agricultural_fraction": possible, "nonag_fraction": nonag,
                        "note": "Class fractions are evidence, not calibrated probabilities. "
                                "Pasture, fallow soil and tree cover remain eligible."})
        checked.append(item)
        if status.startswith("rejected"):
            rejected.append(item)
        else:
            kept.append(item)
    return checked, kept, rejected


def write_lclu(seg, path, metadata):
    validate_class_mapping(seg)
    if seg.mask.shape != (metadata["height"], metadata["width"]):
        raise ValueError("LCLU dimensions differ from the source scene")
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype="uint8",
                       width=metadata["width"], height=metadata["height"],
                       crs=metadata["crs_wkt"], transform=Affine(*metadata["transform"]),
                       compress="deflate") as dst:
        dst.write(seg.mask.astype("uint8"), 1)
        dst.update_tags(class_id_mapping=json.dumps(dict(enumerate(seg.class_names))),
                        coordinate_reference=metadata["coordinate_reference"])
    return path


def save_raster(mask, path, metadata):
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype=str(mask.dtype),
                       width=metadata["width"], height=metadata["height"],
                       crs=metadata["crs_wkt"], transform=Affine(*metadata["transform"]),
                       compress="deflate") as dst:
        dst.write(mask, 1)
        dst.update_tags(coordinate_reference=metadata["coordinate_reference"],
                        instance_values="fN = N+1; 0 = background")
