"""Isolated upstream Delineate V2 inference and GeoPackage export.

Upstream assumes north-up projected imagery. Inference runs on a private pixel
grid (1 unit/pixel), with ORIGINAL RGB pixels and aligned LCLU; public outputs
are converted back to source CRS/affine, or explicitly local pixel coordinates.
No upstream code, checkpoint or configuration file is changed.
"""

import json
import os
import sys
import time
from pathlib import Path

from protocol import serve

# QGIS Python is supported as a GDAL provider on Windows. Spawned upstream
# children execute this module before unpickling; DLL paths must exist then.
_DLL_HANDLES = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    _qgis_root = Path(sys.base_prefix).parent.parent
    if list((_qgis_root / "bin").glob("gdal*.dll")):
        for _dll_dir in (_qgis_root / "bin", _qgis_root / "apps/qgis/bin", _qgis_root / "apps/Qt5/bin"):
            if _dll_dir.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
                os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")
        _gdal_data = _qgis_root / "apps/gdal/share/gdal"
        if _gdal_data.is_dir():
            os.environ.setdefault("GDAL_DATA", str(_gdal_data))

SETTINGS = {}


def init(settings):
    SETTINGS.clear()
    SETTINGS.update(settings)
    upstream = Path(settings["upstream"])
    checkpoint = Path(settings["checkpoint"])
    for path in (upstream / "methods/main/inference.py", upstream / "conf_sample.yaml", checkpoint):
        if not path.is_file():
            raise FileNotFoundError("Delineate required file missing: %s" % path)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    os.environ.setdefault("YOLO_CONFIG_DIR", settings["cache_dir"])
    from osgeo import gdal, ogr
    import rasterio
    import shapely
    import numba
    import ultralytics
    gdal.UseExceptions()
    ogr.UseExceptions()
    return {"model": "Delineate Anything V2", "checkpoint": str(checkpoint),
            "backend": "upstream full-scene overlapping tiles + native seam merging",
            "gdal": gdal.VersionInfo(), "ultralytics": ultralytics.__version__,
            "shapely": shapely.__version__}


def _write_staged(path, data, transform):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=data.shape[0], dtype="uint8", crs="EPSG:3857", transform=transform,
                       compress="deflate") as dst:
        dst.write(data)
        dst.update_tags(INTERNAL_COORDINATES="synthetic pixel grid; never geographic evidence")


def _infer(args):
    import numpy as np
    import rasterio
    import yaml
    from affine import Affine
    from PIL import Image
    from osgeo import ogr
    from methods.main.inference import execute

    root = Path(args["run_dir"]) / "delineate_internal"
    inputs, temp = root / "image", root / "temp"
    inputs.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(args["image_path"]).convert("RGB"))
    h, w = rgb.shape[:2]
    grid = Affine(1, 0, 0, 0, -1, h)
    _write_staged(inputs / "original_rgb.tif", rgb.transpose(2, 0, 1), grid)
    with rasterio.open(args["lclu_path"]) as src:
        lclu = src.read(1)
    if lclu.shape != (h, w):
        raise ValueError("Delineate RGB/LCLU grids differ")
    cfg = args["filter_config"]
    if lclu.min() < 0 or lclu.max() >= cfg["range"]:
        raise ValueError("LCLU IDs outside native filter range")
    staged_lclu = root / "lclu.tif"
    _write_staged(staged_lclu, lclu[None].astype("uint8"), grid)

    with open(Path(SETTINGS["upstream"]) / "conf_sample.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config["model"] = ["large_v2"]
    config["super_resolution"] = 1
    config["treat_as_vrt"] = False
    config["mask_info"] = {"range": cfg["range"], "filter_classes": cfg["filter_classes"],
                           "clip_classes": []}  # required upstream key; clipping disabled
    config["background_info"] = {"background_classes_from_mask": [], "additional_source": None}
    config["data_loader"].update(bands=[3, 2, 1], nodata_value=None, nodata_band=None,
                                 min=[0, 0, 0], max=[255, 255, 255])
    config["postprocess_limits"]["num_workers"] = [1, SETTINGS["post_workers"]]
    for pas in config["passes"]:
        pas["batch_size"] = SETTINGS["batch_size"]
        for model in pas["model_args"]:
            model["minimal_confidence"] = SETTINGS["confidence"]
        pas["delineation_config"].update(
            remaining_area_threshold=cfg["remaining_area_threshold"],
            pixel_area_threshold=SETTINGS["min_area_px"],
            merge_tile_seams=True, merge_region_seams=True)
    # Postprocessing thresholds in the private grid are pixel areas, never metres.
    config["filtering_args"].update(automatic_area_scale=False,
        minimum_area_m2=SETTINGS["min_area_px"], minimum_part_area_m2=0, minimum_hole_area_m2=0)
    config["simplification_args"]["simplify"] = False
    config["giant_filter"]["enabled"] = False  # relies on real metre scale
    native_path = root / "native_fields.gpkg"
    config["execution_args"] = {"src_folder": str(inputs), "temp_folder": str(temp),
        "output_path": str(native_path), "keep_temp": False, "mask_filepath": str(staged_lclu)}
    with open(root / "effective_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    execute([SETTINGS["checkpoint"]], config, False)
    ds = ogr.Open(str(native_path))
    if ds is None:
        raise RuntimeError("Delineate did not produce its expected GeoPackage")
    layer = ds.GetLayerByName("fields")
    if layer is None:
        raise RuntimeError("Delineate GeoPackage has no fields layer")
    # Import this light module from the agent without importing its configuration.
    from field_geometry import transform_geometry, geometry_bounds
    items = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        pixel_geom = transform_geometry(json.loads(geom.ExportToJson()), ~grid)
        items.append({"native_id": feature.GetFID(), "geometry": pixel_geom,
                      "coordinate_reference": "pixel"})
    ds = None
    items.sort(key=lambda f: (geometry_bounds(f["geometry"])[1], geometry_bounds(f["geometry"])[0],
                              f["native_id"]))
    for i, field in enumerate(items):
        field["id"] = "f%d" % i
    proposals_path = root / "native_proposals.json"
    proposals_path.write_text(json.dumps(items), encoding="utf-8")
    return {"fields_path": str(proposals_path), "native_gpkg_path": str(native_path),
            "effective_config_path": str(root / "effective_config.json"), "count": len(items)}


def _export(args):
    from affine import Affine
    from osgeo import ogr, osr
    from field_geometry import transform_geometry

    payload = json.loads(Path(args["payload_path"]).read_text(encoding="utf-8"))
    meta = payload["metadata"]
    path = args["output_path"]
    driver = ogr.GetDriverByName("GPKG")
    if os.path.exists(path):
        driver.DeleteDataSource(path)
    ds = driver.CreateDataSource(path)
    srs = osr.SpatialReference()
    if meta.get("crs_wkt"):
        srs.ImportFromWkt(meta["crs_wkt"])
    else:
        srs.ImportFromWkt('LOCAL_CS["Image pixel coordinates (x right, y down)",'
                          'LOCAL_DATUM["Image",0],UNIT["pixel",1],AXIS["x",EAST],AXIS["y",SOUTH]]')
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    for name, fields in payload["layers"].items():
        layer = ds.CreateLayer(name, srs, geom_type=ogr.wkbMultiPolygon)
        for fname, ftype in (("field_id", ogr.OFTString), ("area_px", ogr.OFTInteger64),
                             ("evidence", ogr.OFTString), ("semantics", ogr.OFTString),
                             ("coord_ref", ogr.OFTString), ("vlm_status", ogr.OFTString),
                             ("class_name", ogr.OFTString)):
            layer.CreateField(ogr.FieldDefn(fname, ftype))
        layer.StartTransaction()
        for field in fields:
            geom = field["geometry"]
            if meta.get("crs_wkt"):
                geom = transform_geometry(geom, Affine(*meta["transform"]))
            ogr_geom = ogr.CreateGeometryFromJson(json.dumps(geom))
            if ogr_geom.GetGeometryType() == ogr.wkbPolygon:
                ogr_geom = ogr.ForceToMultiPolygon(ogr_geom)
            feat = ogr.Feature(layer.GetLayerDefn())
            feat.SetGeometry(ogr_geom)
            feat.SetField("field_id", field["id"])
            feat.SetField("area_px", int(field.get("area_px", 0)))
            feat.SetField("evidence", field.get("agricultural_evidence", {}).get("status", "unverified"))
            feat.SetField("semantics", json.dumps(field.get("semantic_composition", {})))
            feat.SetField("coord_ref", meta["coordinate_reference"])
            feat.SetField("vlm_status", field.get("selection_status", "unselected"))
            feat.SetField("class_name", field.get("field_classification", ""))
            layer.CreateFeature(feat)
        layer.CommitTransaction()
    ds = None
    return {"gpkg_path": path}


def run(args):
    # Used by multiprocessing children too; original repositories stay read-only.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    t0 = time.time()
    out = _export(args) if args.get("action") == "export" else _infer(args)
    out["seconds"] = round(time.time()-t0, 2)
    return out


if __name__ == "__main__":
    serve(init, run)
