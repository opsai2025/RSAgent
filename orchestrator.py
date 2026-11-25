"""
The agent loop -- one plain, sequential function. No LangGraph: there is one
decision (the plan) followed by a fixed order of steps, so a graph would add
scaffolding without adding behaviour.

    run_agent(image, question) -> AgentResult

    V4: field requests run full-scene SegEarth -> native Delineate + semantic
    verification -> contextual tools -> VLM -> original polygon exports/masks.
    Other requests retain the V3 path and its external tiling behavior below.

    1. PLAN      planner.make_plan(question)            router LLM (text only) + rules
    2. TOOLS     detection  (LAE-DINO, LAE env)          only if planned
                 segmentation (SegEarth-OV3, SAM env)    only if planned
                 buildings (UniBuilding, own env)        only if planned
    3. DECIDE    default           -> vlm.answer(image, question, all tool outputs):
                                      the VLM decides, aligned with the tool outputs
                 plain LULC / segmentation map of land-cover classes (nothing to
                 find or select) -> the map IS the answer, VLM not called
    4. SHOW      plan.output_kind (router's choice) renders that ONE decision:
                 "boxes" -> the kept objects' bounding boxes (drawn by the UI)
                 "mask"  -> a binary mask from the tools' pixels (masks.py); if kept
                            detections need object pixels the main segmentation
                            doesn't have, ONE extra SegEarth-OV3 pass runs here, after
                            the VLM, on those classes only -- never in the plan
Every step is recorded in result.steps, which the UI shows as the tool trace.
A failed tool does not stop the run: the VLM still answers from the image and
the failure is visible in the trace.

V02 -- large scenes:

    run_tiled(image, question) -> TiledRun

    0. TILE      tiling.make_tiles(): both sides <= config.TILE_SIZE -> one tile
                 (the image itself, unsplit); otherwise tiles of at most
                 TILE_SIZE px, never padded (tiling.py explains the strategies)
    1. PLAN      once for all tiles (the router reads only the question), unless
                 config.TILE_SHARED_PLAN is off
    2-4.         run_agent() on EACH tile independently -- its own tools, its own
                 VLM call (told which tile of the scene it sees), its own mask,
                 its own runs/<ts>/tile_NN/ folder with input.png + trace.json.
    Tile results are NOT merged (this development stage); runs/<ts>/tiles.json
    records only the tile geometry, so a later stage can map tile boxes back to
    the scene (Tile.to_scene_px).
"""

import dataclasses
import json
import os
import time
import shutil
import colorsys
from pathlib import Path

from PIL import Image
import numpy as np

import config
import masks
import planner
import tiling
import vlm
import drawing
from schemas import AgentResult, Plan, Step, TiledRun, Tile, DetectionOutput, SegmentationOutput, FieldOutput
from tools import detection, segmentation, unibuilding, delineate


def _new_run_dir(image, run_dir=None):
    if run_dir is None:
        stamp = "%s_%03d" % (time.strftime("%Y%m%d_%H%M%S"), int(time.time() * 1000) % 1000)
        run_dir = os.path.join(config.RUNS_DIR, stamp)
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    image_path = os.path.join(run_dir, "input.png")
    image.convert("RGB").save(image_path)
    return run_dir, image_path


def _plan_detail(p: Plan) -> str:
    scope = p.request.land_cover_scope if p.request else "legacy"
    display = "classification_map" if p.output == "tool_output" and p.output_kind == "boxes" else p.output_kind
    return "tools: %s · %s · shown as %s · semantic scope: %s" % (
        ", ".join(p.tools) or "none",
        "plain map, no VLM" if p.output == "tool_output" else "VLM decides",
        display, scope)


def _timed(result, name, fn, detail_fn, notify=None):
    if notify is not None:
        notify(name)          # UI progress: the step that is about to run
    t0 = time.time()
    try:
        out = fn()
        result.steps.append(Step(name, detail_fn(out), round(time.time() - t0, 1)))
        return out
    except Exception as exc:  # noqa: BLE001
        result.steps.append(Step(name, str(exc), round(time.time() - t0, 1), status="error"))
        return None


def tool_output_summary(result: AgentResult) -> str:
    """Deterministic text for tool_output mode (no VLM involved)."""
    lines = []
    seg = result.segmentation
    if seg is not None:
        lines.append("SegEarth-OV3 semantic pixel coverage:")
        lines += ["- %s: %.1f%%" % (k, v) for k, v in
                  sorted(seg.coverage.items(), key=lambda kv: kv[1], reverse=True)]
        if seg.building_class:
            lines.append("Building components (connected regions ≥ %d px): %d"
                         % (config.MIN_BUILDING_AREA_PX, len(seg.buildings)))
    det = result.detection
    uni = result.unibuilding
    if uni is not None:
        lines.append("UniBuilding: %d building components, %.1f%% building pixels. "
                     "Touching roofs may merge into one component."
                     % (len(uni.buildings), uni.coverage))
    if det is not None:
        lines.append("LAE-DINO detections after NMS: " + (", ".join(
            "%s × %d" % kv for kv in det.counts().items()) or "none"))
    if result.fields is not None:
        supported = sum(f["agricultural_evidence"]["status"] == "supported" for f in result.fields.fields)
        lines.append("Field proposals before VLM selection: %d semantically supported, %d uncertain; "
                     "%d rejected by polygon semantic checks. Candidate geometry alone does not "
                     "confirm agriculture." % (supported, len(result.fields.fields)-supported,
                                              len(result.fields.rejected)))
    return "\n".join(lines) or "The tool produced no output."


def _detection_step(result, image_path, plan, score_thr, nms_iou, notify):
    result.detection = _timed(
        result, "LAE-DINO detection",
        lambda: detection.detect(image_path, plan.detection_classes,
                                 score_thr if score_thr is not None else config.DET_SCORE_THR,
                                 nms_iou if nms_iou is not None else config.DET_NMS_IOU),
        lambda d: "classes %s · %d raw → %d after NMS"
                  % (d.classes, len(d.raw), len(d.nms)), notify)


def _unibuilding_step(result, image_path, notify):
    result.unibuilding = _timed(result, "UniBuilding building detection",
        lambda: unibuilding.detect(image_path, result.run_dir),
        lambda u: "%d building components · %.1f%% building pixels" % (len(u.buildings),u.coverage), notify)


def _guard_semantic_pixels(result, raw):
    """Fuse on the original grid before regions, coverage, VLM or Delineate.

    The original RGB is never blacked out/resampled. SegEarth supplies non-building
    classes only. UniBuilding is the sole building source; focused products leave
    its pixels at background, while a field evidence scheme includes its class ID.
    """
    uni, classes = result.unibuilding, result.plan.segmentation_classes
    if uni is None or uni.mask.shape != raw.mask.shape:
        raise ValueError("A source-aligned UniBuilding mask is required for semantic classification")
    names = [c.split(",")[0].strip() for c in classes]
    mask = np.zeros(raw.mask.shape,np.uint8)
    colors = [[0,0,0] for _ in classes]
    building_idx = next((i for i,c in enumerate(classes) if i and segmentation.is_building_class(c)),None)
    for i, entry in enumerate(raw.classes[1:],start=1):
        if segmentation.is_building_class(entry):
            continue
        name = entry.split(",")[0].strip()
        if name not in names:
            raise ValueError("Semantic inference returned a class outside the compiled scope: " + name)
        target = names.index(name)
        mask[raw.mask == i] = target
        colors[target] = list(raw.colors[i])
    changed = int(np.count_nonzero((mask != 0) & uni.mask))
    mask[uni.mask] = building_idx if building_idx is not None else 0
    if building_idx is not None:
        colors[building_idx] = [166,124,60]
    path = os.path.join(result.run_dir,"seg_mask.png")
    Image.fromarray(mask).save(path)
    buildings, regions, labels, ids = segmentation.split_into_regions(mask,names,building_idx)
    result.semantic_building_guard = {
        "method": "original RGB inference followed by native-grid class-ID fusion",
        "building_source": "UniBuilding", "building_mask_path": uni.mask_path,
        "raw_semantic_mask_path": raw.mask_path, "protected_semantic_mask_path": path,
        "building_pixels": int(uni.mask.sum()), "nonbuilding_pixels_removed": changed,
        "semantic_building_id": building_idx,
        "focused_output": "building pixels remain background unless buildings are requested"}
    with open(os.path.join(result.run_dir,"semantic_building_guard.json"),"w",encoding="utf-8") as fh:
        json.dump(result.semantic_building_guard,fh,indent=2)
    return SegmentationOutput(classes,names,colors,
        {name:float(np.mean(mask == i)*100) for i,name in enumerate(names)},mask,path,
        buildings,regions,names[building_idx] if building_idx is not None else None,
        labels,ids,raw.seconds)


def _semantic_step(result, image_path, notify):
    plan = result.plan
    if plan.protect_semantic_buildings and result.unibuilding is None:
        result.steps.append(Step("SegEarth-OV3 segmentation",
            "blocked: required UniBuilding exclusion mask is unavailable",status="error"))
        return
    inference_classes = [c for c in plan.segmentation_classes
                         if not plan.protect_semantic_buildings or not segmentation.is_building_class(c)]
    raw = _timed(result,"SegEarth-OV3 segmentation",
        lambda: segmentation.segment(image_path,inference_classes,result.run_dir,
            **({"out_name":"seg_model_mask.png"} if plan.protect_semantic_buildings else {})),
        lambda s: "classes %s · original RGB input" % s.class_names[1:],notify)
    if raw is None:
        return
    if plan.protect_semantic_buildings:
        result.segmentation = _timed(result,"Semantic building exclusion",
            lambda: _guard_semantic_pixels(result,raw),
            lambda s: "%d UniBuilding pixels protected; %d non-building labels removed; "
                      "coverage and regions recomputed before decisions" % (
                          result.semantic_building_guard["building_pixels"],
                          result.semantic_building_guard["nonbuilding_pixels_removed"]),notify)
    else:
        result.segmentation = raw


def run_agent(image, question, score_thr=None, nms_iou=None, *, run_dir=None, plan=None,
              router_step=None, tile=None, notify=None, source_path=None) -> AgentResult:
    """The V01 agent on ONE image. Called with only the first four arguments it
    behaves exactly as in V01. V02 (run_tiled) also passes:
      run_dir      the tile's own folder (instead of a new runs/<ts>/)
      plan         a plan already made for this question (shared by all tiles),
      router_step  with the router's Step, copied into this result's trace
      tile         the schemas.Tile `image` was cut from: the VLM is told
      notify       callback(step name) before each step, for UI progress
    """
    run_dir, image_path = _new_run_dir(image, run_dir)
    result = AgentResult(question=question, plan=Plan(), run_dir=run_dir, tile=tile)
    result.source_path = source_path or image.info.get("source_path", "") or getattr(image, "filename", "")

    # 1. PLAN ---------------------------------------------------------------
    if plan is None:
        plan = _timed(result, "Router (%s)" % config.ROUTER_MODEL,
                      lambda: planner.make_plan(question), _plan_detail, notify)
        if plan is None:
            # Router unreachable: fall back to the image-only VLM answer, shown as boxes
            plan = Plan(reasoning="router failed -- conservative subject-based fallback")
    else:
        result.steps.append(dataclasses.replace(router_step) if router_step is not None
                            else Step("Router", "plan given by the caller", status="skipped"))
    # All entry points use one policy boundary; supplied/shared plans cannot
    # bypass scope validation or route discrete objects into a semantic map.
    try:
        plan = planner.apply_rules(plan, question)
    except planner.PlanningError as exc:
        result.plan = Plan(reasoning=str(exc))
        result.answer = "The classification subject or class scope could not be resolved. " + str(exc)
        result.answered_by = "Planning needs clarification"
        result.steps.append(Step("Validate request scope", str(exc), status="error"))
        _save_trace(result)
        return result
    result.plan = plan

    # 2. TOOLS --------------------------------------------------------------
    if plan.protect_semantic_buildings:
        _unibuilding_step(result,image_path,notify)
    if plan.use_detection and not plan.use_delineate:
        _detection_step(result, image_path, plan, score_thr, nms_iou, notify)
    if plan.use_segmentation:
        _semantic_step(result,image_path,notify)
    if plan.use_delineate:
        if tile is not None and not tile.is_whole_scene:
            result.steps.append(Step("Delineate V2", "refused: field delineation requires a full scene",
                                     status="error"))
        elif result.segmentation is None:
            result.steps.append(Step("Delineate V2", "SegEarth failed: cannot delineate without semantics",
                                     status="error"))
        else:
            result.fields = _timed(
                result, "Delineate V2 + polygon semantic verification",
                lambda: delineate.delineate(image_path, result.segmentation, run_dir, result.source_path),
                lambda f: "%d native LCLU-filtered candidates; %d eligible, %d semantic rejections; "
                          "filter IDs %s; remaining fraction %.2f" % (
                              len(f.raw_fields), len(f.fields), len(f.rejected),
                              f.filter_config["filter_classes"], f.filter_config["remaining_area_threshold"]), notify)
    if plan.use_detection and plan.use_delineate:
        _detection_step(result, image_path, plan, score_thr, nms_iou, notify)
    if plan.use_unibuilding and not plan.protect_semantic_buildings:
        _unibuilding_step(result,image_path,notify)

    return _decide_and_render(result, image, image_path, notify)


def _field_review_progress(result, event, notify):
    stage = event["stage"]
    if stage == "start":
        if notify:
            notify("VLM field review: %d/%d reviewed; batch of %d" %
                   (event["reviewed"], event["total"], event["fields"]))
        return
    if stage == "policy":
        name, detail, status = "VLM field selection policy", event["detail"], "ok"
    else:
        name = "VLM field batch %s" % event["batch"]
        detail = ("%d fields; %s; %d payload characters" %
                  (event["fields"], event.get("status", stage), event.get("payload_chars", 0)))
        if stage == "complete":
            detail += "; %d/%d reviewed" % (event["reviewed"], event["total"])
        else:
            detail += "; " + event.get("detail", "")
        status = "skipped" if stage == "retry" else "ok"
    result.steps.append(Step(name, detail, event.get("seconds", 0.0), status=status,
                             parent="VLM (%s)" % config.VLM_MODEL))


def _vlm_detail(reply, result, image):
    path = Path(result.run_dir) / "vlm_field_review.json"
    detail = "%d objects in final decision; %d explicit proposal rejections" % (len(reply.objects), len(reply.rejected))
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        return detail + "; %d/%d fields reviewed in bounded requests" % (report["reviewed_fields"], report["total_fields"])
    return detail + " · " + vlm.sent_image_info(image)


def _decide_and_render(result, image, image_path, notify=None):
    plan, question, run_dir, tile = result.plan, result.question, result.run_dir, result.tile
    if plan.protect_semantic_buildings and (result.unibuilding is None or result.segmentation is None):
        result.answer = "The classification could not be completed because required building protection " \
                        "or semantic evidence failed. No verified classification map was produced. See the trace."
        result.answered_by = "Required semantic evidence failed"
        if plan.output_kind in ("mask","classification_map"):
            result.steps.append(Step("Render classification map" if plan.output_kind == "classification_map" else "Mask",
                "not built: required source-aligned building/semantic evidence unavailable",status="skipped"))
        _save_trace(result)
        return result
    # 3. DECIDE ------------------------------------------------------------
    tool_failed = any(s.status == "error" for s in result.steps[1:])
    target_classes = []
    if plan.output == "tool_output" and not tool_failed:
        result.answer = tool_output_summary(result)
        result.answered_by = "Map output — VLM not called (native class pixels)"
        result.steps.append(Step("VLM", "skipped: plain map, nothing to find or select",
                                 status="skipped"))
    else:
        context = tiling.vlm_context(tile) if tile is not None else ""
        if plan.request is not None:
            task = {"operation": plan.request.operation,
                    "targets": [t.model_dump() for t in plan.request.targets],
                    "constraints": plan.request.constraints,
                    "classification_basis": plan.request.classification_basis,
                    "output_classes": plan.request.output_classes}
            context += "\nRESOLVED REQUEST (presentation-independent):\n" + json.dumps(task)
            if plan.request.operation == "classify":
                context += ("\nClassify the requested subjects by the requested class dimension. "
                            "Use each selected proposal's label for its class/type, preserve its ID "
                            "and geometry, and mark ambiguous types as unknown. Context is evidence, "
                            "not an additional target. Only measured proposal IDs can supply class "
                            "geometry; do not invent new boxes. Do not substitute a general land-cover map.")
        reply = _timed(result, "VLM (%s)" % config.VLM_MODEL,
                       lambda: vlm.answer(image, question, result.detection, result.segmentation,
                                          context=context, unibuilding=result.unibuilding,
                                          **({"fields": result.fields, "field_task": True,
                                              "field_target_origins": _target_origins(plan),
                                              "request": plan.request, "run_dir": run_dir,
                                              "progress": lambda event: _field_review_progress(result, event, notify)}
                                             if plan.use_delineate else {})),
                       lambda r: _vlm_detail(r, result, image),
                       notify)
        if reply is not None:
            _enforce_decision_scope(reply, plan)
            for step in reversed(result.steps):
                if step.name == "VLM (%s)" % config.VLM_MODEL:
                    step.detail = _vlm_detail(reply, result, image)
                    break
            result.answer = reply.answer
            result.answered_by = "VLM (%s)" % config.VLM_MODEL
            result.vlm_objects, result.vlm_rejected = reply.objects, reply.rejected
            target_classes = reply.target_classes
        else:
            result.answer = tool_output_summary(result) if (result.detection or result.segmentation or result.unibuilding) \
                else "The VLM call failed -- see the trace."
            result.answered_by = "VLM failed — showing raw tool results"
            review_path = Path(run_dir) / "vlm_field_review.json"
            if review_path.is_file():
                review = json.loads(review_path.read_text(encoding="utf-8"))
                result.answer = (("All field proposals received review decisions, but finalization did not complete. "
                                  if review["reviewed_fields"] == review["total_fields"] else "Field review is incomplete. ") +
                    "%d/%d reviewed; successful batch reviews are saved. No complete final selection is claimed.\n"
                    % (review["reviewed_fields"], review["total_fields"])) + result.answer

    if plan.use_delineate:
        if result.fields is None:
            result.vlm_objects = []
            target_classes = []
            result.answer = "Field delineation could not be completed: required semantic or geometric " \
                            "evidence is unavailable. See the tool trace.\n" + result.answer
        else:
            _timed(result, "Field polygon export",
                   lambda: delineate.export(result.fields, result.vlm_objects, run_dir),
                   lambda path: "Source-aligned GeoPackage + binary/instance rasters: " + path, notify)

    # 4. SHOW -- boxes are drawn by the UI from result.vlm_objects; a mask is built here
    if plan.output_kind == "mask":
        _build_mask(result, plan, image, image_path, target_classes, notify)
    elif plan.output_kind == "classification_map" or (plan.output == "tool_output" and plan.output_kind == "boxes"):
        if any(s.name.startswith("VLM") and s.status == "error" for s in result.steps):
            result.steps.append(Step("Render classification map", "not built: decision failed", status="skipped"))
        else:
            _timed(result, "Render classification map",
                   lambda: _render_classification_map(result, image, target_classes, image_path, notify),
                   lambda path: result.output_caption + " · " + path, notify)

    _save_trace(result)
    return result


def run_tiled(image, question, score_thr=None, nms_iou=None, tile_size=None, strategy=None,
              notify=None, *, source_path=None) -> TiledRun:
    """V02 entry point: cut the scene into tiles (or not, if it is small enough)
    and run the agent on each tile independently.

    notify(tile_index, tile_count, step_name) is called before every step.
    """
    t0 = time.time()
    tile_size = tile_size or config.TILE_SIZE
    strategy = strategy or config.TILE_STRATEGY
    tiles = tiling.make_tiles(image.size, tile_size, strategy)
    # Interpret before deciding scene geometry. Typed agricultural intent may
    # be expressed in any language, without matching a field keyword regex.
    first_plan, first_router_step = _shared_plan(question, len(tiles),
        (lambda step: notify(1, len(tiles), step)) if notify else None)
    try:
        first_plan = planner.apply_rules(first_plan, question)
    except planner.PlanningError:
        # run_agent records a reviewable planning error and runs no model tools.
        pass
    if first_plan.use_delineate:
        tiles = tiling.make_tiles(image.size, max(image.size), strategy)
        strategy = "full_scene_native_delineate"
    if first_router_step.status == "ok" and (len(tiles) == 1 or not config.TILE_SHARED_PLAN):
        first_router_step.detail = _plan_detail(first_plan) + (
            " · single scene plan" if len(tiles) == 1 else " · each tile is planned independently")
    run = TiledRun(question=question, scene_size=tuple(image.size), tile_size=tile_size,
                   strategy=strategy, tiles=tiles)

    def progress(tile):
        return (lambda step: notify(tile.index, tile.count, step)) if notify else None

    if len(tiles) == 1:
        # small enough: the image is processed whole, exactly as in V01
        result = run_agent(image, question, score_thr, nms_iou, tile=tiles[0],
                           notify=progress(tiles[0]), source_path=source_path,
                           plan=first_plan, router_step=first_router_step)
        if first_plan.use_delineate:
            result.steps.append(Step("Scene policy", "External agent tiling bypassed for field requests; "
                "Delineate uses overlapping native tiles and cross-tile/region seam merging."))
            _save_trace(result)
        run.results.append(result)
        run.run_dir = result.run_dir
    else:
        run.run_dir, _ = _new_run_dir(image)          # the full scene, for reference
        digits = max(2, len(str(len(tiles))))
        for tile in tiles:
            tile_dir = os.path.join(run.run_dir, "tile_%0*d" % (digits, tile.index))
            try:
                plan = first_plan if config.TILE_SHARED_PLAN or tile.index == 1 else None
                router_step = first_router_step if plan is not None else None
                result = run_agent(tiling.crop(image, tile), question, score_thr, nms_iou,
                                   run_dir=tile_dir, plan=plan, router_step=router_step,
                                   tile=tile, notify=progress(tile))
            except Exception as exc:  # noqa: BLE001 -- one bad tile must not lose the others
                result = AgentResult(question=question, plan=plan or Plan(), run_dir=tile_dir,
                                     tile=tile, answer="This tile could not be processed: %s" % exc,
                                     answered_by="error")
                result.steps.append(Step(tile.name, str(exc), status="error"))
            run.results.append(result)
    run.seconds = round(time.time() - t0, 1)
    _save_manifest(run)
    return run


def _shared_plan(question, n_tiles, notify):
    """One router call for the whole scene (the router never sees the image)."""
    holder = AgentResult(question=question, plan=Plan())
    plan = _timed(holder, "Router (%s)" % config.ROUTER_MODEL,
                  lambda: planner.make_plan(question), _plan_detail, notify)
    step = holder.steps[0]
    step.detail += (" · planned once, shared by all %d tiles (the router reads only the question)"
                    % n_tiles)
    if plan is None:
        plan = Plan(reasoning="router failed -- conservative subject-based fallback")
    return plan, step


def resume_decision(source_dir, destination=None, notify=None):
    """Replay saved inference through current VLM/selection/rendering only.

    The original run is preserved. Successful bounded reviews are copied for
    signature-checked reuse, so a transient failure need not repeat completed API work.
    """
    source = Path(source_dir).resolve()
    if destination is not None and Path(destination).resolve() == source:
        raise ValueError("Use a different destination to preserve the original run")
    trace = json.loads((source / "trace.json").read_text(encoding="utf-8"))
    image = Image.open(source / "input.png").convert("RGB")
    run_dir, image_path = _new_run_dir(image, destination)
    destination = Path(run_dir)
    cached_plan = Plan.model_validate(trace["plan"])
    plan = planner.apply_rules(cached_plan, trace["question"])
    if (plan.tools != cached_plan.tools or plan.segmentation_classes != cached_plan.segmentation_classes
            or plan.detection_classes != cached_plan.detection_classes
            or plan.protect_semantic_buildings != cached_plan.protect_semantic_buildings):
        raise ValueError("Cached inference does not satisfy the current tool-routing rules. "
                         "Run the question again to obtain the required model evidence.")
    tile_data = trace.get("tile")
    tile = None
    if tile_data:
        x0, y0, x1, y1 = tile_data["box_px"]
        tile = Tile(tile_data["index"], tile_data["count"], tile_data["row"], tile_data["col"],
                    *tile_data["grid"], x0, y0, x1, y1, *tile_data["scene_size_px"])
    result = AgentResult(trace["question"], plan, run_dir=str(destination), tile=tile,
        source_path=trace.get("source_path", ""), resume_source=str(source))
    result.steps.append(Step("Cached inference", "Reuse source image, SegEarth classes and native proposals from "
                             + str(source) + "; no model inference rerun"))
    if plan.use_segmentation:
        saved = trace.get("segmentation") or {}
        mask_path = source / "seg_mask.png"
        if not mask_path.is_file():
            mask_path = Path(saved.get("mask_path", ""))
        if not mask_path.is_file():
            raise FileNotFoundError("Cached SegEarth mask is missing")
        classes = saved.get("classes") or plan.segmentation_classes
        names = saved.get("class_names") or [entry.split(",")[0].strip() for entry in classes]
        colors = saved.get("colors") or [[0, 0, 0]] + [list(int(v*255) for v in colorsys.hsv_to_rgb(
            i/max(1, len(names)-1), .7, .9)) for i in range(len(names)-1)]
        mask = np.asarray(Image.open(mask_path)).copy()
        if mask.shape != (image.height, image.width):
            raise ValueError("Cached semantic mask dimensions differ from the original image")
        saved_path = destination / "seg_mask.png"
        shutil.copy2(mask_path, saved_path)
        building_idx = next((i for i, c in enumerate(classes) if i and segmentation.is_building_class(c)), None)
        buildings, regions, labels, ids = segmentation.split_into_regions(mask, names, building_idx)
        result.segmentation = SegmentationOutput(classes, names, colors, trace.get("coverage") or {},
            mask, str(saved_path), buildings, regions,
            names[building_idx] if building_idx is not None else None, labels, ids)
    if plan.use_detection:
        if trace.get("detections_nms") is None:
            raise ValueError("Cached post-NMS detections are missing")
        result.detection = DetectionOutput(plan.detection_classes, [], trace["detections_nms"], 0)
    if plan.use_unibuilding:
        saved = trace.get("unibuilding")
        if not saved:
            raise ValueError("Cached UniBuilding evidence is missing")
        mask_path = source / "unibuilding_mask.png"
        if not mask_path.is_file():
            mask_path = Path(saved["mask_path"])
        copied = destination / "unibuilding_mask.png"
        shutil.copy2(mask_path, copied)
        result.unibuilding = unibuilding.from_mask(np.asarray(Image.open(copied)) > 0,
            str(copied), threshold=saved["threshold"], min_area=saved["min_area_px"],
            connectivity=saved["connectivity"])
        if result.unibuilding.buildings != saved["buildings"]:
            raise ValueError("Cached building proposal IDs cannot be reconstructed consistently")
    if plan.use_delineate:
        saved = trace.get("fields")
        if not saved:
            raise ValueError("Cached field proposals are missing")
        lclu = destination / "segearth_lclu.tif"
        shutil.copy2(saved["lclu_path"], lclu)
        result.fields = FieldOutput(saved["raw_fields"], saved["eligible"], saved["rejected"],
            str(lclu), "", metadata=saved["metadata"], filter_config=saved["filter_config"])
        # Verify the cached class-ID contract without relabelling/resampling.
        import field_geometry
        if field_geometry.native_filter_config(result.segmentation)["class_id_mapping"] != saved["filter_config"]["class_id_mapping"]:
            raise ValueError("Cached field and SegEarth class-ID mappings differ")
    if plan.protect_semantic_buildings:
        guard=trace.get("semantic_building_guard")
        if not guard:
            raise ValueError("Cached semantic evidence lacks building protection provenance; run inference again")
        result.semantic_building_guard=dict(guard,
            building_mask_path=result.unibuilding.mask_path,
            protected_semantic_mask_path=result.segmentation.mask_path)
        idx=next((i for i,c in enumerate(result.segmentation.classes)
                  if segmentation.is_building_class(c)),0)
        if np.any(result.segmentation.mask[result.unibuilding.mask] != idx):
            raise ValueError("Cached semantic raster does not honor its UniBuilding protection mask")
        if idx and not np.array_equal(result.segmentation.mask == idx,result.unibuilding.mask):
            raise ValueError("Cached building class contains pixels outside its UniBuilding source")
        (destination/"semantic_building_guard.json").write_text(
            json.dumps(result.semantic_building_guard,indent=2),encoding="utf-8")
    if (source / "vlm_batches").is_dir():
        shutil.copytree(source / "vlm_batches", destination / "vlm_batches", dirs_exist_ok=True)
    if (source / "vlm_policy.json").is_file():
        shutil.copy2(source / "vlm_policy.json", destination / "vlm_policy.json")
    return _decide_and_render(result, image, image_path, notify)


def _classification_object_pixels(result, image_path, notify):
    """After instance selection, fetch only the missing measured object pixels.

    LAE remains the instance detector. This rendering pass never uses full LULC
    defaults and never turns a box into foreground when pixel extraction fails.
    """
    needed = []
    for obj in result.vlm_objects:
        if obj.get("origin") != "detector":
            continue
        cls = obj.get("class", "")
        entry = masks.MASK_SEG_CLASS.get(cls, cls)
        if entry and entry not in needed and (result.segmentation is None or
                masks._class_index(result.segmentation, entry) is None):
            needed.append(entry)
    if not needed or not config.MASK_FROM_SEGMENTATION or not image_path:
        return None
    return _timed(result, "SegEarth-OV3 (classification pixels)",
        lambda: segmentation.segment(image_path, ["background"] + needed,
            result.run_dir, out_name="classification_objects_seg.png"),
        lambda s: "measured pixels for selected object classes %s; no box filling" % needed, notify)


def _render_classification_map(result, image, target_classes, image_path=None, notify=None):
    """One source-aligned categorical raster. Only measured pixels can be painted.

    Semantic pixels are laid down first; original field geometry, object pixels
    and UniBuilding pixels take priority in that order. VLM supplies labels and
    selection only. Its freehand boxes cannot supply classification geometry.
    """
    import field_geometry
    import rasterio
    from affine import Affine

    plan, seg, uni = result.plan, result.segmentation, result.unibuilding
    request = plan.request
    targets = [t for t in request.targets if t.role == "target"] if request else []
    semantic_target = (request is None or request.land_cover_scope == "general_scene"
                       or any(t.kind == "land_cover" for t in targets))
    building_target = request is None or any(t.kind == "building" for t in targets)
    object_seg = _classification_object_pixels(result, image_path, notify)
    sources = [s for s in (object_seg, seg) if s is not None]
    out = np.zeros((image.height, image.width), np.uint8)
    evidence, missing = [], []
    labels, aliases = [], {}

    def register(entry):
        name = entry.split(",")[0].strip().lower()
        if name in planner.BACKGROUND_CLASSES:
            return "background/unclassified"
        if name and name not in labels:
            labels.append(name)
        for alias in entry.lower().split(","):
            aliases[alias.strip()] = name
        aliases[entry.strip().lower()] = name
        aliases[",".join(part.strip().lower() for part in entry.split(","))] = name
        return name

    if request and request.output_classes:
        for entry in request.output_classes:
            register(entry)
    elif targets:
        for target in targets:
            for entry in (target.classes or (["building"] if target.kind == "building" else [])):
                register(entry)
    elif seg is not None:
        for entry in seg.classes[1:]:
            register(entry)
    strict_labels = bool(request and request.output_classes)
    for target in targets:
        for entry in target.classes:
            parts = [part.strip().lower() for part in entry.split(",")]
            matches = [name for name in labels if name in parts]
            if len(matches) == 1:
                for alias in entry.lower().split(","):
                    aliases[alias.strip()] = matches[0]
    building_names = ("building", "buildings", "house", "houses", "home", "homes", "roof", "roofs", "rooftop")
    building_label = next((name for name in building_names if name in labels), None)
    if building_label:
        for alias in building_names:
            aliases[alias] = building_label

    def paint(pixels, label, source, window=None):
        if plan.protect_semantic_buildings and uni is not None and source != "UniBuilding component pixels":
            pixels = pixels & ~(uni.mask if window is None else uni.mask[window])
        if not np.any(pixels):
            return
        label = str(label or "unknown").strip().lower()
        if label in planner.BACKGROUND_CLASSES:
            return
        label = aliases.get(label, label)
        if strict_labels and label not in labels:
            label = "unknown"
        if label not in labels:
            register(label)
        if len(labels) > 255:
            raise ValueError("Categorical PNG supports at most 255 foreground classes")
        view = out if window is None else out[window]
        view[pixels] = labels.index(label) + 1
        if source not in evidence:
            evidence.append(source)

    if semantic_target and seg is not None:
        names = seg.class_names[1:] if plan.output == "tool_output" else target_classes
        for name in names:
            idx = masks._class_index(seg, name)
            if idx is not None:
                paint(seg.mask == idx, seg.class_names[idx], "semantic class pixels")
        for obj in result.vlm_objects:
            if obj.get("origin") == "segmentation" and seg.component_labels is not None:
                component = seg.component_id_to_label.get(obj.get("source_id"))
                if component is not None:
                    paint(seg.component_labels == component, obj.get("label") or obj.get("class"),
                          "semantic class pixels")
    for obj in result.vlm_objects:
        if obj.get("origin") == "delineate":
            window, pixels = field_geometry.polygon_pixels(obj["geometry"], image.size)
            paint(pixels, obj.get("field_classification") or obj.get("label") or "agricultural field",
                  "original field polygons", window)
    for obj in result.vlm_objects:
        if obj.get("origin") != "detector":
            continue
        x1, y1, x2, y2 = masks._px(obj["box"], image.width, image.height)
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(image.width,x2), min(image.height,y2)
        window = (slice(y1,y2), slice(x1,x2))
        cls = obj.get("class", "")
        pixels = None
        for source in sources:
            idx = masks._class_index(source, cls)
            if idx is None:
                idx = masks._class_index(source, masks.MASK_SEG_CLASS.get(cls, cls))
            if idx is not None:
                candidate = source.mask[window] == idx
                if candidate.any():
                    pixels = candidate
                    break
        if pixels is None:
            missing.append({"id": obj.get("source_id"), "reason": "No measured object pixels; left unclassified"})
        else:
            paint(pixels, obj.get("label") or cls, "object segmentation pixels inside detections", window)
    if building_target and uni is not None:
        if plan.output == "tool_output":
            paint(uni.mask, "building", "UniBuilding component pixels")
        else:
            for obj in result.vlm_objects:
                if obj.get("origin") != "unibuilding":
                    continue
                component = uni.component_id_to_label.get(obj.get("source_id"))
                if component is not None:
                    x1,y1,x2,y2 = masks._px(obj["box"], image.width, image.height)
                    window = (slice(max(0,y1),min(image.height,y2)), slice(max(0,x1),min(image.width,x2)))
                    paint(uni.component_labels[window] == component, obj.get("label") or "building",
                          "UniBuilding component pixels", window)
    missing += [{"id": None, "reason": "VLM box has no measured pixels; left unclassified"}
                for obj in result.vlm_objects if obj.get("origin") == "vlm"]
    if len(labels) > 255:
        raise ValueError("Categorical PNG supports at most 255 foreground classes")
    colors = {"background/unclassified": [0,0,0], **drawing.class_colors(labels)}
    table = [{"id": 0, "name": "background/unclassified", "color": [0,0,0]}] + [
        {"id": i+1, "name": name, "color": list(colors[name])} for i,name in enumerate(labels)]
    for row in table:
        row.update(pixels=int(np.count_nonzero(out == row["id"])),
                   coverage_percent=float(np.mean(out == row["id"])*100))
    mapping = {row["id"]: row["name"] for row in table}
    # Indexed PNG preserves the IDs as well as the class colours, without a
    # legend strip or source-image blending changing the raster's dimensions.
    palette = [v for row in table for v in row["color"]]
    canvas = Image.fromarray(out).convert("P")
    canvas.putpalette(palette + [0]*(768-len(palette)))
    result.output_image_path = os.path.join(result.run_dir, "classification_map.png")
    canvas.save(result.output_image_path)
    legend = drawing.add_legend(canvas, [row["name"] for row in table],
        [row["color"] for row in table], {row["name"]:row["coverage_percent"] for row in table})
    legend_path = os.path.join(result.run_dir,"classification_map_legend.png")
    comparison_path = os.path.join(result.run_dir,"classification_comparison.png")
    legend.save(legend_path)
    drawing.classification_comparison(image,legend).save(comparison_path)
    tile = result.tile
    meta = field_geometry.source_metadata((tile.scene_w,tile.scene_h) if tile else image.size,
                                           result.source_path)
    transform = Affine(*meta["transform"])
    if tile:
        transform *= Affine.translation(tile.x0, tile.y0)
    tif_path = os.path.join(result.run_dir, "classification_ids.tif")
    with rasterio.open(tif_path, "w", driver="GTiff", width=image.width, height=image.height,
            count=1, dtype="uint8", crs=meta["crs_wkt"], transform=transform, compress="deflate") as dst:
        dst.write(out,1)
        dst.write_colormap(1, {row["id"]: tuple(row["color"]) for row in table})
        dst.update_tags(class_id_mapping=json.dumps(mapping),
                        coordinate_reference=meta["coordinate_reference"],
                        background_value="0 = background or unclassified")
    result.output_caption = "Classification mask · " + (", ".join(evidence) or "no measured subjects selected")
    if missing:
        result.output_caption += "; %d objects without pixels left unclassified" % len(missing)
    result.answer += "\nClassification mask pixel coverage: " + "; ".join(
        "%s %.3f%%" % (row["name"], row["coverage_percent"]) for row in table) + "."
    with open(os.path.join(result.run_dir, "classification_map.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": labels, "class_id_mapping": mapping, "class_table": table,
            "evidence": evidence, "unrasterized": missing,
            "semantic_building_guard": result.semantic_building_guard,
            "overlap_priority": ["semantic", "field polygons", "object pixels", "UniBuilding"],
            "raster_path": tif_path, "width": image.width, "height": image.height,
            "legend_image_path": legend_path, "comparison_image_path": comparison_path,
            "request": request.model_dump() if request else None}, fh, indent=2)
    return result.output_image_path


def _target_origins(plan):
    if plan.request is None:
        return ["delineate"]
    kinds = {t.kind for t in plan.request.targets if t.role == "target"}
    return [origin for kind, origin in (("agricultural_field", "delineate"),
            ("object", "detector"), ("building", "unibuilding"), ("land_cover", "segmentation"))
            if kind in kinds]


def _enforce_decision_scope(reply, plan):
    """Context can inform a decision without becoming an extra output subject."""
    if plan.request is None or not plan.request.targets:
        return
    targets = [t for t in plan.request.targets if t.role == "target"]
    allowed = set(_target_origins(plan))
    def names(kind):
        return {value.strip().lower() for t in targets if t.kind == kind
                for entry in t.classes for value in [entry, *entry.split(",")]}
    semantic = names("land_cover")
    if plan.request.land_cover_scope == "general_scene":
        semantic |= {value.strip().lower() for entry in plan.segmentation_classes
                     for value in [entry, *entry.split(",")]}
    objects = names("object")
    kept = []
    for obj in reply.objects:
        origin = obj.get("origin")
        if origin == "vlm" and not plan.use_delineate and plan.request.operation != "classify":
            kept.append(obj)
            continue
        if origin not in allowed:
            continue
        cls = str(obj.get("class") or obj.get("label") or "").strip().lower()
        if origin == "segmentation" and cls not in semantic:
            continue
        if origin == "detector" and objects and cls not in objects:
            continue
        kept.append(obj)
    reply.objects = kept
    reply.target_classes = [name for name in reply.target_classes if name.lower() in semantic]


def _build_mask(result: AgentResult, plan: Plan, image, image_path, target_classes, notify=None):
    """The mask rendering of the decision. Runs after the VLM and never touches
    the plan; its only possible side trip is one segmentation pass for the
    pixels of the detections the VLM kept."""
    if plan.use_delineate and result.fields is None:
        result.final_mask_note = "not built: field evidence unavailable"
        result.steps.append(Step("Mask", result.final_mask_note, status="skipped"))
        return
    if plan.output == "tool_output" and (result.segmentation is not None or result.unibuilding is not None):
        result.final_mask = np.zeros((image.height,image.width),bool)
        parts = []
        if result.segmentation is not None:
            pixels, note = masks.from_plain_map(result.segmentation)
            result.final_mask |= pixels
            parts.append(note)
        if result.unibuilding is not None and "unibuilding" in _target_origins(plan):
            result.final_mask |= result.unibuilding.mask
            parts.append("building pixels from UniBuilding")
        result.final_mask_note = "; ".join(parts)
    elif result.vlm_objects or target_classes:
        object_seg = None
        needed = masks.object_seg_classes(result.vlm_objects, result.segmentation)
        if needed and config.MASK_FROM_SEGMENTATION:
            object_seg = _timed(
                result, "SegEarth-OV3 (mask pixels)",
                lambda: segmentation.segment(image_path, ["background"] + needed,
                                             result.run_dir, out_name="mask_objects_seg.png"),
                lambda s: "pixels for the kept objects: %s" % [c.split(",")[0] for c in needed],
                notify)
        result.final_mask, result.final_mask_note = masks.from_decision(
            result.vlm_objects, target_classes, result.segmentation, image.size,
            object_seg=object_seg, unibuilding=result.unibuilding)
    else:
        vlm_failed = any(s.name.startswith("VLM") and s.status == "error" for s in result.steps)
        result.final_mask_note = ("not built: the VLM call failed" if vlm_failed
                                  else "not built: nothing was selected to put in the mask")
        result.steps.append(Step("Mask", result.final_mask_note, status="skipped"))
        return
    if plan.protect_semantic_buildings and result.unibuilding is not None and "unibuilding" not in _target_origins(plan):
        result.final_mask &= ~result.unibuilding.mask
        result.final_mask_note += "; UniBuilding pixels excluded from non-building targets"
    Image.fromarray((result.final_mask * 255).astype("uint8")).save(
        os.path.join(result.run_dir, "final_mask.png"))
    result.steps.append(Step("Mask", result.final_mask_note))


def _save_trace(result: AgentResult):
    trace = {
        "question": result.question,
        "tile": result.tile.as_dict() if result.tile is not None else None,
        "plan": result.plan.model_dump(),
        "steps": [s.__dict__ for s in result.steps],
        "answer": result.answer,
        "answered_by": result.answered_by,
        "detections_nms": result.detection.nms if result.detection else None,
        "coverage": result.segmentation.coverage if result.segmentation else None,
        "buildings": len(result.unibuilding.buildings) if result.unibuilding else None,
        "source_path": result.source_path,
        "resume_source": result.resume_source or None,
        "segmentation": {"classes": result.segmentation.classes,
                         "class_names": result.segmentation.class_names,
                         "colors": result.segmentation.colors,
                         "mask_path": result.segmentation.mask_path} if result.segmentation else None,
        "semantic_building_guard": result.semantic_building_guard,
        "vlm_field_review": os.path.join(result.run_dir, "vlm_field_review.json")
                            if os.path.isfile(os.path.join(result.run_dir, "vlm_field_review.json")) else None,
        "fields": {"raw_fields": result.fields.raw_fields, "eligible": result.fields.fields,
                   "rejected": result.fields.rejected, "filter_config": result.fields.filter_config,
                   "metadata": result.fields.metadata, "lclu_path": result.fields.lclu_path,
                   "gpkg_path": result.fields.gpkg_path} if result.fields else None,
        "unibuilding": {
            "buildings": result.unibuilding.buildings,
            "coverage_percent": result.unibuilding.coverage,
            "mask_path": result.unibuilding.mask_path,
            "probability_path": result.unibuilding.probability_path,
            "threshold": result.unibuilding.threshold,
            "min_area_px": result.unibuilding.min_area_px,
            "connectivity": result.unibuilding.connectivity,
        } if result.unibuilding else None,
        "regions": len(result.segmentation.regions) if result.segmentation else None,
        "vlm_objects": result.vlm_objects,
        "vlm_rejected": result.vlm_rejected,
        "final_mask": result.final_mask_note or None,
        "presentation": {"image_path": result.output_image_path, "caption": result.output_caption},
    }
    with open(os.path.join(result.run_dir, "trace.json"), "w", encoding="utf-8") as fh:
        json.dump(trace, fh, ensure_ascii=False, indent=2)


def _save_manifest(run: TiledRun):
    """runs/<ts>/tiles.json: how the scene was cut and where each tile's own
    trace lives. Geometry only -- the tile answers are deliberately not merged."""
    manifest = {
        "question": run.question,
        "scene_size_px": list(run.scene_size),
        "tile_size": run.tile_size,
        "strategy": run.strategy,
        "tiled": run.tiled,
        "seconds": run.seconds,
        "tiles": [dict(t.as_dict(), run_dir=os.path.relpath(r.run_dir, run.run_dir))
                  for t, r in zip(run.tiles, run.results)],
    }
    with open(os.path.join(run.run_dir, "tiles.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
