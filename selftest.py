"""
Offline self-test -- no models, no API key, no GPU. Run in the SAM env:

    python selftest.py

Checks the parts that are easy to break silently: the planner rules, the worker
protocol (with a fake worker), building components, drawing, JSON extraction,
and the orchestrator's routing (tool output vs VLM answer) with stubbed tools.
V02 adds (sections 9-10): tiling -- tiles are never padded, cover the image,
stay <= TILE_SIZE -- and run_tiled() running the whole agent once per tile.
"""

import os
import sys
import tempfile
import textwrap

import numpy as np
from PIL import Image

import config
import drawing
import orchestrator
import planner
import vlm
from llm_client import extract_json
from schemas import DetectionOutput, Plan, SegmentationOutput
from tools import detection, segmentation, unibuilding
from tools.worker_client import ToolWorker, ToolWorkerError

OK = []
SCENE_SEG_CLASSES = [c for c in config.DEFAULT_LULC_CLASSES if not segmentation.is_building_class(c)]


def check(name, cond):
    OK.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name)


# 1. planner rules -----------------------------------------------------------
p = planner.apply_rules(Plan(use_detection=True, detection_classes=["House", "small vehicle"]),
                        "how many houses and cars?")
check("R1 building moved to UniBuilding",
      p.detection_classes == ["small vehicle"] and p.use_unibuilding
      and not p.use_segmentation and p.tools == ["detection", "unibuilding"])

p = planner.apply_rules(Plan(), "Create a land use land cover map of this image")
check("R2 map request -> default LULC + tool_output",
      p.use_segmentation and p.output == "tool_output"
      and p.segmentation_classes == SCENE_SEG_CLASSES and p.use_unibuilding)

p = planner.apply_rules(Plan(), "یک نقشه کاربری اراضی بساز")
check("R2 Persian map request", p.use_segmentation and p.output == "tool_output")

p = planner.apply_rules(Plan(use_detection=True, detection_classes=[], output="tool_output"),
                        "describe the scene")
check("R2/R3 no classes, no map request -> vlm_answer",
      not p.use_detection and not p.use_segmentation and p.output == "vlm_answer")

# R2 -- the VLM is skipped ONLY for a pure map / segmentation-output request,
# even when the router LLM wrongly says tool_output (your "show all sport fields" run)
SPORT = ["tennis court", "soccer ball field"]
p = planner.apply_rules(Plan(use_detection=True, detection_classes=SPORT, output="tool_output"),
                        "show all sport fields")
check("R2 'show all sport fields' -> VLM decides (router said tool_output)",
      p.output == "vlm_answer" and set(SPORT) <= set(p.detection_classes))
for q in ["how many buildings are there?", "show me the buildings",
          "what percentage of land cover is forest?", "how many buildings in the segmentation map?",
          "Describe the land cover", "چند ساختمان وجود دارد؟"]:
    p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["building"],
                                 output="tool_output"), q)
    check("R2 question -> VLM: %s" % q, p.output == "vlm_answer")
for q in ["segment the roads", "give me the water map", "Segment water and vegetation",
          "قطعه‌بندی جاده‌ها را انجام بده"]:
    p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["road"],
                                 output="tool_output"), q)
    check("R2 pure segmentation request -> map only: %s" % q, p.output == "tool_output")
p = planner.apply_rules(Plan(use_detection=True, detection_classes=["ship"], output="vlm_answer"),
                        "Can you create a land cover map?")
check("R2 LULC map forced to map-only, detection dropped",
      p.output == "tool_output" and not p.use_detection and p.use_segmentation)

p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["Water, River", "roof"]),
                        "how much water?")
check("R1/R4 background first, roof routed to UniBuilding, water preserved",
      p.segmentation_classes == ["background", "water,river"] and p.use_unibuilding)

# R2 / R3 -- right tool per class, and BOTH tools when the question needs both
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["tennis court", "stadium"],
                             output="tool_output"), "show all sport fields maps")
check("R2 countable objects moved from segmentation to detection",
      {"tennis court", "stadium"} <= set(p.detection_classes) and not p.use_segmentation
      and p.output == "vlm_answer")
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["building"]),
                        "which houses have a car parked next to them?")
check("R3 buildings + cars -> both tools",
      p.use_unibuilding and not p.use_segmentation and p.detection_classes == ["small vehicle"])
p = planner.apply_rules(Plan(use_detection=True, detection_classes=["large vehicle"]),
                        "which trucks near the buildings cannot leave the parking lot?")
check("R3 trucks + buildings -> both tools",
      p.detection_classes == ["large vehicle"] and p.use_unibuilding and not p.use_segmentation)
p = planner.apply_rules(Plan(), "چند ماشین کنار ساختمان‌ها پارک شده؟")
check("R3 Persian cars + buildings -> both tools",
      p.detection_classes == ["small vehicle"] and p.use_unibuilding and not p.use_segmentation)
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["building"], output="tool_output"),
                        "detect all building and show a binary map that just contain the largest building")
check("R4 map WITH reasoning (largest) -> VLM decides", p.output == "vlm_answer")

# R4 / R5 -- output_kind (boxes / mask) is the ROUTER's choice and only an
# output format: rules pass it through and it never changes tools or decider
p = planner.apply_rules(Plan(use_detection=True, detection_classes=["large vehicle"],
                             output="tool_output", output_kind="mask"),
                        "create a mask from yellow vehicles")      # the real run that failed
check("'mask from yellow vehicles' -> detection + VLM, shown as mask, no extra segmentation",
      p.output == "vlm_answer" and p.output_kind == "mask"
      and set(p.detection_classes) == {"large vehicle", "small vehicle"}
      and not p.use_segmentation)
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["building"],
                             output="tool_output", output_kind="mask"), "create a mask of the biggest home")
check("'mask of the biggest home' -> VLM, shown as mask", p.output == "vlm_answer" and p.output_kind == "mask")
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["bare soil"],
                             output="tool_output", output_kind="mask"), "create a mask of the empty land")
check("'mask of the empty land' -> VLM, shown as mask", p.output == "vlm_answer" and p.output_kind == "mask")
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["building"],
                             output="tool_output"), "segment the buildings")
check("'segment the buildings' -> UniBuilding + VLM, boxes default",
      p.output == "vlm_answer" and p.output_kind == "boxes" and p.use_unibuilding
      and not p.use_detection and not p.use_segmentation)
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["bare soil", "grass"],
                             output_kind="boxes"), "show places that are a good candidate to construct buildings")
check("rules never turn a mask on: router said boxes -> boxes", p.output_kind == "boxes")
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["building"],
                             output="tool_output"), "create a landuse landcover map")
check("LULC map -> full default scheme even if the router picked only 'building'",
      p.segmentation_classes == SCENE_SEG_CLASSES and p.output_kind == "boxes" and p.use_unibuilding)

# The same question with and without mask wording plans IDENTICALLY (tools,
# classes, decider) -- only output_kind differs. Same router answer both times.
INVARIANT = [
    ("show the yellow vehicles", Plan(use_detection=True, detection_classes=["small vehicle"])),
    ("give me the water", Plan(use_segmentation=True, segmentation_classes=["water"],
                               output="tool_output")),
    ("segment the roads", Plan(use_segmentation=True, segmentation_classes=["road"],
                               output="tool_output")),
    ("create a land cover map", Plan(use_segmentation=True, output="tool_output")),
    ("the biggest home", Plan(use_unibuilding=True)),
    ("ships near the harbor", Plan(use_detection=True, detection_classes=["ship", "harbor"])),
    ("خودروهای زرد", Plan(use_detection=True, detection_classes=["small vehicle"])),
]
for q, router in INVARIANT:
    for suffix in (" as a mask", " as a binary mask", " ماسک"):
        a = planner.apply_rules(router.model_copy(update={"output_kind": "boxes"}), q)
        b = planner.apply_rules(router.model_copy(update={"output_kind": "mask"}), q + suffix)
        same = (a.tools, a.detection_classes, a.segmentation_classes, a.output) == \
               (b.tools, b.detection_classes, b.segmentation_classes, b.output)
        check("mask wording changes no plan: %r + %r" % (q, suffix),
              same and a.output_kind == "boxes" and b.output_kind == "mask")

# output_kind is lenient: a near-miss from the router must not kill the plan
check("output_kind normalised: 'Binary Mask' -> mask, 'bounding boxes' / None -> boxes",
      Plan(output_kind="Binary Mask").output_kind == "mask"
      and Plan(output_kind="bounding boxes").output_kind == "boxes"
      and Plan(output_kind=None).output_kind == "boxes" and Plan().output_kind == "boxes")
check("router JSON with output_kind parses",
      Plan.model_validate({"use_detection": True, "detection_classes": ["ship"],
                           "output_kind": "mask"}).output_kind == "mask")
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["farmland", "water"],
                             output="tool_output"), "create a land use land cover map")
check("LULC map -> full scheme; farmland/water already represented as synonyms",
      p.segmentation_classes == SCENE_SEG_CLASSES and p.use_unibuilding)
p = planner.apply_rules(Plan(use_detection=True, detection_classes=["tennis court"],
                             output="tool_output"), "segment the sport fields")
check("'segment the sport fields' -> objects -> detection + VLM",
      p.output == "vlm_answer" and "tennis court" in p.detection_classes)
p = planner.apply_rules(Plan(), "how many cars are there?")
check("default output_kind -> boxes", p.output_kind == "boxes")

# "car parking" is a place, not a car
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["parking lot"]),
                        "detect all car parkings")                   # the real run that failed
check("'detect all car parkings' -> parking-lot segmentation only, no car detection",
      not p.use_detection and p.segmentation_classes == ["background", "parking lot"])
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["parking lot"]),
                        "how many cars are in the car park?")
check("'cars in the car park' -> cars still detected", p.detection_classes == ["small vehicle"])

# Explicit buildings repair either older route, including synonyms and Persian.
for term in ["building", "buildings", "house", "houses", "home", "homes", "roof",
             "roofs", "rooftop", "rooftops", "ساختمان", "خانه", "منزل", "بام", "سقف"]:
    for router in (Plan(), Plan(use_detection=True, detection_classes=[term]),
                   Plan(use_segmentation=True, segmentation_classes=[term])):
        p = planner.apply_rules(router, "show the " + term)
        check("building term repaired: " + term,
              p.tools == ["unibuilding"] and p.output == "vlm_answer")
for q in ["show me the buildings", "how many houses are there?", "highlight the largest building"]:
    p = planner.apply_rules(Plan(output="tool_output"), q)
    check("building request: " + q, p.tools == ["unibuilding"] and p.output == "vlm_answer")
for q, cls in [("cars next to buildings", "small vehicle"),
               ("trucks near buildings", "large vehicle")]:
    p = planner.apply_rules(Plan(), q)
    check("mixed objects and buildings: " + q,
          p.tools == ["detection", "unibuilding"] and p.detection_classes == [cls])
p = planner.apply_rules(Plan(use_segmentation=True, segmentation_classes=["building", "road"]),
                        "locate buildings near the road")
check("buildings + road -> UniBuilding and SegEarth road only",
      p.tools == ["segmentation", "unibuilding"]
      and p.segmentation_classes == ["background", "road"] and p.output == "vlm_answer")
p = planner.apply_rules(Plan(use_unibuilding=True), "create a land cover map")
check("generic LULC preserves scene classes with buildings supplied by UniBuilding",
      p.tools == ["segmentation","unibuilding"] and p.segmentation_classes == SCENE_SEG_CLASSES)
check("ships still use LAE only", planner.apply_rules(Plan(), "show the ships").tools == ["detection"])
p = planner.apply_rules(Plan(use_segmentation=True, use_unibuilding=True),
                        "create a land cover map and detect the buildings")
check("explicit building task alongside general LULC keeps non-building land-cover classes",
      p.tools == ["segmentation", "unibuilding"] and p.output == "vlm_answer"
      and p.segmentation_classes == [c for c in config.DEFAULT_LULC_CLASSES
                                     if not segmentation.is_building_class(c)])
check("non-building rooftop-like word is not a building class",
      not segmentation.is_building_class("greenhouse gas"))

# 2. worker protocol with a fake worker ----------------------------------------
tmp = tempfile.mkdtemp()
fake = os.path.join(config.WORKERS_DIR, "_fake_worker_selftest.py")
with open(fake, "w") as fh:
    fh.write(textwrap.dedent("""
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from protocol import serve
        def init(s):
            print("noisy model loading output")      # must not break the protocol
            return {"loaded": s["x"]}
        def run(a):
            if a.get("boom"): raise ValueError("tool exploded")
            if a.get("die"): os._exit(3)
            print("noisy inference")
            return {"echo": a["v"] * 2}
        serve(init, run)
    """))
try:
    w = ToolWorker("fake", sys.executable, fake, {"x": 7}, tmp)
    check("worker call + init", w.call({"v": 21}) == {"echo": 42} and w.load_info == {"loaded": 7})
    try:
        w.call({"boom": True})
        check("worker error propagates", False)
    except ToolWorkerError as exc:
        check("worker error propagates", "tool exploded" in str(exc))
    check("worker survives an error", w.call({"v": 1}) == {"echo": 2})
    try:
        w.call({"die": True})
        check("worker crash detected", False)
    except ToolWorkerError:
        check("worker crash detected", True)
    # must not wait for poll() to notice (on Windows it can lag -> Errno 22)
    check("dead worker discarded immediately", w._proc is None)
    check("worker restarts after crash", w.call({"v": 2}) == {"echo": 4})
    w.close()
finally:
    os.remove(fake)

# 3. building components -------------------------------------------------------
m = np.zeros((100, 200), bool)
m[10:30, 10:40] = True        # 600 px
m[50:60, 100:110] = True      # 100 px
m[90:92, 190:192] = True      # 4 px -> noise
comps = segmentation.connected_components(m, 40)
check("connected components (2 kept, largest first)",
      len(comps) == 2 and comps[0]["area_px"] == 600 and comps[0]["bbox"] == [0.05, 0.1, 0.2, 0.3])

# 4. JSON extraction + drawing -------------------------------------------------
check("extract_json repairs .5 / trailing comma, never returns an inner object",
      extract_json('{"answer": "x", "box": [.1, .2,], "o": {"id": "b1"}}') ==
      {"answer": "x", "box": [0.1, 0.2], "o": {"id": "b1"}})
check("extract_json with fences/preamble",
      extract_json('Sure:\n```json\n{"answer": "a {b}", "objects": []}\n```') ==
      {"answer": "a {b}", "objects": []})
img = Image.new("RGB", (200, 100), (90, 120, 90))
mask = np.zeros((100, 200), np.uint8)
mask[:, 100:] = 1
over = drawing.add_legend(drawing.seg_overlay(img, mask, [[0, 0, 0], [255, 0, 0]]),
                          ["background", "building"], [[0, 0, 0], [255, 0, 0]], {"building": 50.0})
check("overlay + legend render", over.size[0] > 200)

# 5. router receives the question only -- no image (LLM stubbed) ---------------
import llm_client
sent = []
answers = ['{"operation":"count","targets":[{"kind":"object","classes":["ship"]}],"requires_reasoning":true}']
def _fake_complete(body, stream):
    sent.append(__import__("copy").deepcopy(body))
    if len(sent) == 1:  # first attempt: pretend the endpoint rejects response_format
        err = Exception("400"); err.status_code = 400; raise err
    return answers[0]
llm_client._complete = _fake_complete
plan = planner.make_plan("count the vessels")
check("router gets the question as plain text (no image), retries without response_format",
      plan.detection_classes == ["ship"] and sent[-1]["messages"][1]["content"] == "count the vessels"
      and "response_format" in sent[0] and "response_format" not in sent[-1])

# 5b. the VLM gets the ORIGINAL image: full size, lossless ------------------------
import base64, io as _io
big = Image.fromarray(np.random.randint(0, 255, (3855, 5001, 3), np.uint8))
part = llm_client.image_part(big, "original")
sent_img = Image.open(_io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1])))
check("VLM image: original 5001x3855, PNG, pixel-identical, detail=original",
      sent_img.size == (5001, 3855) and sent_img.format == "PNG"
      and part["image_url"]["detail"] == "original"
      and np.array_equal(np.asarray(sent_img), np.asarray(big)))
huge, note = llm_client.fit_patch_limit(Image.new("RGB", (8000, 6000)))
check("only an image over the API's 30,000-patch limit is reduced, just enough",
      llm_client.patches(huge.size) <= 30000 and llm_client.patches(huge.size) > 29000 and bool(note))

# 6. VLM alignment with the tool outputs ------------------------------------------
props = {"d0": {"label": "small vehicle", "box": [0.1, 0.1, 0.2, 0.2], "origin": "detector"},
         "b0": {"label": "building", "box": [0.5, 0.5, 0.9, 0.9], "origin": "unibuilding"}}
out = vlm.align([{"id": "d0"}, {"id": "d0"},                              # kept + duplicate
                 {"label": "house", "box": [0.52, 0.5, 0.9, 0.88]},       # drawn by VLM ~ b0
                 {"label": "ship", "box": [0.0, 0.8, 0.05, 0.9]},         # truly new
                 {"id": "d7"}, {"box": [1]}], props)                      # junk
check("align: id -> tool box, duplicates dropped",
      out[0]["box"] == props["d0"]["box"] and out[0]["origin"] == "detector" and len(out) == 3)
check("align: VLM box snapped onto matching building component",
      out[1]["source_id"] == "b0" and out[1]["box"] == props["b0"]["box"])
check("align: unmatched object stays origin=vlm", out[2]["origin"] == "vlm")
check("answer language follows the question",
      vlm.answer_language("detect red cars") == "English"
      and vlm.answer_language("ماشین های قرمز را پیدا کن") == "Persian (فارسی)")
bm = np.zeros((10, 20), bool); bm[1:3, 1:4] = True; bm[6:9, 10:18] = True
uni_c = unibuilding.from_mask(bm, min_area=1)
sel = uni_c.component_labels == uni_c.component_id_to_label["b0"]
check("selected-building map contains only the chosen component",
      sel.sum() == 24 and not sel[1:3, 1:4].any())

# 6a. tool outputs weigh more than the VLM's doubts ------------------------------
props2 = {"d0": {"label": "ship", "box": [0.1, 0.1, 0.2, 0.2], "origin": "detector"},
          "d1": {"label": "ship", "box": [0.3, 0.1, 0.4, 0.2], "origin": "detector"},
          "d2": {"label": "ship", "box": [0.5, 0.1, 0.6, 0.2], "origin": "detector"},
          "d3": {"label": "harbor", "box": [0.0, 0.0, 0.7, 0.3], "origin": "detector"}}
rej = vlm.valid_rejections([{"id": "d1", "reason": "this is a pier, not a ship"},
                            {"id": "d2", "reason": ""},                 # no reason -> ignored
                            {"id": "d0", "reason": "unclear"}],         # also kept -> tool wins
                           props2, [{"source_id": "d0"}])
check("rejection needs a reason, and a kept proposal cannot be rejected",
      [r["id"] for r in rej] == ["d1"])
kept = vlm.keep_by_default(["ship"], props2, [], rej)
check("'all of a class': every ship kept unless rejected, context class (harbor) not added",
      sorted(o["source_id"] for o in kept) == ["d0", "d2"])

# 6b. mask output from the VLM decision (tool pixels) -----------------------------
import masks
mm = np.zeros((100, 200), np.uint8)
mm[10:20, 10:20] = 1                     # "car" pixels inside the first box only
comps3, labels3, id3 = segmentation.label_components(mm == 2, 1)
seg_m = SegmentationOutput(classes=["background", "car,vehicle", "bare soil"],
                           class_names=["background", "car", "bare soil"], colors=[[0, 0, 0]] * 3,
                           coverage={}, mask=mm, mask_path="")
mm[60:70, 0:50] = 2                      # bare soil
objs = [{"origin": "detector", "class": "small vehicle", "box": [0.05, 0.1, 0.15, 0.3], "source_id": "d0"},
        {"origin": "detector", "class": "small vehicle", "box": [0.5, 0.5, 0.6, 0.6], "source_id": "d1"},
        {"origin": "vlm", "class": "", "box": [0.9, 0.9, 1.0, 1.0], "source_id": None}]
fm, note = masks.from_decision(objs, ["bare soil"], seg_m, (200, 100))
check("mask: car pixels inside box, filled box fallback, VLM box, land-cover class",
      fm[10:20, 10:20].all() and not fm[20:30, 10:30].any()   # pixels, not the whole box
      and fm[50:60, 100:120].all() and fm[90:100, 180:200].all() and fm[60:70, 0:50].all())

# 6c. every segmented class becomes selectable areas (r*), not only buildings ------
rm = np.zeros((100, 200), np.uint8)
rm[5:15, 5:15] = 1                                  # a building
rm[40:80, 10:60] = 2; rm[40:80, 120:190] = 2        # two parking lots
bld, reg, lab, idmap = segmentation.split_into_regions(rm, ["background", "building", "parking lot"], 1)
check("regions: 1 building component + 2 parking-lot areas with their class",
      [b["id"] for b in bld] == ["b0"] and [(r["id"], r["class"]) for r in reg]
      == [("r0", "parking lot"), ("r1", "parking lot")] and reg[0]["area_px"] == 2800)
seg_r = SegmentationOutput(classes=["background", "building,roof,house", "parking lot"],
                           class_names=["background", "building", "parking lot"],
                           colors=[[0, 0, 0]] * 3, coverage={}, mask=rm, mask_path="",
                           buildings=bld, regions=reg, building_class="building",
                           component_labels=lab, component_id_to_label=idmap)
check("region pixels rebuild exactly one parking lot",
      segmentation.component_mask(seg_r, ["r1"]).sum() == 2000)
out_r = vlm.align([{"id": "r0"}, {"id": "r1"}], vlm.proposals(None, seg_r))
check("VLM can select parking lots by id -> their boxes",
      [o["label"] for o in out_r] == ["parking lot", "parking lot"]
      and out_r[0]["box"] == reg[0]["bbox"])
check("tool JSON lists the parking-lot areas",
      vlm.tool_results_json(img, None, seg_r)["segmentation"]["regions"]["parking lot"]["count"] == 2)
check("SegEarth LULC buildings never supply explicit building proposals",
      "b0" not in vlm.proposals(None, seg_r)
      and "buildings" not in vlm.tool_results_json(img, None, seg_r)["segmentation"])
uni_r = unibuilding.from_mask(rm == 1)
props_r = vlm.proposals(None, seg_r, uni_r)
check("UniBuilding supplies b* with distinct origin alongside SegEarth r*",
      props_r["b0"]["origin"] == "unibuilding" and props_r["r0"]["origin"] == "segmentation")
selected = vlm.align([{"id": "b0"}], props_r)
fm, note = masks.from_decision(selected, [], seg_r, (200, 100), unibuilding=uni_r)
check("selected building mask uses UniBuilding pixels and needs no SegEarth object pass",
      np.array_equal(fm, rm == 1) and "UniBuilding" in note
      and masks.object_seg_classes(selected, seg_r) == [])
split = vlm.align([{"id": "b0", "box": [0.025, 0.05, 0.05, 0.15], "note": "split"}], props_r)
fm, _ = masks.from_decision(split, [], None, (200, 100), unibuilding=uni_r)
check("split/refined building mask clips component pixels to selected box", fm.sum() == 50)
diagonal = np.eye(3, dtype=bool)
check("UniBuilding connectivity follows repository default 8; configurable to 4",
      len(unibuilding.from_mask(diagonal, min_area=1).buildings) == 1
      and len(unibuilding.from_mask(diagonal, min_area=1, connectivity=4).buildings) == 3)
check("empty UniBuilding mask gives zero proposals, zero coverage",
      unibuilding.from_mask(np.zeros((10, 10), bool)).buildings == []
      and unibuilding.from_mask(np.zeros((10, 10), bool)).coverage == 0)
# Exercise real VLM message construction/alignment without making an API call.
real_chat = vlm.chat_json
vlm_messages = []
def _uni_chat(model, messages, schema, **kwargs):
    vlm_messages.append(messages)
    return vlm.VLMReply(answer="one building", keep_all_classes=["building"]), ""
vlm.chat_json = _uni_chat
reply = vlm.answer(img, "show buildings near the road", None, seg_r, unibuilding=uni_r)
vlm.chat_json = real_chat
check("VLM receives original + SegEarth + UniBuilding images, JSON, then aligns b*",
      sum(p["type"] == "image_url" for p in vlm_messages[0][1]["content"]) == 3
      and '"unibuilding"' in vlm_messages[0][1]["content"][-1]["text"]
      and reply.objects[0]["origin"] == "unibuilding")

# 7. orchestrator routing with stubbed tools -----------------------------------
config.RUNS_DIR = os.path.join(tmp, "runs")
calls = []
fake_seg = SegmentationOutput(classes=["background", config.BUILDING_SEG_CLASS],
                              class_names=["background", "building"], colors=[[0, 0, 0], [255, 0, 0]],
                              coverage={"background": 50.0, "building": 50.0}, mask=mask,
                              mask_path="", buildings=comps, building_class="building")
fake_det = DetectionOutput(classes=["ship"], raw=[], nms=[
    {"id": "d0", "label": "ship", "bbox": [0.1, 0.1, 0.2, 0.2], "score": 0.5}], seconds=0.1)
segmentation.segment = lambda *a, **k: calls.append("seg") or fake_seg
detection.detect = lambda *a, **k: calls.append("det") or fake_det
fake_uni = unibuilding.from_mask(mask > 0)
unibuilding.detect = lambda *a, **k: calls.append("uni") or fake_uni
vlm.answer = lambda *a, **k: calls.append("vlm") or vlm.VLMReply(answer="stub", objects=[
    {"label": "ship", "box": [0.1, 0.1, 0.2, 0.2], "origin": "detector", "source_id": "d0"}])

planner.make_plan = lambda q: planner.apply_rules(Plan(), q)
r = orchestrator.run_agent(img, "create a land cover map")
check("LULC map: UniBuilding first, segmentation runs, VLM skipped, no binary mask output",
      calls == ["uni","seg"] and "coverage" in r.answer
      and any(st.name == "VLM" and st.status == "skipped" for st in r.steps)
      and r.final_mask is None)
calls.clear()
planner.make_plan = lambda q: planner.apply_rules(Plan(
    use_unibuilding=True, output="vlm_answer",
    output_kind="mask"), q)
vlm.answer = lambda *a, **k: calls.append("vlm") or vlm.VLMReply(answer="stub", objects=
    vlm.align([{"id": "b0"}], vlm.proposals(None, None, fake_uni)))
r = orchestrator.run_agent(img, "segment the buildings as a binary mask")
check("building mask: UniBuilding + VLM, no SegEarth pass, real building pixels",
      calls == ["uni", "vlm"] and r.final_mask is not None
      and np.array_equal(r.final_mask, mask != 0))

calls.clear()
planner.make_plan = lambda q: planner.apply_rules(Plan(
    use_detection=True, detection_classes=["ship"], use_unibuilding=True), q)
vlm.answer = lambda *a, **k: calls.append("vlm") or vlm.VLMReply(answer="stub", objects=[
    {"label": "ship", "box": [0.1, 0.1, 0.2, 0.2], "origin": "detector", "source_id": "d0"}])
r = orchestrator.run_agent(img, "ships and buildings?")
check("vlm_answer: both tools then VLM; output_kind boxes -> no mask built",
      calls == ["det", "uni", "vlm"] and r.answer == "stub" and len(r.vlm_objects) == 1
      and r.final_mask is None)
check("UniBuilding execution is visible in the saved trace",
      any(s.name.startswith("UniBuilding") and s.status == "ok" for s in r.steps)
      and __import__("json").load(open(os.path.join(r.run_dir, "trace.json"), encoding="utf-8"))[
          "unibuilding"]["buildings"] == fake_uni.buildings)
calls.clear()
planner.make_plan = lambda q: planner.apply_rules(Plan(use_detection=True, detection_classes=["small vehicle"],
    use_segmentation=True, segmentation_classes=["road"], use_unibuilding=True), q)
r = orchestrator.run_agent(img, "cars near buildings along the road")
check("all three tools execute before VLM, UniBuilding protects semantic pixels first", calls == ["uni", "det", "seg", "vlm"])
calls.clear()
def _uni_failure(*a, **k):
    calls.append("uni")
    raise RuntimeError("checkpoint unavailable")
unibuilding.detect = _uni_failure
planner.make_plan = lambda q: planner.apply_rules(Plan(use_unibuilding=True), q)
r = orchestrator.run_agent(img, "show buildings")
check("UniBuilding failure follows existing behavior: error trace + VLM fallback",
      calls == ["uni", "vlm"] and r.unibuilding is None
      and any(s.status == "error" and "checkpoint unavailable" in s.detail for s in r.steps))
unibuilding.detect = lambda *a, **k: calls.append("uni") or fake_uni

# output_kind "mask" with a detection: the object pixels come from ONE extra
# segmentation pass AFTER the VLM -- the plan itself has no segmentation
calls.clear()
vlm.answer = lambda *a, **k: calls.append("vlm") or vlm.VLMReply(answer="stub", objects=[
    {"label": "ship", "class": "ship", "box": [0.1, 0.1, 0.2, 0.2], "origin": "detector",
     "source_id": "d0"}])
planner.make_plan = lambda q: planner.apply_rules(Plan(
    use_detection=True, detection_classes=["ship"], output_kind="mask"), q)
r = orchestrator.run_agent(img, "create a mask of the ships")
check("mask of detections: plan has no segmentation; pixels pass runs after the VLM",
      not r.plan.use_segmentation and calls == ["det", "vlm", "seg"]
      and any(st.name == "SegEarth-OV3 (mask pixels)" for st in r.steps)
      and r.final_mask is not None and r.final_mask.any())

calls.clear()
vlm.answer = lambda *a, **k: calls.append("vlm") or vlm.VLMReply(answer="none of them are yellow")
planner.make_plan = lambda q: planner.apply_rules(Plan(
    use_detection=True, detection_classes=["ship"], output_kind="mask"), q)
r = orchestrator.run_agent(img, "mask of the yellow ships")
check("mask requested but nothing selected -> no mask, a 'skipped' Mask step says why",
      r.final_mask is None and calls == ["det", "vlm"]
      and any(st.name == "Mask" and st.status == "skipped" for st in r.steps))

check("VLM tool JSON has UniBuilding buildings + detections + land cover",
      {"detection", "segmentation", "unibuilding"} <= set(vlm.tool_results_json(img, fake_det, fake_seg, fake_uni))
      and vlm.tool_results_json(img, fake_det, fake_seg, fake_uni)["unibuilding"]["count"] == 1)

# 8. the VLM never sees the pre-NMS detections --------------------------------
poisoned = DetectionOutput(
    classes=["ship"],
    raw=[{"id": "raw0", "label": "ship", "bbox": [0.80, 0.80, 0.90, 0.90], "score": 0.95},
         {"id": "raw1", "label": "ship", "bbox": [0.10, 0.10, 0.20, 0.20], "score": 0.50}],
    nms=[{"id": "d0", "label": "ship", "bbox": [0.10, 0.10, 0.20, 0.20], "score": 0.50,
          "internal_note": "must not be sent"}],
    seconds=0.1)
sent = vlm.tool_results_json(img, poisoned, None)["detection"]
check("VLM sees post-NMS detections only, with whitelisted fields",
      [d["id"] for d in sent["detections"]] == ["d0"]
      and all(set(d) <= set(vlm.VLM_DETECTION_FIELDS) for d in sent["detections"])
      and sent["count_per_class"] == {"ship": 1}
      and "raw" not in repr(sent) and "internal_note" not in repr(sent))

sys.path.insert(0, config.WORKERS_DIR)
import lae_worker  # noqa: E402 -- stdlib + protocol only; mmdet is imported lazily

raw_rows = [{"id": "raw%d" % i, "label": "ship", "bbox": box, "score": score}
            for i, (box, score) in enumerate([([0.10, 0.10, 0.20, 0.20], 0.90),
                                              ([0.11, 0.11, 0.21, 0.21], 0.80),   # suppressed
                                              ([0.50, 0.50, 0.60, 0.60], 0.70)])]
kept = lae_worker.number_after_nms(lae_worker.nms(raw_rows, 0.3, True))
check("NMS survivors are renumbered d0..dn-1, raw rows keep their own ids",
      [d["id"] for d in kept] == ["d0", "d1"]
      and [r["id"] for r in raw_rows] == ["raw0", "raw1", "raw2"]
      and [d["score"] for d in kept] == [0.90, 0.70])

# 9. V02 tiling: never padded, full coverage, at most TILE_SIZE ------------------
import json  # noqa: E402
import random  # noqa: E402

import tiling  # noqa: E402


def coverage_counts(tiles, size):
    cov = np.zeros((size[1], size[0]), np.int32)
    for t in tiles:
        cov[t.y0:t.y1, t.x0:t.x1] += 1
    return cov


def inside(tiles, size):
    return all(0 <= t.x0 < t.x1 <= size[0] and 0 <= t.y0 < t.y1 <= size[1] for t in tiles)


ts = tiling.make_tiles((7000, 3000), 2000, "balanced")
check("balanced 7000x3000 -> 4 x 2 tiles of 1750 x 1500, row-major Tile 1..8",
      len(ts) == 8 and {t.size for t in ts} == {(1750, 1500)}
      and [t.name for t in ts] == ["Tile %d" % k for k in range(1, 9)]
      and (ts[0].row, ts[0].col, ts[3].col, ts[4].row) == (1, 1, 4, 2)
      and ts[3].box == (5250, 0, 7000, 1500))
check("balanced: every pixel in exactly one tile (no overlap, no gap)",
      (coverage_counts(ts, (7000, 3000)) == 1).all() and inside(ts, (7000, 3000)))
tf = tiling.make_tiles((7000, 3000), 2000, "fixed")
check("fixed 7000x3000 -> 4 x 2 tiles of exactly 2000 x 2000; edge tiles shifted inside",
      len(tf) == 8 and {t.size for t in tf} == {(2000, 2000)} and inside(tf, (7000, 3000))
      and tf[3].box == (5000, 0, 7000, 2000) and tf[4].box == (0, 1000, 2000, 3000)
      and (coverage_counts(tf, (7000, 3000)) >= 1).all())
check("fixed: overlap is reported", "5000" in tiling.overlap_note(tf)
      and tiling.overlap_note(ts) == "")
check("<= 2000 x 2000 -> one tile = the whole image (both strategies)",
      all(len(tiling.make_tiles(s, 2000, m)) == 1 and tiling.make_tiles(s, 2000, m)[0].box
          == (0, 0) + s for s in [(2000, 2000), (640, 480), (2000, 1)] for m in tiling.STRATEGIES)
      and not tiling.needs_tiling((2000, 2000), 2000) and tiling.needs_tiling((2001, 10), 2000))
check("only the long side is cut: 7000 x 1500 -> 4 x 1",
      [t.size for t in tiling.make_tiles((7000, 1500), 2000, "balanced")] == [(1750, 1500)] * 4)
check("no sliver tiles: 2010 px -> 1005 + 1005 (balanced)",
      tiling.axis_spans(2010, 2000, "balanced") == [(0, 1005), (1005, 2010)])
random.seed(0)
ok_random = True
for _ in range(300):
    size = (random.randint(1, 9000), random.randint(1, 9000))
    tsize = random.choice([500, 1000, 2000])
    for mode in tiling.STRATEGIES:
        tt = tiling.make_tiles(size, tsize, mode)
        xs = tiling.axis_spans(size[0], tsize, mode)
        ys = tiling.axis_spans(size[1], tsize, mode)
        ok_random &= (inside(tt, size) and all(max(t.size) <= tsize for t in tt)
                      and xs[0][0] == 0 and xs[-1][1] == size[0] and ys[0][0] == 0
                      and ys[-1][1] == size[1] and len(xs) == -(-size[0] // tsize)
                      and all(a[1] >= b[0] for a, b in zip(xs, xs[1:]))          # no gaps
                      and (mode != "balanced" or all(a[1] == b[0] for a, b in zip(xs, xs[1:])))
                      and (mode != "fixed" or size[0] < tsize
                           or all(e - s == tsize for s, e in xs)))
check("300 random sizes: inside the image, <= tile size, full cover, minimal tile count",
      ok_random)

scene = Image.fromarray(np.random.randint(0, 255, (3000, 7000, 3), np.uint8))
arr = np.asarray(scene)
crops = [tiling.crop(scene, t) for t in ts + tf]
check("crop = exact pixels of the window, never padded (both strategies)",
      all(c.size == t.size and np.array_equal(np.asarray(c), arr[t.y0:t.y1, t.x0:t.x1])
          for c, t in zip(crops, ts + tf)))
bad = __import__("dataclasses").replace(ts[3], x1=7100)      # 100 px past the right edge
try:
    tiling.crop(scene, bad)
    check("crop refuses a window outside the image (PIL would pad it)", False)
except ValueError:
    check("crop refuses a window outside the image (PIL would pad it)", True)
check("tile box -> scene pixels (for a later merge stage)",
      ts[5].to_scene_px([0.0, 0.0, 1.0, 1.0]) == list(ts[5].box)
      and ts[5].to_scene_px([0.5, 0.5, 0.5, 0.5]) == [1750 + 875, 1500 + 750] * 2)
ctx = tiling.vlm_context(ts[2])
check("VLM tile context names the tile, its scene window and size; empty when unsplit",
      "Tile 3 of 8" in ctx and "7000 x 3000" in ctx and "x 3500-5249" in ctx
      and "1750 x 1500" in ctx and tiling.vlm_context(tiling.make_tiles((640, 480))[0]) == "")
grid = drawing.draw_tile_grid(drawing.fit(scene, 800), ts, highlight=3, max_side=800)
check("tile grid / locator renders at display size", max(grid.size) == 800)

# 10. V02 orchestrator: run_tiled runs the whole agent per tile -------------------
calls.clear()
router_calls, vlm_seen = [], []
planner.make_plan = lambda q: router_calls.append(q) or planner.apply_rules(Plan(
    use_detection=True, detection_classes=["ship"]), q)


def _fake_vlm(image, question, det, seg, context="", unibuilding=None):
    vlm_seen.append((image.size, context))
    return vlm.VLMReply(answer="tile answer %d" % len(vlm_seen), objects=[
        {"label": "ship", "class": "ship", "box": [0.1, 0.1, 0.2, 0.2], "origin": "detector",
         "source_id": "d0"}])


vlm.answer = _fake_vlm
det_inputs = []
detection.detect = lambda path, *a, **k: det_inputs.append(Image.open(path).size) or fake_det
steps_seen = []
big_scene = Image.new("RGB", (4500, 1000), (40, 80, 120))
run = orchestrator.run_tiled(big_scene, "how many ships?", tile_size=2000, strategy="balanced",
                             notify=lambda i, n, s: steps_seen.append((i, n, s)))
check("run_tiled 4500x1000 -> 3 tiles of 1500x1000, one independent result each",
      run.tiled and len(run.tiles) == 3 and len(run.results) == 3
      and [r.tile.index for r in run.results] == [1, 2, 3]
      and [r.answer for r in run.results] == ["tile answer 1", "tile answer 2", "tile answer 3"])
check("router called ONCE (shared plan), its step copied into every tile's trace",
      router_calls == ["how many ships?"]
      and all(r.steps[0].name.startswith("Router") and "shared by all 3 tiles" in r.steps[0].detail
              for r in run.results))
check("each tile: detector + VLM got the unpadded 1500x1000 tile with its own context",
      det_inputs == [(1500, 1000)] * 3 and [s for s, _ in vlm_seen] == [(1500, 1000)] * 3
      and ["Tile %d of 3" % k in c for k, (_, c) in enumerate(vlm_seen, 1)] == [True] * 3)
check("progress callback reports tile k of 3 for each step",
      {(i, n) for i, n, _ in steps_seen} == {(1, 3), (2, 3), (3, 3)}
      and any(s.startswith("VLM") for _, _, s in steps_seen))
tile_dirs = [r.run_dir for r in run.results]
manifest = json.load(open(os.path.join(run.run_dir, "tiles.json"), encoding="utf-8"))
traces = [json.load(open(os.path.join(d, "trace.json"), encoding="utf-8")) for d in tile_dirs]
check("runs/<ts>/tile_NN/ per tile (input.png = the tile) + tiles.json with geometry only",
      [os.path.basename(d) for d in tile_dirs] == ["tile_01", "tile_02", "tile_03"]
      and all(Image.open(os.path.join(d, "input.png")).size == (1500, 1000) for d in tile_dirs)
      and Image.open(os.path.join(run.run_dir, "input.png")).size == (4500, 1000)
      and [t["box_px"] for t in manifest["tiles"]] == [[0, 0, 1500, 1000], [1500, 0, 3000, 1000],
                                                       [3000, 0, 4500, 1000]]
      and "answer" not in json.dumps(manifest)
      and [t["tile"]["name"] for t in traces] == ["Tile 1", "Tile 2", "Tile 3"])

# a tile whose processing blows up does not take the other tiles with it
real_run_agent = orchestrator.run_agent
def _flaky(image, question, *a, **k):
    if k.get("tile") is not None and k["tile"].index == 2:
        raise RuntimeError("disk full")
    return real_run_agent(image, question, *a, **k)
orchestrator.run_agent = _flaky
run = orchestrator.run_tiled(big_scene, "how many ships?", tile_size=2000)
orchestrator.run_agent = real_run_agent
check("one failing tile -> error result for that tile only, the others answered",
      [r.answered_by for r in run.results][1] == "error"
      and "disk full" in run.results[1].answer
      and run.results[0].answer.startswith("tile answer") and run.results[2].answer.startswith("tile answer"))

# a small image is NOT split: exactly the V01 path, no tile context for the VLM
router_calls.clear(); vlm_seen.clear()
run = orchestrator.run_tiled(img, "how many ships?")
check("image <= TILE_SIZE -> one result/router call; resolved request context without tile context",
      not run.tiled and len(run.results) == 1 and router_calls == ["how many ships?"]
      and len(vlm_seen) == 1 and vlm_seen[0][0] == (200, 100)
      and "IMAGE CONTEXT" not in vlm_seen[0][1] and "RESOLVED REQUEST" in vlm_seen[0][1]
      and run.run_dir == run.results[0].run_dir
      and os.path.isfile(os.path.join(run.run_dir, "tiles.json")))

config.TILE_SHARED_PLAN = False
router_calls.clear()
orchestrator.run_tiled(big_scene, "how many ships?", tile_size=2000)
check("TILE_SHARED_PLAN off -> the router runs once per tile", len(router_calls) == 3)
config.TILE_SHARED_PLAN = True

print("\n%d/%d passed" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
