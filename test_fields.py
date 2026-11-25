"""V4 contract tests: no learned models, network, credentials or GPU.

Run: python -X utf8 -m unittest -v test_fields
The four landscape fixtures test measured semantic acceptance, not model accuracy.
"""

import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

import config
import drawing
import field_geometry as geo
import masks
import orchestrator
import planner
import vlm
from schemas import FieldOutput, Plan, SegmentationOutput, RequestIntent, RequestTarget
from tools import delineate, segmentation, unibuilding
from tools.worker_client import ToolWorker, ToolWorkerError


def polygon(x1, y1, x2, y2, hole=None):
    rings = [[[x1,y1], [x2,y1], [x2,y2], [x1,y2], [x1,y1]]]
    if hole:
        rings += polygon(*hole)["coordinates"]
    return {"type": "Polygon", "coordinates": rings}


def seg_fixture(mask=None, classes=None):
    classes = classes or ["background"]+list(config.SEMANTIC_CLASSES.values())
    names = [c.split(",")[0] for c in classes]
    mask = np.full((100,100), names.index("cropland"), np.uint8) if mask is None else mask
    return SegmentationOutput(classes, names, [[(i*23)%256, (i*71)%256, (i*41)%256]
        for i in range(len(names))], {n: float((mask == i).mean()*100) for i,n in enumerate(names)},
        mask, "")


def output_fixture(seg=None):
    seg = seg or seg_fixture()
    raw = [{"id":"f0", "geometry":polygon(10,10,90,90, (30,30,60,60))},
           {"id":"f1", "geometry":polygon(0,0,8,8)}]
    checked, kept, rejected = geo.verify_fields(raw, seg)
    return FieldOutput(checked, kept, rejected, "", "", metadata=geo.source_metadata((100,100)),
                       filter_config=geo.native_filter_config(seg))


class RoutingTests(unittest.TestCase):
    def test_whole_image_classification_uses_configured_classes_without_changing_focused_scope(self):
        for question in ("classify image","classify the image","classify this scene",
                         "create a classification map","create a classification map of this image",
                         "Can you classify the uploaded satellite image please?",
                         "I would like a classification map for this scene",
                         "Use predefined classes to classify the image",
                         "این تصویر را طبقه بندی کن"):
            req=RequestIntent(operation="classify")
            p=planner.compile_intent(req,question)
            self.assertEqual(p.tools,["segmentation","unibuilding"])
            self.assertEqual(p.segmentation_classes,[c for c in config.DEFAULT_LULC_CLASSES
                if not segmentation.is_building_class(c)])
            self.assertEqual(p.request.land_cover_scope,"general_scene")
            self.assertEqual(p.output_kind,"classification_map")
            self.assertEqual(p.output,"tool_output")
            self.assertTrue(p.protect_semantic_buildings)
            self.assertEqual(p.request.output_classes,["buildings","grass","tree","meadow","road","water","bareland"])
            self.assertEqual(len(p.request.output_classes),7)
            self.assertNotIn("industrial area",p.request.output_classes)
            self.assertEqual(planner.apply_rules(p,question).model_dump(),p.model_dump())
        for category in ("vegetation","vehicle types","minerals","roof materials","water bodies"):
            question="Create a classification map of "+category
            self.assertFalse(planner._unrestricted_classification(question))
            with self.assertRaises(planner.PlanningError):
                planner.compile_intent(RequestIntent(operation="classify",land_cover_scope="general_scene",
                    targets=[RequestTarget(kind="land_cover",category="scene")]),question)

    def test_continuous_cover_cannot_be_sent_to_detector_by_a_wrong_kind(self):
        for cls in ("road", "roads", "water", "forest", "tree", "grass", "farmland",
                    "bare soil", "sand", "parking lot", "highway"):
            for display in ("boxes","mask","classification_map"):
                req=RequestIntent(operation="classify",presentation=display,
                    targets=[RequestTarget(kind="object",classes=[cls])])
                plan=planner.compile_intent(req,"Classify "+cls)
                self.assertEqual(plan.tools,["segmentation","unibuilding"],cls)
                self.assertEqual(plan.segmentation_classes,["background",cls])
                self.assertEqual(plan.request.targets[0].kind,"land_cover")
                self.assertEqual(plan.request.land_cover_scope,"requested_classes")
                self.assertEqual(planner.apply_rules(plan,"Classify "+cls).model_dump(),plan.model_dump())
        req=RequestIntent(operation="classify",targets=[RequestTarget(kind="object",
            category="roads",classes=["paved road","unpaved road"])])
        self.assertEqual(planner.compile_intent(req,"Classify roads").tools,["segmentation","unibuilding"])

    def test_buildings_and_roads_use_separate_pixel_tools_even_with_wrong_router_kind(self):
        req=RequestIntent(operation="classify",presentation="classification_map",
            output_classes=["building","road","background/unclassified"],targets=[
                RequestTarget(kind="building",classes=["building"]),
                RequestTarget(kind="object",category="roads",classes=["road"])])
        for reasoning in (False,True):
            p=planner.compile_intent(req.model_copy(update={"requires_reasoning":reasoning}),
                "create a classification map of buildings and roads")
            self.assertEqual(p.tools,["segmentation","unibuilding"])
            self.assertEqual(p.segmentation_classes,["background","road"])
            self.assertEqual(p.detection_classes,[])
            self.assertEqual(p.request.output_classes,["building","road"])
            self.assertEqual(p.output,"vlm_answer" if reasoning else "tool_output")

    def test_subject_scope_and_presentation_are_independent_for_arbitrary_categories(self):
        for category, kind, classes, tools in (
            ("vegetation", "land_cover", ["reed","mangrove"], ["segmentation","unibuilding"]),
            ("water bodies", "land_cover", ["river","lake","pond"], ["segmentation","unibuilding"]),
            ("vehicle types", "object", ["car","truck","bus"], ["detection"]),
            ("infrastructure", "object", ["crane","excavator"], ["detection"]),
            ("building use", "building", ["residential","industrial"], ["unibuilding"]),
        ):
            plans=[]
            for display in ("boxes","mask","classification_map"):
                request=RequestIntent(operation="classify",presentation=display,
                    land_cover_scope="requested_classes" if kind=="land_cover" else "none",
                    targets=[RequestTarget(kind=kind,category=category,classes=classes)])
                p=planner.compile_intent(request,"Classify "+category)
                self.assertEqual(p.tools,tools)
                self.assertEqual(p.segmentation_classes,["background"]+classes if kind=="land_cover" else [])
                self.assertEqual(p.detection_classes,classes if kind=="object" else [])
                self.assertEqual(p.output,"tool_output" if kind=="land_cover" else "vlm_answer")
                self.assertEqual(planner.apply_rules(p,"Classify "+category).model_dump(),p.model_dump())
                plans.append((p.tools,p.segmentation_classes,p.detection_classes,p.output))
            self.assertEqual(plans[0],plans[1])
            self.assertEqual(plans[1],plans[2])

    def test_generic_vocabulary_requires_explicit_scope_and_targeted_classes_cannot_be_empty(self):
        for category in ("vegetation","vehicle type","surface material","water bodies"):
            invalid=RequestIntent(operation="classify",land_cover_scope="general_scene")
            with self.assertRaises(planner.PlanningError):
                planner.compile_intent(invalid,"Create classification map of "+category)
            invalid=RequestIntent(operation="classify",land_cover_scope="requested_classes",
                targets=[RequestTarget(kind="land_cover",category=category)])
            with self.assertRaises(planner.PlanningError):
                planner.compile_intent(invalid,"Create classification map of "+category)

    def test_attribute_output_classes_require_reasoning_without_changing_subject_tool(self):
        req=RequestIntent(operation="classify",land_cover_scope="requested_classes",
            targets=[RequestTarget(kind="land_cover",category="vegetation",classes=["grass","tree"])],
            classification_basis="condition",output_classes=["healthy","damaged"])
        p=planner.compile_intent(req,"Classify vegetation condition")
        self.assertEqual(p.tools,["segmentation","unibuilding"])
        self.assertEqual(p.output,"vlm_answer")
        self.assertEqual(p.segmentation_classes,["background","grass","tree"])
    def test_mixed_subjects_keep_their_own_tools_and_context_role(self):
        request=RequestIntent(operation="classify",presentation="classification_map",
            land_cover_scope="requested_classes",requires_reasoning=True,
            targets=[RequestTarget(kind="land_cover",category="vegetation",classes=["grass","tree"]),
                     RequestTarget(kind="building",category="building",role="context")])
        p=planner.compile_intent(request,"Classify vegetation near buildings")
        self.assertEqual(set(p.tools),{"segmentation","unibuilding"})
        self.assertEqual(p.segmentation_classes,["background","grass","tree"])
        self.assertEqual(p.output,"vlm_answer")

    def test_invalid_scope_is_repaired_by_semantic_router_not_topic_dictionary(self):
        invalid=RequestIntent(operation="classify",land_cover_scope="general_scene")
        valid=RequestIntent(operation="classify",land_cover_scope="requested_classes",
            presentation="classification_map",targets=[RequestTarget(kind="land_cover",
                category="surface materials",classes=["asphalt","gravel","concrete"])])
        with patch.object(planner,"chat_json",side_effect=[(invalid,"{}"),(valid,"{}")]) as router:
            p=planner.make_plan("Create a classification map of surface materials")
        self.assertEqual(router.call_count,2)
        self.assertEqual(p.segmentation_classes,["background","asphalt","gravel","concrete"])
        self.assertIs(router.call_args.args[2],RequestIntent)

    def test_semantic_building_pixels_do_not_start_building_detection_for_coverage(self):
        q="What percentage of this image is farmland?"
        p=planner.enforce_building_routing(planner.apply_rules(Plan(),q),q)
        self.assertEqual(p.tools,["segmentation","unibuilding"])
        self.assertEqual(p.segmentation_classes,["background",planner._default_entry_for("cropland")])

    def test_required_routes_with_hostile_router(self):
        cases = [
            ("Delineate all agricultural fields", {"segmentation","delineate"}),
            ("Agricultural fields in an unknown mixed landscape", {"segmentation","delineate"}),
            ("Find agricultural fields and exclude urban parks", {"segmentation","delineate"}),
            ("Agricultural fields near buildings", {"segmentation","delineate","unibuilding"}),
            ("Agricultural fields containing vehicles", {"segmentation","delineate","detection"}),
            ("Agricultural fields near buildings and vehicles", {"segmentation","delineate","detection","unibuilding"}),
            ("What percentage is farmland?", {"segmentation"}),
            ("Agricultural coverage", {"segmentation"}),
            ("Urban green spaces and parks", {"segmentation"}),
            ("Identify urban green spaces and football fields", {"segmentation","detection"}),
            ("Find sports facilities", {"detection"}),
        ]
        for question, tools in cases:
            with self.subTest(question=question):
                seed = Plan(use_delineate=True, output="tool_output")
                p = planner.apply_rules(seed, question)
                expected=tools | ({"unibuilding"} if "segmentation" in tools else set())
                self.assertEqual(set(p.tools), expected)
                if p.use_delineate:
                    self.assertEqual(p.output, "vlm_answer")
                    self.assertEqual(p.segmentation_classes,["background"]+list(config.SEMANTIC_CLASSES.values()))

    def test_format_is_independent(self):
        q = "Agricultural fields near buildings and vehicles"
        a = planner.apply_rules(Plan(output_kind="boxes"), q)
        b = planner.apply_rules(Plan(output_kind="mask"), q+" as a binary mask")
        self.assertEqual(a.tools, b.tools)
        self.assertEqual(a.segmentation_classes, b.segmentation_classes)
        self.assertEqual(a.output, b.output)
        self.assertEqual(b.output_kind, "mask")

    def test_boundaries_cannot_be_plain_map(self):
        p = planner.apply_rules(Plan(output="tool_output"), "Create a segmentation map of agricultural field boundaries")
        self.assertTrue(p.use_delineate)
        self.assertEqual(p.output, "vlm_answer")

    def test_buildings_remain_unibuilding(self):
        p = planner.apply_rules(Plan(use_detection=True, detection_classes=["house"],
                                   use_segmentation=True, segmentation_classes=["roof"]), "show buildings")
        self.assertEqual(p.tools, ["unibuilding"])


class EvidenceTests(unittest.TestCase):
    def test_orchard_is_distinct_from_cropland_and_supplies_agricultural_evidence(self):
        s=seg_fixture()
        self.assertNotEqual(s.class_names.index("cropland"),s.class_names.index("orchard"))
        s.mask[:]=s.class_names.index("orchard")
        fields=output_fixture(s)
        self.assertEqual(fields.fields[0]["agricultural_evidence"]["status"],"supported")
        self.assertEqual(fields.fields[0]["agricultural_evidence"]["cropland_fraction"],0)
        self.assertEqual(fields.fields[0]["agricultural_evidence"]["agricultural_cover_fraction"],1)

    def test_dynamic_id_mapping_not_dictionary_order(self):
        classes = ["background"] + list(reversed(list(config.SEMANTIC_CLASSES.values())))
        s = seg_fixture(classes=classes)
        cfg = geo.native_filter_config(s)
        self.assertEqual(cfg["range"], 13)
        self.assertEqual(cfg["filter_classes"], [s.class_names.index(n) for n in
            ("buildings","road","urban green space","sports field","industrial area","water")])
        self.assertNotIn(s.class_names.index("meadow"), cfg["filter_classes"])
        self.assertNotIn(s.class_names.index("tree"), cfg["filter_classes"])

    def test_four_landscapes(self):
        s = seg_fixture()
        idx = {n:i for i,n in enumerate(s.class_names)}
        results = {}
        for landscape, cls, expected in (
            ("agricultural_only", "cropland", "supported"),
            ("urban_parks_lawns", "urban green space", "rejected_nonagricultural"),
            ("industrial", "industrial area", "rejected_nonagricultural"),
            ("mixed_urban_rural", "meadow", "uncertain")):
            with self.subTest(landscape=landscape):
                s.mask[:] = idx[cls]
                if landscape == "mixed_urban_rural":
                    s.mask[:20] = idx["urban green space"]
                    s.mask[80:] = idx["cropland"]
                f = output_fixture(s)
                self.assertEqual(f.raw_fields[0]["agricultural_evidence"]["status"], expected)
                results[landscape] = f.raw_fields[0]["agricultural_evidence"]
        Path("validation/landscape_contract_results.json").write_text(json.dumps(results, indent=2))

    def test_pasture_fallow_forest_and_unknown_are_uncertain_not_rejected(self):
        for name in ("meadow","bareland","tree","background"):
            s = seg_fixture()
            s.mask[:] = s.class_names.index(name)
            f = output_fixture(s)
            self.assertEqual(len(f.fields), 2)
            self.assertEqual(f.fields[0]["agricultural_evidence"]["status"], "uncertain")

    def test_invalid_mapping_fails_closed(self):
        s = seg_fixture()
        s.class_names = list(reversed(s.class_names))
        with self.assertRaises(ValueError):
            geo.native_filter_config(s)
        s = seg_fixture()
        s.mask[:] = 99
        with self.assertRaises(ValueError):
            geo.native_filter_config(s)

    def test_adapter_detects_silent_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d)/"mask.png")
            Image.new("L",(100,100)).save(path)
            image = str(Path(d)/"rgb.png")
            Image.new("RGB",(100,100)).save(image)
            response = {"mask_path":path, "class_names":["background","building"]}
            with patch.object(segmentation, "worker") as worker:
                worker.return_value.call.return_value = response
                with self.assertRaisesRegex(ValueError, "mapping"):
                    segmentation.segment(image, config.DEFAULT_LULC_CLASSES, d)


class GeometryTests(unittest.TestCase):
    def test_holes_and_multipart_masks(self):
        f = output_fixture().fields[0]
        mask = geo.field_mask([f], (100,100))
        self.assertEqual(mask.sum(), 80*80-30*30)
        self.assertFalse(mask[45,45])
        multi = {"id":"f8", "geometry":{"type":"MultiPolygon", "coordinates":[
            polygon(1,1,5,5)["coordinates"], polygon(10,10,15,15)["coordinates"]]}}
        self.assertEqual(geo.field_mask([multi],(100,100)).sum(), 16+25)

    def test_alignment_preserves_geometry_and_uncertainty(self):
        s = seg_fixture()
        s.mask[:] = s.class_names.index("meadow")
        fields = output_fixture(s)
        props = vlm.proposals(None, s, fields=fields)
        obj = vlm.align([{"id":"f0", "box":[0,0,1,1],"selection_status":"supported"},
                         {"id":"f0", "note":"split"}], props)
        self.assertEqual(len(obj), 1)
        self.assertEqual(obj[0]["geometry"], fields.fields[0]["geometry"])
        self.assertEqual(obj[0]["box"], fields.fields[0]["bbox"])
        self.assertEqual(obj[0]["selection_status"], "uncertain")
        all_kept = vlm.keep_by_default(["agricultural field"], props, [], [])
        self.assertEqual(all_kept[0]["geometry"], fields.fields[0]["geometry"])

    def test_mask_uses_polygon_instead_of_bbox(self):
        f = output_fixture()
        obj = vlm.align([{"id":"f0"}], vlm.proposals(None,None,fields=f))
        mask, note = masks.from_decision(obj, [], None, (100,100))
        self.assertEqual(mask.sum(), 5500)
        self.assertIn("original field polygon", note)
        picture = drawing.draw_vlm_objects(Image.new("RGB",(100,100)), obj)
        self.assertEqual(picture.size,(100,100))

    def test_geo_metadata_rotated_affine_and_pixel_reference(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d)/"source.tif")
            transform = Affine(2, .3, 500000, .2, -2, 4000000)
            with rasterio.open(path,"w",driver="GTiff",width=100,height=100,count=3,
                               dtype="uint8",crs="EPSG:32639",transform=transform) as dst:
                dst.write(np.ones((3,100,100),np.uint8)*90)
            image = geo.read_image(path)
            meta = geo.source_metadata(image.size,path)
            out = geo.write_lclu(seg_fixture(),str(Path(d)/"lclu.tif"),meta)
            with rasterio.open(out) as src:
                self.assertEqual(src.transform,transform)
                self.assertEqual(src.crs.to_epsg(),32639)
                self.assertEqual(src.shape,(100,100))
            geom = polygon(1,2,5,8)
            world = geo.transform_geometry(geom,transform)
            back = geo.transform_geometry(world,~transform)
            np.testing.assert_allclose(back["coordinates"],geom["coordinates"],atol=1e-8)
            pngmeta = geo.source_metadata((100,100))
            self.assertIsNone(pngmeta["crs_wkt"])
            self.assertEqual(pngmeta["coordinate_reference"],"pixel")

    def test_context_containment_respects_holes(self):
        from schemas import DetectionOutput
        f = output_fixture()
        det = DetectionOutput(["small vehicle"],[],[
            {"id":"d0","label":"small vehicle","bbox":[.4,.4,.5,.5],"score":.1},
            {"id":"d1","label":"small vehicle","bbox":[.1,.1,.2,.2],"score":.1}],0)
        relations = vlm.field_context_relations(f,det,None,(100,100))["f0"]
        self.assertEqual([d["id"] for d in relations["inside"]],["d1"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        def empty_buildings(image_path,run_dir):
            with Image.open(image_path) as image:
                return unibuilding.from_mask(np.zeros((image.height,image.width),bool))
        guard=patch.object(orchestrator.unibuilding,"detect",side_effect=empty_buildings)
        guard.start()
        self.addCleanup(guard.stop)

    def test_focused_cover_uses_original_rgb_and_excludes_buildings_before_statistics_and_regions(self):
        pixels=np.zeros((100,100),bool)
        pixels[10:40,10:40]=True
        uni=unibuilding.from_mask(pixels)
        original=Image.new("RGB",(100,100),(71,93,126))
        events=[]
        def building(image_path,run_dir):
            events.append("building")
            return uni
        def semantic(image_path,classes,run_dir,**kwargs):
            events.append("semantic")
            np.testing.assert_array_equal(np.asarray(Image.open(image_path)),np.asarray(original))
            self.assertEqual(classes,["background","grass","tree"])
            return seg_fixture(np.ones((100,100),np.uint8),classes)
        req=RequestIntent(operation="classify",presentation="classification_map",
            targets=[RequestTarget(kind="land_cover",category="vegetation",classes=["grass","tree"])])
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.unibuilding,"detect",side_effect=building), \
             patch.object(segmentation,"segment",side_effect=semantic), \
             patch.object(vlm,"answer") as reasoning:
            r=orchestrator.run_agent(original,"Classify vegetation",run_dir=d,
                plan=planner.compile_intent(req,"Classify vegetation"))
            self.assertEqual(events,["building","semantic"])
            reasoning.assert_not_called()
            self.assertEqual(r.segmentation.coverage["grass"],91)
            self.assertEqual(sum(region["area_px"] for region in r.segmentation.regions),9100)
            self.assertTrue(np.all(r.segmentation.mask[pixels]==0))
            out=np.asarray(Image.open(r.output_image_path))
            self.assertTrue(np.all(out[pixels]==0))
            self.assertEqual(np.count_nonzero(out),9100)
            meta=json.loads((Path(d)/"classification_map.json").read_text())
            self.assertEqual(meta["classes"],["grass","tree"])
            self.assertEqual(r.semantic_building_guard["nonbuilding_pixels_removed"],900)

    def test_general_scene_building_pixels_come_exclusively_from_unibuilding(self):
        pixels=np.zeros((100,100),bool); pixels[25:45,25:45]=True
        uni=unibuilding.from_mask(pixels)
        req=RequestIntent(operation="classify")
        plan=planner.compile_intent(req,"classify image")
        classes=plan.segmentation_classes
        raw=np.full((100,100),[c.split(",")[0] for c in classes].index("meadow"),np.uint8)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.unibuilding,"detect",return_value=uni), \
             patch.object(segmentation,"segment",return_value=seg_fixture(raw,classes)) as segment, \
             patch.object(vlm,"answer") as reasoning:
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"classify image",run_dir=d,plan=plan)
            self.assertFalse(any(segmentation.is_building_class(c) for c in segment.call_args.args[1]))
            reasoning.assert_not_called()
            meta=json.loads((Path(d)/"classification_map.json").read_text())
            self.assertEqual(meta["classes"],["buildings","grass","tree","meadow","road","water","bareland"])
            building_id=next(int(i) for i,name in meta["class_id_mapping"].items() if name=="buildings")
            out=np.asarray(Image.open(r.output_image_path))
            np.testing.assert_array_equal(out==building_id,pixels)
            self.assertNotIn("unknown",meta["classes"])
            plain=Image.open(r.output_image_path).convert("RGB")
            legend=Image.open(meta["legend_image_path"])
            self.assertGreater(legend.width,plain.width)
            np.testing.assert_array_equal(np.asarray(legend)[:100,:100],np.asarray(plain))
            with Image.open(meta["comparison_image_path"]) as comparison:
                self.assertGreater(comparison.width,legend.width+100)
                self.assertGreater(comparison.height,100)

    def test_building_protection_failure_blocks_semantic_inference_and_map(self):
        req=RequestIntent(operation="classify",presentation="classification_map",
            targets=[RequestTarget(kind="land_cover",classes=["grass"])])
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.unibuilding,"detect",side_effect=RuntimeError("model unavailable")), \
             patch.object(segmentation,"segment") as semantic, \
             patch.object(vlm,"answer") as reasoning:
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Classify grass",run_dir=d,
                plan=planner.compile_intent(req,"Classify grass"))
            semantic.assert_not_called(); reasoning.assert_not_called()
            self.assertFalse(r.output_image_path)
            self.assertEqual(r.answered_by,"Required semantic evidence failed")

    def test_delineate_receives_building_fused_semantics_before_polygon_checks(self):
        pixels=np.ones((100,100),bool); pixels[90:]=False
        uni=unibuilding.from_mask(pixels)
        observed=[]
        def field(image_path,semantic,*args):
            observed.append(semantic)
            return output_fixture(semantic)
        req=RequestIntent(operation="delineate",targets=[RequestTarget(kind="agricultural_field")])
        p=planner.compile_intent(req,"Delineate agricultural fields")
        classes=[c for c in p.segmentation_classes if not segmentation.is_building_class(c)]
        raw=seg_fixture(np.full((100,100),[c.split(",")[0] for c in classes].index("cropland"),np.uint8),classes)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.unibuilding,"detect",return_value=uni), \
             patch.object(segmentation,"segment",return_value=raw), \
             patch.object(delineate,"delineate",side_effect=field), \
             patch.object(delineate,"export",return_value="fields.gpkg"), \
             patch.object(vlm,"answer",return_value=vlm.VLMReply(answer="No fields")):
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Delineate agricultural fields",plan=p,run_dir=d)
            self.assertTrue(np.all(observed[0].mask[pixels]==observed[0].class_names.index("buildings")))
            self.assertEqual(len(r.fields.fields),0)
            self.assertEqual(len(r.fields.rejected),2)

    def test_combined_map_is_one_native_class_id_mask_with_building_overlap_priority(self):
        from tools import unibuilding
        road=np.zeros((100,100),np.uint8)
        road[20:40,:]=1
        binary=np.zeros((100,100),bool)
        binary[10:45,10:16]=True
        binary[39:45,10:35]=True
        uni=unibuilding.from_mask(binary)
        req=RequestIntent(operation="classify",presentation="classification_map",
            output_classes=["building","road"],targets=[
                RequestTarget(kind="building",classes=["building"]),
                RequestTarget(kind="object",classes=["road"])])
        expected=road*2
        expected[binary]=1
        with tempfile.TemporaryDirectory() as d, \
             patch.object(segmentation,"segment",return_value=seg_fixture(road,["background","road"])) as seg, \
             patch.object(orchestrator.unibuilding,"detect",return_value=uni), \
             patch.object(orchestrator.detection,"detect") as detector, \
             patch.object(vlm,"answer") as reasoning:
            r=orchestrator.run_agent(Image.new("RGB",(100,100),(124,91,42)),
                "create a classification map of buildings and roads",run_dir=d,
                plan=planner.compile_intent(req,"create a classification map of buildings and roads"))
            detector.assert_not_called()
            reasoning.assert_not_called()
            self.assertEqual(seg.call_args.args[1],["background","road"])
            png=Image.open(r.output_image_path)
            self.assertEqual(png.mode,"P")
            self.assertEqual(png.size,(100,100))
            np.testing.assert_array_equal(np.asarray(png),expected)
            with rasterio.open(Path(d)/"classification_ids.tif") as tif:
                np.testing.assert_array_equal(tif.read(1),expected)
                self.assertEqual(json.loads(tif.tags()["class_id_mapping"]),
                    {"0":"background/unclassified","1":"building","2":"road"})
            self.assertFalse(any(s.status=="error" for s in r.steps))

    def test_vlm_cannot_paint_a_road_box_when_no_road_pixels_exist(self):
        from tools import unibuilding
        uni=unibuilding.from_mask(np.zeros((100,100),bool))
        req=RequestIntent(operation="classify",presentation="classification_map",requires_reasoning=True,
            output_classes=["road"],targets=[RequestTarget(kind="object",classes=["road"])])
        reply=vlm.VLMReply(answer="Review",objects=[{"origin":"vlm","class":"road",
            "label":"road","box":[0,0,1,1],"source_id":None}],target_classes=["road"])
        with tempfile.TemporaryDirectory() as d, \
             patch.object(segmentation,"segment",return_value=seg_fixture(
                 np.zeros((100,100),np.uint8),["background","road"])), \
             patch.object(vlm,"answer",return_value=reply), \
             patch.object(drawing,"draw_boxes") as boxes:
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Classify roads carefully",run_dir=d,
                plan=planner.compile_intent(req,"Classify roads carefully"))
            self.assertEqual(r.vlm_objects,[])
            self.assertEqual(np.count_nonzero(np.asarray(Image.open(r.output_image_path))),0)
            boxes.assert_not_called()

    def test_detection_without_pixels_stays_unclassified_instead_of_becoming_a_rectangle(self):
        from schemas import DetectionOutput
        req=RequestIntent(operation="classify",presentation="classification_map",output_classes=["car"],
            targets=[RequestTarget(kind="object",classes=["car"])])
        det=DetectionOutput(["car"],[],[{"id":"d0","label":"car","bbox":[.1,.1,.9,.9],"score":.5}],0)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.detection,"detect",return_value=det), \
             patch.object(segmentation,"segment",return_value=seg_fixture(
                 np.zeros((100,100),np.uint8),["background","car"])), \
             patch.object(vlm,"chat_json",return_value=(vlm.VLMReply(answer="Reviewed",keep_all_classes=["car"]),"")):
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Classify cars",run_dir=d,
                plan=planner.compile_intent(req,"Classify cars"))
            self.assertEqual(np.count_nonzero(np.asarray(Image.open(r.output_image_path))),0)
            metadata=json.loads((Path(d)/"classification_map.json").read_text())
            self.assertEqual(metadata["unrasterized"][0]["id"],"d0")
            self.assertIn("left unclassified",r.output_caption)

    def test_categorical_geotiff_preserves_source_affine_and_tile_offset(self):
        import tiling
        transform=Affine(2,.3,500000,.2,-2,4000000)
        req=RequestIntent(operation="classify",presentation="classification_map",
            targets=[RequestTarget(kind="land_cover",classes=["water"])])
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"scene.tif"
            with rasterio.open(source,"w",driver="GTiff",width=200,height=200,count=3,
                    dtype="uint8",crs="EPSG:32639",transform=transform) as dst:
                dst.write(np.zeros((3,200,200),np.uint8))
            tile=tiling.make_tiles((200,200),100)[1]
            with patch.object(segmentation,"segment",return_value=seg_fixture(
                    np.ones((100,100),np.uint8),["background","water"])):
                r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Classify water",run_dir=Path(d)/"out",
                    plan=planner.compile_intent(req,"Classify water"),tile=tile,source_path=str(source))
            with rasterio.open(Path(r.run_dir)/"classification_ids.tif") as src:
                self.assertEqual(src.crs.to_epsg(),32639)
                self.assertEqual(src.transform,transform*Affine.translation(tile.x0,tile.y0))
                self.assertEqual(src.shape,(100,100))

    def test_field_classification_raster_preserves_polygon_holes(self):
        from schemas import AgentResult
        req=RequestIntent(operation="classify",presentation="classification_map",output_classes=["cropland"],
            targets=[RequestTarget(kind="agricultural_field")])
        fields=output_fixture()
        obj=vlm.align([{"id":"f0","classification_label":"cropland"}],vlm.proposals(None,None,fields=fields))
        with tempfile.TemporaryDirectory() as d:
            result=AgentResult("Classify fields",planner.compile_intent(req,"Classify fields"),
                fields=fields,vlm_objects=obj,run_dir=d)
            orchestrator._render_classification_map(result,Image.new("RGB",(100,100)),[])
            pixels=np.asarray(Image.open(result.output_image_path))
            np.testing.assert_array_equal(pixels,geo.field_mask([fields.fields[0]],(100,100)).astype("uint8"))

    def test_context_classes_cannot_become_extra_classification_targets(self):
        from schemas import DetectionOutput
        request=RequestIntent(operation="classify",presentation="classification_map",
            land_cover_scope="requested_classes",constraints=["near ships"],
            targets=[RequestTarget(kind="object",classes=["car"]),
                     RequestTarget(kind="object",classes=["ship"],role="context"),
                     RequestTarget(kind="land_cover",classes=["water"],role="context")])
        detections=DetectionOutput(["car","ship"],[],[
            {"id":"d0","label":"car","bbox":[.1,.1,.3,.3],"score":.5},
            {"id":"d1","label":"ship","bbox":[.5,.5,.9,.8],"score":.6}],0)
        semantic=seg_fixture(np.ones((100,100),np.uint8),["background","water"])
        semantic.regions=[{"id":"r0","class":"water","bbox":[0,0,1,1],"area_px":10000}]
        car_pixels=np.zeros((100,100),np.uint8)
        car_pixels[15:20,15:22]=1
        object_pixels=seg_fixture(car_pixels,["background","car"])
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.detection,"detect",return_value=detections), \
             patch.object(segmentation,"segment",side_effect=[semantic,object_pixels]), \
             patch.object(vlm,"chat_json",return_value=(vlm.VLMReply(answer="Classified target",
                 objects=[{"id":"d0"},{"id":"d1"},{"id":"r0"}],target_classes=["water"]),"")):
            p=planner.compile_intent(request,"Classify cars near ships on water")
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Classify cars near ships on water",run_dir=d,plan=p)
            self.assertEqual([o["source_id"] for o in r.vlm_objects],["d0"])
            data=json.loads((Path(d)/"classification_map.json").read_text())
            self.assertEqual(data["classes"],["car"])
            self.assertEqual(data["evidence"],["object segmentation pixels inside detections"])
            self.assertEqual(np.count_nonzero(np.asarray(Image.open(r.output_image_path))),35)

    def test_field_context_does_not_suppress_requested_objects_or_add_field_counts(self):
        from schemas import DetectionOutput
        fields=output_fixture()
        det=DetectionOutput(["car"],[],[{"id":"d0","label":"car","bbox":[.1,.1,.2,.2],"score":.5}],0)
        reply=vlm.VLMReply(answer="Vehicle classified",objects=[{"id":"f0"},{"id":"d0"}])
        def transport(model, messages, schema, **kwargs):
            if schema is vlm.FieldReviewPolicy:
                return vlm.FieldReviewPolicy(),""
            if schema is vlm.FieldBatchReply:
                data=json.loads(messages[1]["content"][-1]["text"])
                return vlm.FieldBatchReply(judgments=[vlm.FieldJudgment(id=f["id"],decision="keep")
                    for f in data["fields"]]),""
            if schema is vlm.FieldSummaryReply:
                return vlm.FieldSummaryReply(answer="Fields reviewed"),""
            return reply,""
        with patch.object(vlm,"chat_json",side_effect=transport):
            answer=vlm.answer(Image.new("RGB",(100,100)),"Classify vehicles within measured fields",det,
                None,fields=fields,field_task=True,field_target_origins=["detector"])
        self.assertEqual([o["source_id"] for o in answer.objects],["d0"])
        self.assertNotIn("Field evidence:",answer.answer)

    def test_object_classification_uses_detector_then_vlm_then_measured_pixels(self):
        from schemas import DetectionOutput
        request=RequestIntent(operation="classify",presentation="classification_map",
            classification_basis="vehicle type",output_classes=["car","bus"],
            targets=[RequestTarget(kind="object",category="vehicle",classes=["car","bus"])])
        detections=DetectionOutput(["car","bus"],[],[
            {"id":"d0","label":"car","bbox":[.1,.1,.3,.3],"score":.5},
            {"id":"d1","label":"bus","bbox":[.5,.5,.9,.8],"score":.6}],0)
        pixels=np.zeros((100,100),np.uint8)
        pixels[12:22,13:23]=1
        pixels[55:65,55:70]=2
        pixel_output=seg_fixture(pixels,["background","car","bus"])
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.detection,"detect",return_value=detections) as detector, \
             patch.object(segmentation,"segment",return_value=pixel_output) as semantic, \
             patch.object(orchestrator.unibuilding,"detect") as uni, \
             patch.object(vlm,"chat_json",return_value=(vlm.VLMReply(answer="Types classified",
                 keep_all_classes=["car","bus"]),"")) as reasoning:
            p=planner.compile_intent(request,"Create classification map of vehicle types")
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Create classification map of vehicle types",
                run_dir=d,plan=p)
            detector.assert_called_once()
            semantic.assert_called_once()
            self.assertEqual(semantic.call_args.args[1],["background","car","bus"])
            self.assertFalse(p.use_segmentation)
            uni.assert_not_called()
            reasoning.assert_called_once()
            self.assertTrue(Path(r.output_image_path).is_file())
            data=json.loads((Path(d)/"classification_map.json").read_text())
            self.assertEqual(set(data["classes"]),{"car","bus"})
            self.assertEqual(data["evidence"],["object segmentation pixels inside detections"])
            np.testing.assert_array_equal(np.asarray(Image.open(r.output_image_path)),pixels)
            self.assertFalse([s for s in r.steps if s.status=="error"])

    def test_building_classification_map_uses_component_pixels_without_segearth(self):
        from tools import unibuilding
        binary=np.zeros((100,100),bool)
        binary[10:25,10:25]=True
        binary[60:85,60:85]=True
        buildings=unibuilding.from_mask(binary)
        request=RequestIntent(operation="classify",presentation="classification_map",
            classification_basis="building use",output_classes=["residential","industrial"],
            targets=[RequestTarget(kind="building",category="buildings")])
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.unibuilding,"detect",return_value=buildings) as uni, \
             patch.object(orchestrator.detection,"detect") as detector, \
             patch.object(segmentation,"segment") as semantic, \
             patch.object(vlm,"chat_json",return_value=(vlm.VLMReply(answer="Building classes",
                 objects=[{"id":"b0","label":"industrial"},{"id":"b1","label":"residential"}]),"")):
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Create a building use classification map",
                run_dir=d,plan=planner.compile_intent(request,"Create a building use classification map"))
            uni.assert_called_once()
            semantic.assert_not_called()
            detector.assert_not_called()
            data=json.loads((Path(d)/"classification_map.json").read_text())
            self.assertEqual(data["evidence"],["UniBuilding component pixels"])
            self.assertEqual(set(data["classes"]),{"residential","industrial"})
            self.assertFalse([s for s in r.steps if s.status=="error"])

    def test_focused_semantic_map_is_direct_and_cannot_expand_defaults(self):
        request=RequestIntent(operation="classify",land_cover_scope="requested_classes",
            presentation="classification_map",targets=[RequestTarget(kind="land_cover",
                category="coastal cover",classes=["water","sand"])])
        semantic=seg_fixture(np.ones((100,100),np.uint8),["background","water","sand"])
        with tempfile.TemporaryDirectory() as d, \
             patch.object(segmentation,"segment",return_value=semantic) as segment, \
             patch.object(orchestrator.detection,"detect") as detector, \
             patch.object(vlm,"answer") as reasoning:
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Create coastal cover classification map",
                run_dir=d,plan=planner.compile_intent(request,"Create coastal cover classification map"))
            self.assertEqual(segment.call_args.args[1],["background","water","sand"])
            detector.assert_not_called()
            reasoning.assert_not_called()
            self.assertTrue(Path(r.output_image_path).is_file())
            self.assertFalse([s for s in r.steps if s.status=="error"])

    def test_unresolved_classification_fails_without_any_default_tool_run(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(segmentation,"segment") as semantic, \
             patch.object(orchestrator.detection,"detect") as detector, \
             patch.object(vlm,"answer") as reasoning:
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"Create mineral composition classification map",
                run_dir=d,plan=Plan(use_segmentation=True))
            semantic.assert_not_called()
            detector.assert_not_called()
            reasoning.assert_not_called()
            self.assertEqual(r.answered_by,"Planning needs clarification")
            self.assertTrue(any(s.status=="error" for s in r.steps))

    def test_supplied_building_plan_cannot_bypass_unibuilding(self):
        from tools import unibuilding
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.detection,"detect") as detector, \
             patch.object(segmentation,"segment") as semantic, \
             patch.object(unibuilding,"detect",return_value=unibuilding.from_mask(np.zeros((100,100),bool))) as uni, \
             patch.object(vlm,"answer",return_value=vlm.VLMReply(answer="No buildings selected")):
            r=orchestrator.run_agent(Image.new("RGB",(100,100)),"detect buildings",run_dir=d,
                plan=Plan(use_detection=True,detection_classes=["building"],
                          use_segmentation=True,segmentation_classes=["background","roof"]))
            detector.assert_not_called()
            semantic.assert_not_called()
            uni.assert_called_once()
            self.assertEqual(r.plan.tools,["unibuilding"])

    def test_hung_worker_times_out_and_is_discarded(self):
        with tempfile.TemporaryDirectory() as d:
            script=Path(d)/"hung_worker.py"
            script.write_text("import sys,json,time\nfor line in sys.stdin:\n"
                " r=json.loads(line)\n if r['cmd']=='init': print(json.dumps({'ok':True,'result':{}}),flush=True)\n"
                " else: time.sleep(30)\n")
            worker=ToolWorker("timeout-fixture",sys.executable,str(script),{},d,timeout=1)
            with self.assertRaisesRegex(ToolWorkerError,"exceeded"):
                worker.call({})
            self.assertFalse(worker.alive)

    def test_seg_failure_blocks_delineate_and_invented_fields(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(orchestrator.segmentation,"segment",side_effect=RuntimeError("SAM failed")), \
             patch.object(orchestrator.delineate,"delineate") as native, \
             patch.object(orchestrator.vlm,"answer",return_value=vlm.VLMReply(answer="tool unavailable")):
            r = orchestrator.run_agent(Image.new("RGB",(100,100)), "Delineate agricultural fields",
                plan=Plan(use_delineate=True,output_kind="mask"),run_dir=d)
            native.assert_not_called()
            self.assertIsNone(r.fields)
            self.assertIsNone(r.final_mask)
            self.assertTrue(any(s.status=="error" and "Delineate" in s.name for s in r.steps))

    def test_full_scene_order_and_all_context_tools(self):
        from schemas import DetectionOutput
        from tools import unibuilding
        events = []
        def semantic(*a, **k):
            events.append("seg")
            return seg_fixture(np.full((100,300),seg_fixture().class_names.index("cropland"),np.uint8))
        def field(*a, **k):
            events.append("delineate")
            return output_fixture(semantic())
        def decide(*a, **k):
            events.append("vlm")
            self.assertTrue(k["field_task"])
            self.assertEqual(a[0].size,(300,100))
            return vlm.VLMReply(answer="checked")
        def detect(*a):
            events.append("det")
            return DetectionOutput(["small vehicle","large vehicle"],[],[],0)
        def buildings(*a):
            events.append("uni")
            return unibuilding.from_mask(np.zeros((100,300),bool))
        with tempfile.TemporaryDirectory() as d, patch.object(config,"RUNS_DIR",d), \
             patch.object(planner,"make_plan",return_value=planner.compile_intent(RequestIntent(
                 operation="delineate",targets=[RequestTarget(kind="agricultural_field"),
                 RequestTarget(kind="object",classes=["small vehicle","large vehicle"],role="context"),
                 RequestTarget(kind="building",role="context")]),"agricultural fields")), \
             patch.object(segmentation,"segment",side_effect=semantic), \
             patch.object(delineate,"delineate",side_effect=field), \
             patch.object(delineate,"export",return_value="test.gpkg"), \
             patch.object(orchestrator.detection,"detect",side_effect=detect), \
             patch.object(orchestrator.unibuilding,"detect",side_effect=buildings), \
             patch.object(vlm,"answer",side_effect=decide):
            r = orchestrator.run_tiled(Image.new("RGB",(300,100)),
                "Agricultural fields near buildings and vehicles",tile_size=100)
            self.assertEqual(len(r.results),1)
            self.assertFalse(r.tiled)
            self.assertEqual(r.strategy,"full_scene_native_delineate")
            self.assertEqual(events[0],"uni")
            self.assertLess(events.index("uni"),events.index("seg"))
            self.assertLess(events.index("seg"),events.index("delineate"))
            self.assertLess(events.index("delineate"),events.index("vlm"))
            self.assertIn("uni",events)
            self.assertIn("det",events)
            self.assertFalse([s for s in r.results[0].steps if s.status=="error"])

    def test_field_task_never_uses_r_regions_as_final_geometry(self):
        s = seg_fixture()
        s.regions=[{"id":"r0","class":"cropland","bbox":[0,0,1,1],"area_px":10000}]
        f = output_fixture(s)
        def transport(model,messages,schema,**kwargs):
            if schema is vlm.FieldReviewPolicy:
                return vlm.FieldReviewPolicy(independent_conditions=["Independent test condition"]),""
            if schema is vlm.FieldBatchReply:
                data=json.loads(messages[1]["content"][-1]["text"])
                self.assertNotIn("regions",data)
                self.assertTrue(all("geometry" not in f for f in data["fields"]))
                return vlm.FieldBatchReply(judgments=[vlm.FieldJudgment(id=f["id"],
                    decision="keep" if f["id"]=="f0" else "not_selected",reason="Independent test condition") for f in data["fields"]]),""
            return vlm.FieldSummaryReply(answer="Selected eligible fields"),""
        with patch.object(vlm,"chat_json",side_effect=transport):
            out = vlm.answer(Image.new("RGB",(100,100)),"Delineate agricultural fields",None,s,
                             fields=f,field_task=True)
        self.assertEqual([o["source_id"] for o in out.objects],["f0"])
        self.assertEqual(out.target_classes,[])


class BoundedFieldReviewTests(unittest.TestCase):
    def fixture(self, n=9, largest_last=False):
        image=Image.new("RGB",(256,256),(100,130,80))
        names=[c.split(",")[0] for c in ["background"]+list(config.SEMANTIC_CLASSES.values())]
        semantic=seg_fixture(np.full((256,256),names.index("cropland"),np.uint8))
        raw=[]
        for i in range(n):
            x,y=(i%32)*8,(i//32)*8
            raw.append({"id":"f%d"%i,"geometry":polygon(x,y,x+6,y+6)})
        if largest_last:
            raw[-1]["geometry"]=polygon(100,100,170,170,(120,120,130,130))
        checked,kept,rejected=geo.verify_fields(raw,semantic)
        fields=FieldOutput(checked,kept,rejected,"","",metadata=geo.source_metadata(image.size),
                           filter_config=geo.native_filter_config(semantic))
        request=RequestIntent(operation="delineate",targets=[RequestTarget(kind="agricultural_field")])
        return image,semantic,fields,request

    def transport(self, model, messages, schema, **kwargs):
        if schema is vlm.FieldReviewPolicy:
            return vlm.FieldReviewPolicy(),""
        if schema is vlm.FieldSummaryReply:
            return vlm.FieldSummaryReply(answer="Reviewed semantic evidence."),""
        payload=json.loads(messages[1]["content"][-1]["text"])
        return vlm.FieldBatchReply(judgments=[vlm.FieldJudgment(id=f["id"],decision="keep")
            for f in payload["fields"]]),""

    def test_all_candidates_reviewed_in_bounded_payloads_with_original_geometry(self):
        import base64,io
        image,semantic,fields,request=self.fixture(130,True)
        original=json.dumps(fields.fields,sort_keys=True)
        accepted=[]
        def transport(model,messages,schema,**kwargs):
            if schema is vlm.FieldBatchReply:
                data=json.loads(messages[1]["content"][-1]["text"])
                self.assertNotIn("policy",data)
                self.assertNotIn("limit",data["review_policy"])
                self.assertNotIn("rank_by",data["review_policy"])
                self.assertLessEqual(len(data["fields"]),7)
                self.assertLessEqual(len(messages[1]["content"][-1]["text"]),6000)
                self.assertTrue(all("geometry" not in row for row in data["fields"]))
                accepted.extend(row["id"] for row in data["fields"])
                for item in messages[1]["content"]:
                    if item["type"]=="image_url":
                        with Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"].split(",",1)[1]))) as view:
                            self.assertLessEqual(max(view.size),128)
            return self.transport(model,messages,schema,**kwargs)
        with tempfile.TemporaryDirectory() as d,patch.object(config,"VLM_FIELD_BATCH_SIZE",7), \
             patch.object(config,"VLM_FIELD_TEXT_BUDGET",6000), \
             patch.object(config,"VLM_FIELD_IMAGE_MAX_SIDE",128),patch.object(config,"VLM_FIELD_MIN_IMAGE_SIDE",32), \
             patch.object(vlm,"chat_json",side_effect=transport):
            reply=vlm.answer(image,"Delineate all agricultural fields",None,semantic,fields=fields,
                             field_task=True,request=request,run_dir=d)
            report=json.loads((Path(d)/"vlm_field_review.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["reviewed_fields"],130)
            self.assertEqual(len(accepted),len(set(accepted)))
            self.assertEqual(set(accepted),{f["id"] for f in fields.fields})
            self.assertEqual(len(reply.objects),130)
            self.assertEqual(json.dumps(fields.fields,sort_keys=True),original)
            by_id={f["id"]:f for f in fields.fields}
            self.assertTrue(all(o["geometry"]==by_id[o["source_id"]]["geometry"] for o in reply.objects))

    def test_one_id_with_apply_to_all_note_is_not_a_complete_review(self):
        image,semantic,fields,request=self.fixture()
        def incomplete(*args,**kwargs):
            if args[2] is vlm.FieldBatchReply:
                return vlm.FieldBatchReply(judgments=[vlm.FieldJudgment(id="f0",decision="keep",
                    reason="Apply this to all other fields")]),""
            return self.transport(*args,**kwargs)
        with tempfile.TemporaryDirectory() as d,patch.object(vlm,"chat_json",side_effect=incomplete):
            with self.assertRaisesRegex(vlm.LLMError,"coverage mismatch"):
                vlm.answer(image,"Review all fields",None,semantic,fields=fields,field_task=True,request=request,run_dir=d)
            report=json.loads((Path(d)/"vlm_field_review.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["reviewed_fields"],0)

    def test_missing_ids_can_be_repaired_without_implicitly_keeping_them(self):
        image,semantic,fields,request=self.fixture()
        calls=[]
        def repair(*args,**kwargs):
            if args[2] is vlm.FieldBatchReply:
                calls.append(1)
                if len(calls)==1:
                    return vlm.FieldBatchReply(judgments=[vlm.FieldJudgment(id="f0",decision="keep")]),""
            return self.transport(*args,**kwargs)
        with patch.object(vlm,"chat_json",side_effect=repair):
            reply=vlm.answer(image,"Review all fields",None,semantic,fields=fields,field_task=True,request=request)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(reply.objects),9)

    def test_all_matches_cannot_be_pruned_using_an_invented_top_one_limit(self):
        image,semantic,fields,request=self.fixture(4)
        def invalid(*args,**kwargs):
            if args[2] is vlm.FieldBatchReply:
                data=json.loads(args[1][1]["content"][-1]["text"])
                return vlm.FieldBatchReply(judgments=[vlm.FieldJudgment(id=f["id"],
                    decision="keep" if i==0 else "not_selected",reason="Outside top-1 area ranking")
                    for i,f in enumerate(data["fields"])]),""
            return self.transport(*args,**kwargs)
        with patch.object(vlm,"chat_json",side_effect=invalid):
            with self.assertRaisesRegex(vlm.LLMError,"no independent exclusion"):
                vlm.answer(image,"Review all fields",None,semantic,fields=fields,field_task=True,request=request)

    def test_context_overflow_splits_batches_and_never_drops_ids(self):
        from llm_client import ContextWindowError
        image,semantic,fields,request=self.fixture(12)
        accepted=[]
        def limited(*args,**kwargs):
            if args[2] is vlm.FieldBatchReply:
                data=json.loads(args[1][1]["content"][-1]["text"])
                if len(data["fields"])>3:
                    raise ContextWindowError("Input exceeds the context window")
                accepted.extend(f["id"] for f in data["fields"])
            return self.transport(*args,**kwargs)
        with tempfile.TemporaryDirectory() as d,patch.object(vlm,"chat_json",side_effect=limited):
            reply=vlm.answer(image,"Review all fields",None,semantic,fields=fields,field_task=True,request=request,run_dir=d)
            report=json.loads((Path(d)/"vlm_field_review.json").read_text())
            self.assertTrue(any(b["status"]=="split" for b in report["batches"]))
        self.assertEqual(len(reply.objects),12)
        self.assertEqual(len(accepted),len(set(accepted)))

    def test_single_field_overflow_reduces_only_review_images(self):
        from llm_client import ContextWindowError
        import base64,io
        image,semantic,fields,request=self.fixture(1)
        sizes=[]
        def limited(*args,**kwargs):
            if args[2] is vlm.FieldBatchReply:
                item=next(c for c in args[1][1]["content"] if c["type"]=="image_url")
                with Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"].split(",",1)[1]))) as view:
                    sizes.append(max(view.size))
                if sizes[-1]>32:
                    raise ContextWindowError("context window exceeded")
            return self.transport(*args,**kwargs)
        with patch.object(config,"VLM_FIELD_IMAGE_MAX_SIDE",128),patch.object(config,"VLM_FIELD_MIN_IMAGE_SIDE",32), \
             patch.object(vlm,"chat_json",side_effect=limited):
            reply=vlm.answer(image,"Review fields",None,semantic,fields=fields,field_task=True,request=request)
        self.assertEqual(sizes,[128,64,32])
        self.assertEqual(reply.objects[0]["geometry"],fields.fields[0]["geometry"])

    def test_ranking_is_global_after_all_batches(self):
        image,semantic,fields,request=self.fixture(11,True)
        request=request.model_copy(update={"operation":"select","constraints":["largest measured field"]})
        def ranked(*args,**kwargs):
            if args[2] is vlm.FieldReviewPolicy:
                return vlm.FieldReviewPolicy(mode="ranked",rank_by="area_px",descending=True,limit=1),""
            return self.transport(*args,**kwargs)
        with tempfile.TemporaryDirectory() as d,patch.object(config,"VLM_FIELD_BATCH_SIZE",3), \
             patch.object(vlm,"chat_json",side_effect=ranked):
            reply=vlm.answer(image,"Select largest field",None,semantic,fields=fields,field_task=True,request=request,run_dir=d)
            report=json.loads((Path(d)/"vlm_field_review.json").read_text())
        self.assertEqual([o["source_id"] for o in reply.objects],["f10"])
        self.assertEqual(report["reviewed_fields"],11)
        self.assertEqual(len(report["not_selected_ids"]),10)
        self.assertEqual(reply.rejected,[])

    def test_completed_batches_are_reused_after_later_failure(self):
        image,semantic,fields,request=self.fixture(9)
        calls=[]
        def interrupted(*args,**kwargs):
            if args[2] is vlm.FieldBatchReply:
                calls.append(1)
                if len(calls)==2:
                    raise vlm.LLMError("connection unavailable")
            return self.transport(*args,**kwargs)
        with tempfile.TemporaryDirectory() as d,patch.object(config,"VLM_FIELD_BATCH_SIZE",4), \
             patch.object(vlm,"chat_json",side_effect=interrupted):
            with self.assertRaises(vlm.LLMError):
                vlm.answer(image,"Review all fields",None,semantic,fields=fields,field_task=True,request=request,run_dir=d)
            self.assertEqual(json.loads((Path(d)/"vlm_field_review.json").read_text())["reviewed_fields"],4)
            reply=vlm.answer(image,"Review all fields",None,semantic,fields=fields,field_task=True,request=request,run_dir=d)
            report=json.loads((Path(d)/"vlm_field_review.json").read_text())
        self.assertEqual(len(reply.objects),9)
        self.assertEqual(len(calls),4)
        self.assertEqual(report["batches"][0]["status"],"cached")

    def test_api_context_error_is_not_retried_with_same_input(self):
        import llm_client
        error=Exception("Your input exceeds the context window of this model")
        error.status_code=400
        with patch.object(llm_client,"_complete",side_effect=error) as complete:
            with self.assertRaises(llm_client.ContextWindowError):
                llm_client.chat_json("model",[{"role":"system","content":"Review"},{"role":"user","content":"Fields"}],
                                     vlm.FieldBatchReply,json_mode=True,temperature=0,retries=2)
        self.assertEqual(complete.call_count,1)

    def test_resume_reuses_inference_and_preserves_original_trace(self):
        image,semantic,fields,request=self.fixture(5)
        request=request.model_copy(update={"presentation":"mask"})
        with tempfile.TemporaryDirectory(dir=Path.cwd()/"validation") as d:
            source=Path(d)/"original"; source.mkdir()
            target=Path(d)/"recovered"
            image.save(source/"input.png")
            Image.fromarray(semantic.mask).save(source/"seg_mask.png")
            binary=np.zeros(semantic.mask.shape,np.uint8)
            Image.fromarray(binary).save(source/"unibuilding_mask.png")
            uni=unibuilding.from_mask(binary>0)
            fields.lclu_path=geo.write_lclu(semantic,str(source/"segearth_lclu.tif"),fields.metadata)
            trace={"question":"Delineate all agricultural fields","plan":planner.compile_intent(request,"Delineate all agricultural fields").model_dump(),
                   "coverage":semantic.coverage,"fields":{"raw_fields":fields.raw_fields,"eligible":fields.fields,
                       "rejected":fields.rejected,"metadata":fields.metadata,"filter_config":fields.filter_config,"lclu_path":fields.lclu_path}}
            trace["unibuilding"]={"buildings":uni.buildings,"threshold":uni.threshold,
                "min_area_px":uni.min_area_px,"connectivity":uni.connectivity,
                "mask_path":str(source/"unibuilding_mask.png")}
            trace["semantic_building_guard"]={"building_source":"UniBuilding","building_pixels":0}
            original=json.dumps(trace)
            (source/"trace.json").write_text(original)
            with patch.object(orchestrator.segmentation,"segment") as seg_model, \
                 patch.object(orchestrator.delineate,"delineate") as field_model, \
                 patch.object(orchestrator.delineate,"export",return_value=str(target/"fields.gpkg")), \
                 patch.object(vlm,"chat_json",side_effect=self.transport):
                result=orchestrator.resume_decision(source,target.relative_to(Path.cwd()))
            seg_model.assert_not_called(); field_model.assert_not_called()
            self.assertEqual(len(result.vlm_objects),5)
            self.assertEqual((source/"trace.json").read_text(),original)
            self.assertEqual(result.resume_source,str(source.resolve()))
            self.assertTrue(Path(result.run_dir).is_absolute())
            self.assertTrue((target/"final_mask.png").is_file())
            self.assertFalse([s for s in result.steps if s.status=="error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
