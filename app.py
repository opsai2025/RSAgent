"""
Remote-sensing agent V3 -- Streamlit UI.

    conda activate SAM
    streamlit run app.py

The UI only collects input and shows the TiledRun: how the image was cut, and
for every tile its own AgentResult. All logic is in orchestrator.py / tiling.py.

Layout
  input      the image with the tile grid it WILL be cut into, the question, Run
  results    one image <= TILE_SIZE  -> a single result page (no tiling)
             several tiles           -> tabs "Overview", "Tile 1", "Tile 2", ...
                                        (more than TILE_TABS_MAX -> a tile picker)
  each tile  1. final answer (prominent) + the result picture (boxes / mask / map)
             2. 🧭 Agent planning           collapsed, opens on demand
             3. 🛠️ Tool calls & outputs     collapsed, opens on demand
Tiles, planning and tool sections are rendered lazily (only what is open), and
every picture is built once per run, so clicking around stays fast.
"""

import io
import json
import os
import hashlib
from pathlib import Path

import streamlit as st
from PIL import Image

import config
import drawing
import tiling
import field_geometry
from ui_composer import TASKS, compose_request
from orchestrator import run_tiled, resume_decision

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

st.set_page_config(page_title="RSAgent Studio", page_icon="🛰️", layout="wide")
with st.sidebar:
    ui_mode = st.radio("Interface", ["Studio", "Classic"], horizontal=True,
                       help="Classic keeps the original layout. Your analysis and tools are the same.")
if ui_mode == "Studio":
    st.markdown("""
    <style>
    .stApp { background: #f5f8fb; }
    [data-testid="stSidebar"] { background: #ecf2f5; }
    [data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1500px; }
    .rs-hero { background: linear-gradient(115deg, #102a38, #124e57 67%, #207c76);
        color: #fff; padding: 2rem 2.3rem; border-radius: 18px; margin-bottom: 1.2rem; }
    .rs-hero .eyebrow { color: #9dded2; letter-spacing: .16em; font-size: .72rem;
        font-weight: 700; text-transform: uppercase; }
    .rs-hero h1 { color: #fff; font-size: 2.45rem; margin: .25rem 0 .4rem; line-height: 1.1; }
    .rs-hero p { color: #d5eeeb; font-size: 1rem; max-width: 690px; margin: 0; }
    .rs-steps { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 1.4rem; }
    .rs-steps span { background: #e4f0ef; color: #145e59; border-radius: 30px;
        font-size: .8rem; font-weight: 600; padding: .42rem .85rem; }
    div[data-testid="stFileUploader"] { border: 1px solid #cbdcdd; border-radius: 14px;
        background: #fff; padding: .8rem; }
    div[data-testid="stButton"] button[kind="primary"] { border-radius: 9px;
        background: #116e65; border-color: #116e65; }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
<style>
[class*="st-key-answer-card"] { border-left: 5px solid #2e86de !important; }
[class*="st-key-answer-card"] p, [class*="st-key-answer-card"] li { font-size: 1.08rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

STEP_ICONS = {"ok": "✅", "error": "❌", "skipped": "⏭️"}
ORIGIN_NAMES = {"detector": "🟢 LAE detection", "segmentation": "🔵 segmentation area",
                "delineate": "Delineate field polygon",
                "unibuilding": "🟠 UniBuilding building",
                "vlm": "🟣 found by the VLM"}


# ---------------------------------------------------------------- helpers
def _cache() -> dict:
    """Pictures built for the current run (cleared when a new run starts)."""
    return st.session_state.setdefault("render_cache", {})


def _picture(key, make, fmt="JPEG"):
    """Display-size encoded picture, built once per run. Streamlit re-executes the
    script on every interaction; without this each click re-draws every image."""
    cache = _cache()
    if key not in cache:
        buf = io.BytesIO()
        img = drawing.fit(make(), config.DISPLAY_MAX_SIDE).convert("RGB")
        img.save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
        cache[key] = buf.getvalue()
    return cache[key]


def _full_png(key, make):
    """Full-size PNG bytes for a download button, built once per run."""
    cache = _cache()
    if key not in cache:
        buf = io.BytesIO()
        make().save(buf, format="PNG")
        cache[key] = buf.getvalue()
    return cache[key]


def _tile_status(result) -> str:
    """ok | partial | error -- for tab labels and the overview table."""
    errors = [s for s in result.steps if s.status == "error"]
    if not errors:
        return "ok"
    return "error" if result.answered_by in ("error", "") else "partial"


def _tile_seconds(result) -> float:
    """Time spent on this tile (a router call shared by all tiles is not counted)."""
    steps = result.steps
    if steps and "shared by all" in steps[0].detail:
        steps = steps[1:]
    return round(sum(s.seconds for s in steps if not getattr(s, "parent", "")), 1)


def _rid(run) -> str:
    """Short id of a run (its folder name) for widget keys and the picture cache."""
    return os.path.basename(run.run_dir)


def _slug(run, i) -> str:
    return ("tile_%02d" % run.tiles[i].index) if run.tiled else "image"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Tiling")
    tile_size = st.number_input(
        "Max tile side (px)", 256, 8000, config.TILE_SIZE, 100,
        help="An image whose sides are both ≤ this is processed whole. Larger images are "
             "cut into tiles no bigger than this, and the agent runs on each tile separately.")
    strategy = st.radio(
        "How tiles are cut", tiling.STRATEGIES,
        index=tiling.STRATEGIES.index(config.TILE_STRATEGY)
        if config.TILE_STRATEGY in tiling.STRATEGIES else 0,
        format_func={"balanced": "Balanced: equal tiles, no overlap",
                     "fixed": "Fixed: exact size, edge tiles overlap"}.get,
        help="No tile is ever padded. Balanced: each long side is split into equal parts "
             "(7000 px → 4 × 1750). Fixed: tiles of exactly the max size; the last row / "
             "column is shifted back inside the image, so it overlaps its neighbour.")
    st.subheader("Detection")
    score_thr = st.number_input("Score threshold", 0.0, 1.0, config.DET_SCORE_THR, 0.01)
    nms_iou = st.number_input("NMS IoU", 0.0, 1.0, config.DET_NMS_IOU, 0.05)
    st.subheader("Models")
    st.caption("Router: `%s`  \nVLM: `%s`" % (config.ROUTER_MODEL, config.VLM_MODEL))
    st.caption("LAE env python: `%s`" % (config.LAE_PYTHON or "NOT FOUND"))
    st.caption("SAM env python: `%s`" % (config.SAM_PYTHON or "NOT FOUND"))
    st.caption("UniBuilding env python: `%s`" % (config.UNIBUILD_PYTHON or "NOT FOUND"))
    st.caption("Delineate env python: `%s`" % (config.DELINEATE_PYTHON or "NOT FOUND"))
    st.caption("Models load on first use (~1 min each on CPU) and stay loaded.")

if ui_mode == "Studio":
    st.markdown("""
    <div class="rs-hero"><div class="eyebrow">Remote sensing · assisted analysis</div>
    <h1>See more in every scene.</h1>
    <p>Ask a question about an aerial or satellite image. RSAgent plans an analysis,
    combines relevant model evidence, and presents results you can inspect.</p></div>
    <div class="rs-steps"><span>01 &nbsp; Upload imagery</span><span>02 &nbsp; Ask a question</span>
    <span>03 &nbsp; Review evidence &amp; outputs</span></div>
    """, unsafe_allow_html=True)
else:
    st.title("🛰️ Remote Sensing Agent V4")
    st.caption("Large images are cut into tiles · each tile: router → LAE-DINO and/or "
               "SegEarth-OV3 and/or UniBuilding → VLM answer")
    st.caption("Agricultural field boundaries: full-scene SegEarth + Delineate → semantic verification "
               "→ VLM-selected polygons. Field requests use native Delineate tiling and merging.")

# ---------------------------------------------------------------- input
with st.expander("Explore analysis capabilities"):
    st.caption("Choose a task below to compose a request, or write your own. The agent plans "
               "which tools to call for each request; selecting a task does not force a model.")
    for name, capability in TASKS.items():
        st.markdown("**%s** — %s" % (name, capability["description"]))
    st.caption("Results adapt to the plan and may include an answer, maps, masks, field polygons, "
               "tool outputs and an execution trace. Available results depend on the run.")

if ui_mode == "Studio":
    st.subheader("Start with an image")
    st.caption("PNG, JPEG, WebP or TIFF · GeoTIFF spatial metadata is retained for supported exports")
uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "tif", "tiff", "webp"])
if uploaded is None:
    st.info("Upload an aerial / satellite image to begin.")
    st.stop()

upload_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
if st.session_state.get("upload_id") != upload_id:
    try:
        data = uploaded.getvalue()
        upload_dir = Path(config.RUNS_DIR) / "uploads" / hashlib.sha256(data).hexdigest()[:20]
        upload_dir.mkdir(parents=True, exist_ok=True)
        source_path = upload_dir / ("original" + Path(uploaded.name).suffix.lower())
        source_path.write_bytes(data)
        new_image = field_geometry.read_image(source_path)
    except Exception as exc:  # noqa: BLE001
        st.error("Could not read this image: %s" % exc)
        st.stop()
    st.session_state.update(upload_id=upload_id, upload_image=new_image,
                            upload_thumb=drawing.fit(new_image, config.PREVIEW_MAX_SIDE))
    for stale in ("run", "run_image", "run_thumb", "render_cache"):
        st.session_state.pop(stale, None)
image = st.session_state["upload_image"]
thumb = st.session_state["upload_thumb"]

if ui_mode == "Studio":
    st.caption("Choose a starting point or write a question. The planner decides which tools to use.")
st.subheader("Compose an analysis")
task = st.selectbox("Task", ["Write my own request"] + list(TASKS),
                    help="Task controls help draft a request. The planner decides which tools are needed.")
if task != "Write my own request":
    capability = TASKS[task]
    st.caption(capability["description"])
    subject_col, output_col = st.columns([2, 1])
    with subject_col:
        subject = st.text_input(capability["subject_label"], value=capability["subject_default"],
                                key="composer-subject-" + task)
    with output_col:
        output = st.selectbox("Requested output", capability["outputs"], key="composer-output-" + task)
    context = st.text_input("Condition or spatial context (optional)",
                            placeholder="e.g. near the river", key="composer-context-" + task)
    if st.button("Add to question", key="compose-request"):
        try:
            st.session_state["analysis-question"] = compose_request(task, subject, context, output)
        except ValueError as exc:
            st.warning(str(exc))

question = st.text_input("Question", key="analysis-question", placeholder="e.g. Delineate agricultural fields near buildings · "
                         "Create a land use land cover map · How many ships are docked?")
planned_tiles = tiling.make_tiles(image.size, int(tile_size), strategy)
col_img, col_side = st.columns([3, 2], gap="large")
with col_img:
    if len(planned_tiles) > 1:
        st.image(drawing.draw_tile_grid(thumb, planned_tiles, max_side=config.PREVIEW_MAX_SIDE),
                 caption="%d × %d px → %s" % (image.size[0], image.size[1],
                                              tiling.grid_summary(planned_tiles)))
    else:
        st.image(thumb, caption="%d × %d px" % image.size)
with col_side:
    if len(planned_tiles) > 1:
        st.markdown("**Proposed image grid: %s.**" % tiling.grid_summary(planned_tiles))
        st.caption("The execution plan determines the final scene policy. Agricultural field geometry "
                   "uses the full scene with native Delineate tiling and seam merging.")
        st.markdown("Each tile is analysed **independently** by the full agent (planning → "
                    "tools → VLM) at full resolution, without padding. You get one answer "
                    "per tile; tile results are not merged at this stage.")
        overlap = tiling.overlap_note(planned_tiles)
        if overlap:
            st.caption("⚠️ Overlap (fixed tiles): %s. Objects there can appear in two tiles."
                       % overlap)
    else:
        st.markdown("**No tiling needed**: both sides are ≤ %d px, so the image is analysed "
                    "as a whole." % int(tile_size))
    run_clicked = st.button("Run", type="primary", disabled=not question.strip(),
                            icon=":material/play_arrow:")

if run_clicked:
    n_tiles = len(planned_tiles)
    with st.status("Agent running… (the first run loads the models)", expanded=True) as status:
        bar = st.progress(0.0, text="Starting…")

        def notify(index, count, step):
            prefix = "Tile %d of %d · " % (index, count) if count > 1 else ""
            bar.progress(min((index - 1) / count, 1.0), text=prefix + step + "…")

        run = run_tiled(image, question, score_thr, nms_iou, int(tile_size), strategy, notify)
        n_tiles = len(run.tiles)
        bar.progress(1.0, text="Done")
        failed = sum(_tile_status(r) == "error" for r in run.results)
        status.update(label="Finished in %.0f s · %d tile%s%s" % (
            run.seconds, n_tiles, "s" if n_tiles != 1 else "",
            " · %d failed" % failed if failed else ""),
            state="error" if failed == n_tiles else "complete", expanded=False)
    st.session_state.update(run=run, run_image=image, run_thumb=thumb)
    st.session_state.pop("render_cache", None)


# ---------------------------------------------------------------- one tile's page
def render_answer_visual(run, i, tile_img):
    """The result picture next to the answer: exactly what the answer selected,
    shown the way the router chose (boxes OR mask), or the map for a plain map."""
    res, key = run.results[i], "%s-%d" % (_rid(run), i)
    plan = res.plan
    if plan.output_kind == "classification_map" or (plan.output == "tool_output" and res.output_image_path):
        if res.output_image_path and os.path.isfile(res.output_image_path):
            metadata_path = Path(res.run_dir) / "classification_map.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            def with_legend():
                raw = Image.open(res.output_image_path).convert("RGB")
                table = metadata.get("class_table",[])
                return drawing.add_legend(raw,[row["name"] for row in table],
                    [row["color"] for row in table],
                    {row["name"]:row["coverage_percent"] for row in table}) if table else raw
            map_tab, compare_tab = st.tabs(["Classification map", "Compare with original"])
            with map_tab:
                st.image(_picture(key + "-class-legend",with_legend,"PNG"),caption=res.output_caption)
            with compare_tab:
                st.image(_picture(key + "-class-comparison",
                    lambda: drawing.classification_comparison(tile_img,with_legend()),"PNG"),
                    caption="Original image and classification map with matching pixel scale")
            st.download_button("Download classification map with legend",
                _full_png(key + "-class-legend-png",with_legend),file_name="classification_map_legend.png",
                mime="image/png",key="dl-class-"+key,on_click="ignore")
            st.download_button("Download class-ID mask (PNG)",Path(res.output_image_path).read_bytes(),
                file_name="classification_map.png",mime="image/png",key="dl-class-ids-"+key,on_click="ignore")
            st.download_button("Download original/map comparison",_full_png(key+"-compare-png",
                lambda: drawing.classification_comparison(tile_img,with_legend())),
                file_name="classification_comparison.png",mime="image/png",key="dl-compare-"+key,on_click="ignore")
            if metadata_path.is_file():
                if metadata.get("class_table"):
                    st.dataframe([{"ID": row["id"], "Class": row["name"],
                                   "RGB": str(row["color"]), "Coverage %": round(row["coverage_percent"],2)}
                                  for row in metadata["class_table"]], hide_index=True)
                st.download_button("Download class legend (JSON)", metadata_path.read_bytes(),
                    file_name="classification_map.json", mime="application/json", key="dl-class-json-" + key,
                    on_click="ignore")
            raster_path = Path(res.run_dir) / "classification_ids.tif"
            if raster_path.is_file():
                st.download_button("Download classification IDs (GeoTIFF)", raster_path.read_bytes(),
                    file_name="classification_ids.tif", mime="image/tiff", key="dl-class-tif-" + key,
                    on_click="ignore")
        else:
            st.info("The requested classification map could not be produced. See the execution trace.")
    elif plan.output_kind == "mask":
        if res.final_mask is not None:
            binary = lambda: Image.fromarray((res.final_mask * 255).astype("uint8"))  # noqa: E731
            t_over, t_bin = st.tabs(["Overlay", "Binary mask"])
            t_over.image(_picture(key + "-mask-over", lambda: drawing.seg_overlay(
                tile_img, res.final_mask.astype("uint8"), [[0, 0, 0], [0, 199, 255]])))
            t_bin.image(_picture(key + "-mask-bin", binary, "PNG"))
            st.caption("🎭 Mask result · white = selected pixels · built from: "
                       + res.final_mask_note)
            st.download_button("Download mask (PNG, full size)",
                               _full_png(key + "-mask-png", binary),
                               file_name="%s_mask.png" % _slug(run, i), mime="image/png",
                               key="dl-mask-" + key, on_click="ignore", icon=":material/download:")
        else:
            st.info("A mask was requested but none was built — "
                    + (res.final_mask_note or "see the tool calls"))
            st.image(_picture(key + "-input", lambda: tile_img), caption="Input")
    elif res.vlm_objects:
        origins = [o["origin"] for o in res.vlm_objects]
        st.image(_picture(key + "-boxes", lambda: drawing.draw_vlm_objects(tile_img, res.vlm_objects)),
                 caption="%d selected · Field polygons %d · 🟢 LAE detections %d · 🔵 segmentation areas %d · "
                         "🟠 UniBuilding buildings %d · "
                         "🟣 found by VLM %d" % (len(origins), origins.count("delineate"), origins.count("detector"),
                                                 origins.count("segmentation"), origins.count("unibuilding"),
                                                 origins.count("vlm")))
    elif plan.output == "tool_output" and res.segmentation is not None:
        seg = res.segmentation
        st.image(_picture(key + "-seg-over", lambda: drawing.add_legend(
            drawing.seg_overlay(tile_img, seg.mask, seg.colors), seg.class_names, seg.colors,
            seg.coverage)), caption="🗺️ Classification map (the map is the answer)")
    else:
        st.image(_picture(key + "-input", lambda: tile_img),
                 caption="Input%s — nothing to draw for this answer"
                         % (" tile" if run.tiled else ""))


def render_plan(run, i):
    res = run.results[i]
    plan = res.plan
    classification_display = plan.output_kind == "classification_map" or (
        plan.output == "tool_output" and plan.output_kind == "boxes")
    if plan.use_delineate:
        st.markdown("**Field instances (Delineate Anything V2)**: original RGB + SegEarth LCLU; "
                    "native filtering, polygon semantic verification and VLM reasoning. "
                    "Processed as a full scene with native seam merging.")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**Object detection (LAE-DINO)**  \n" + (
        "✅ " + ", ".join(plan.detection_classes) if plan.use_detection else "— not called"))
    c2.markdown("**Segmentation (SegEarth-OV3)**  \n" + (
        "✅ " + ", ".join(c.split(",")[0] for c in plan.segmentation_classes[1:])
        if plan.use_segmentation else "— not called"))
    c3.markdown("**Buildings (UniBuilding)**  \n" + (
        "✅ buildings / houses / roofs" if plan.use_unibuilding else "— not called"))
    c4.markdown("**Decision / output**  \n" + (
        "🗺️ plain map — VLM not called" if plan.output == "tool_output"
        else "💬 VLM decides with the tool outputs") + "  \n" + (
        "shown as a **classification map**" if classification_display else
        "🎭 shown as a **mask**" if plan.output_kind == "mask" else (
            "shown as **field polygons**" if plan.use_delineate else "🔲 shown as **bounding boxes**")))
    if plan.reasoning:
        st.markdown("**Router reasoning:** " + plan.reasoning)
    if plan.rule_notes:
        st.markdown("**Rules applied**")
        for note in plan.rule_notes:
            st.caption("• Rule " + note)
    if res.steps and "shared by all" in res.steps[0].detail:
        st.caption("ℹ️ The router reads only the question (never the image), so this plan was "
                   "made once and shared by all %d tiles. Every tile still ran its own tools "
                   "and its own VLM call." % len(run.tiles))


def render_tools(run, i, tile_img):
    res, key = run.results[i], "%s-%d" % (_rid(run), i)
    st.markdown("**Step trace**")
    for s in res.steps:
        st.markdown("%s **%s** · %.1f s" % (STEP_ICONS.get(s.status, s.status), s.name, s.seconds))
        if s.status == "error":
            st.code(s.detail, language=None, wrap_lines=True)
        elif s.detail:
            st.caption(s.detail)

    seg, det, uni = res.segmentation, res.detection, res.unibuilding
    names = (["Segmentation"] if seg is not None else []) + (["Detection"] if det is not None else []) \
        + (["UniBuilding"] if uni is not None else []) \
        + (["Fields"] if res.fields is not None else []) \
        + ["VLM decision", "Input", "Raw data"]
    tabs = dict(zip(names, st.tabs(names)))

    if res.fields is not None:
        with tabs["Fields"]:
            fields = res.fields
            selected = [o for o in res.vlm_objects if o.get("origin") == "delineate"]
            for title, proposals in (("Native Delineate output after LCLU filtering", fields.raw_fields),
                                     ("Semantically eligible fields (uncertain retained)", fields.fields),
                                     ("Final VLM-selected fields", selected)):
                st.image(_picture(key + title, lambda p=proposals: drawing.draw_fields(tile_img, p)),
                         caption=title + " · %d polygons" % len(proposals))
            st.dataframe([{"field": f["id"], "area px": f["area_px"],
                           "evidence": f["agricultural_evidence"]["status"],
                           "cropland fraction": f["agricultural_evidence"]["cropland_fraction"],
                           "nonag fraction": f["agricultural_evidence"]["nonag_fraction"]}
                          for f in fields.raw_fields], hide_index=True)
            st.caption("Coordinates: %s. Orange candidates are uncertain, not confirmed agriculture. "
                       "Raw native output includes native LCLU filtering; no extra unfiltered pass."
                       % fields.metadata["coordinate_reference"])
            st.json(fields.filter_config, expanded=False)
            render_field_downloads(res, key)

    if seg is not None:
        with tabs["Segmentation"]:
            col_a, col_b = st.columns(2)
            col_a.image(_picture(key + "-seg-over", lambda: drawing.add_legend(
                drawing.seg_overlay(tile_img, seg.mask, seg.colors), seg.class_names, seg.colors,
                seg.coverage)), caption="Overlay")
            col_b.image(_picture(key + "-seg-map", lambda: drawing.add_legend(
                drawing.seg_color_map(seg.mask, seg.colors), seg.class_names, seg.colors), "PNG"),
                caption="Classification map")
            st.download_button("Download classification map (PNG, full size)",
                               _full_png(key + "-seg-png",
                                         lambda: drawing.seg_color_map(seg.mask, seg.colors)),
                               file_name="%s_classification_map.png" % _slug(run, i),
                               mime="image/png", key="dl-seg-" + key, on_click="ignore",
                               icon=":material/download:")
            areas = seg.buildings + seg.regions
            if areas:
                st.image(_picture(key + "-areas", lambda: drawing.draw_boxes(
                    tile_img, [{"bbox": a["bbox"], "text": a["id"], "color": (255, 214, 10)}
                               for a in areas])),
                    caption="Selectable areas: %d building components (b*), %d other areas (r*)"
                    % (len(seg.buildings), len(seg.regions)))

    if uni is not None:
        with tabs["UniBuilding"]:
            col_a, col_b = st.columns(2)
            col_a.image(_picture(key + "-uni-over", lambda: drawing.seg_overlay(
                tile_img, uni.mask.astype("uint8"), [[0, 0, 0], [255, 149, 0]])),
                caption="UniBuilding building pixels · %.1f%% coverage" % uni.coverage)
            binary = lambda: Image.fromarray(uni.mask.astype("uint8") * 255)
            col_b.image(_picture(key + "-uni-mask", binary, "PNG"), caption="Building mask")
            st.image(_picture(key + "-uni-boxes", lambda: drawing.draw_boxes(
                tile_img, [{"bbox": b["bbox"], "text": b["id"], "color": (255, 149, 0)}
                           for b in uni.buildings])),
                caption="%d building components (b*); touching roofs may merge" % len(uni.buildings))
            st.download_button("Download UniBuilding mask (PNG, full size)",
                               _full_png(key + "-uni-png", binary),
                               file_name="%s_unibuilding.png" % _slug(run, i), mime="image/png",
                               key="dl-uni-" + key, on_click="ignore")

    if det is not None:
        with tabs["Detection"]:
            col_a, col_b = st.columns(2)
            col_a.image(_picture(key + "-det-raw", lambda: drawing.draw_detections(tile_img, det.raw)),
                        caption="Raw (above the score threshold): %d" % len(det.raw))
            col_b.image(_picture(key + "-det-nms", lambda: drawing.draw_detections(tile_img, det.nms)),
                        caption="After NMS: %d · %s" % (len(det.nms), ", ".join(
                            "%s × %d" % kv for kv in det.counts().items()) or "none"))

    with tabs["VLM decision"]:
        st.caption(res.answered_by or "—")
        if res.tile is not None and not res.tile.is_whole_scene and res.answered_by.startswith("VLM"):
            st.markdown("**Tile context sent to the VLM with the question**")
            st.caption(tiling.vlm_context(res.tile))
        if res.vlm_objects:
            st.markdown("**Objects in the final decision: %d**" % len(res.vlm_objects))
            st.dataframe([{"label": o.get("label", ""), "origin": ORIGIN_NAMES.get(o["origin"], o["origin"]),
                           "tool id": o.get("source_id") or "—", "note": o.get("note", "")}
                          for o in res.vlm_objects], hide_index=True)
        else:
            st.caption("No objects selected.")
        if res.vlm_rejected:
            st.markdown("**Rejected tool proposals (false positives): %d**" % len(res.vlm_rejected))
            for r in res.vlm_rejected:
                st.markdown("- `%s` — %s" % (r["id"], r["reason"]))

    with tabs["Input"]:
        st.image(_picture(key + "-input", lambda: tile_img),
                 caption="%s as analysed: %d × %d px, not resized, not padded" % (
                     run.tiles[i].name if run.tiled else "Image", *tile_img.size))

    with tabs["Raw data"]:
        st.json({"tile": res.tile.as_dict() if res.tile else None, "plan": res.plan.model_dump(),
                 "run_dir": res.run_dir, "steps": [s.__dict__ for s in res.steps],
                 "detections_nms": det.nms if det else None,
                 "unibuilding": {"buildings": uni.buildings, "coverage_percent": uni.coverage,
                                  "mask_path": uni.mask_path} if uni else None,
                 "vlm_objects": res.vlm_objects, "vlm_rejected": res.vlm_rejected},
                expanded=False)


def render_tile(run, i, scene, scene_thumb):
    res, tile = run.results[i], run.tiles[i]
    key = "%s-%d" % (_rid(run), i)
    tile_img = tiling.crop(scene, tile)

    if run.tiled:
        head, locator = st.columns([5, 1])
        with head:
            st.markdown("### %s of %d" % (tile.name, tile.count))
            st.caption(tile.describe())
        locator.image(_picture(key + "-locator", lambda: drawing.draw_tile_grid(
            scene_thumb, run.tiles, highlight=tile.index, max_side=320)),
            caption="Position in the scene")
    else:
        st.caption("Whole image, %d × %d px. " % tile.size + (
            "Field workflow uses Delineate's native internal tiling and merging."
            if res.plan.use_delineate else "No external tiling needed."))

    # 1. the final answer, first and prominent ---------------------------------
    col_answer, col_visual = st.columns([2, 3], gap="large")
    with col_answer:
        with st.container(border=True, key="answer-card-%d" % i):
            st.markdown("#### 💬 Answer")
            st.markdown(res.answer.replace("\n", "  \n") if res.answer else "_No answer._")
            st.caption(res.answered_by)
        facts = []
        if res.vlm_objects:
            facts.append("%d object%s selected" % (len(res.vlm_objects),
                                                   "s" if len(res.vlm_objects) != 1 else ""))
        if res.vlm_rejected:
            facts.append("%d proposal%s rejected" % (len(res.vlm_rejected),
                                                     "s" if len(res.vlm_rejected) != 1 else ""))
        facts.append("⏱ %.0f s" % _tile_seconds(res))
        st.caption(" · ".join(facts))
        errors = [s for s in res.steps if s.status == "error"]
        if errors:
            st.warning("%d step%s failed (%s) — see **Tool calls & outputs**." % (
                len(errors), "s" if len(errors) != 1 else "", ", ".join(s.name for s in errors)))
    with col_visual:
        render_answer_visual(run, i, tile_img)
    if res.fields is not None:
        render_field_downloads(res, key + "-answer")
        if st.button("Review field proposals using cached inference", key="review-cached-" + key):
            with st.status("Reviewing saved proposals; SegEarth and Delineate inference are reused.", expanded=True):
                updated = resume_decision(res.run_dir, notify=lambda step: st.write(step))
            run.results[i] = updated
            if not run.tiled:
                run.run_dir = updated.run_dir
            st.session_state["run"] = run
            st.session_state.pop("render_cache", None)
            st.rerun()

    # 2./3. the workflow, collapsed until the user asks for it -----------------
    plan_box = st.expander("Agent planning", icon="🧭", key="plan-" + key, on_change="rerun")
    if plan_box.open:
        with plan_box:
            render_plan(run, i)
    tools_box = st.expander("Tool calls & outputs · %d steps" % len(res.steps), icon="🛠️",
                            key="tools-" + key, on_change="rerun")
    if tools_box.open:
        with tools_box:
            render_tools(run, i, tile_img)


def render_field_downloads(res, key):
    for label, filename, mime in (
        ("Download field polygons (GeoPackage)", "fields.gpkg", "application/geopackage+sqlite3"),
        ("Download field evidence (JSON)", "field_proposals.json", "application/json"),
        ("Download SegEarth class IDs (GeoTIFF)", "segearth_lclu.tif", "image/tiff"),
        ("Download selected field instances (GeoTIFF)", "fields_selected_instances.tif", "image/tiff"),
        ("Download selected field binary mask (GeoTIFF)", "fields_selected_binary.tif", "image/tiff")):
        path = Path(res.run_dir) / filename
        if path.is_file():
            st.download_button(label, path.read_bytes(), file_name=filename, mime=mime,
                               key=key + filename, on_click="ignore")


def render_overview(run, scene_thumb):
    status_icon = {"ok": "✅", "partial": "⚠️", "error": "❌"}
    statuses = [_tile_status(r) for r in run.results]
    overlap = tiling.overlap_note(run.tiles)
    col_map, col_table = st.columns([1, 1], gap="large")
    col_map.image(_picture("%s-overview" % _rid(run), lambda: drawing.draw_tile_grid(
        scene_thumb, run.tiles, max_side=config.PREVIEW_MAX_SIDE)),
        caption="%d × %d px → %s" % (run.scene_size[0], run.scene_size[1],
                                     tiling.grid_summary(run.tiles)))
    with col_table:
        st.markdown("**%s**, analysed independently. Open a tile to see its answer, its "
                    "planning and its tool calls." % tiling.grid_summary(run.tiles))
        st.caption("Tile results are not merged at this stage. Strategy: **%s**%s." % (
            run.strategy, " — overlap: " + overlap if overlap else ", no overlap"))
        st.dataframe([{"tile": t.name, "status": status_icon[s],
                       "scene window (px)": "x %d–%d · y %d–%d" % (t.x0, t.x1 - 1, t.y0, t.y1 - 1),
                       "size": "%d × %d" % t.size, "s": _tile_seconds(r)}
                      for t, r, s in zip(run.tiles, run.results, statuses)], hide_index=True)
        st.caption("Run folder: `%s` (one `tile_NN/` sub-folder per tile)" % run.run_dir)


# ---------------------------------------------------------------- results
run = st.session_state.get("run")
if run is None:
    st.stop()
scene, scene_thumb = st.session_state["run_image"], st.session_state["run_thumb"]

st.divider()
st.subheader("Results")
st.caption("❓ %s · %s · %.0f s" % (run.question, tiling.grid_summary(run.tiles), run.seconds))

if not run.tiled:
    render_tile(run, 0, scene, scene_thumb)
elif len(run.tiles) <= config.TILE_TABS_MAX:
    marks = {"ok": "", "partial": " ⚠️", "error": " ❌"}
    labels = ["🗺️ Overview"] + [t.name + marks[_tile_status(r)]
                               for t, r in zip(run.tiles, run.results)]
    tabs = st.tabs(labels, key="tiles-" + _rid(run), on_change="rerun")
    if tabs[0].open:
        with tabs[0]:
            render_overview(run, scene_thumb)
    for i, tab in enumerate(tabs[1:]):
        if tab.open:                      # only the tile on screen is rendered
            with tab:
                render_tile(run, i, scene, scene_thumb)
else:
    # too many tiles for a tab bar: an overview + a picker with previous / next
    with st.expander("Overview — tile map and status", icon="🗺️"):
        render_overview(run, scene_thumb)
    pick_key = "tile-pick-" + _rid(run)
    n = len(run.tiles)

    def _step(delta):
        st.session_state[pick_key] = (st.session_state.get(pick_key, 0) + delta) % n

    c_prev, c_pick, c_next = st.columns([1, 4, 1], vertical_alignment="bottom")
    c_prev.button("Previous", on_click=_step, args=(-1,), icon=":material/arrow_back:",
                  width="stretch")
    picked = c_pick.selectbox("Tile", list(range(n)), key=pick_key,
                              format_func=lambda k: "%s — %s" % (
                                  run.tiles[k].name, run.tiles[k].describe()))
    c_next.button("Next", on_click=_step, args=(1,), icon=":material/arrow_forward:",
                  width="stretch")
    render_tile(run, picked, scene, scene_thumb)
