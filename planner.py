"""Interpret subjects, operation and scope first; compile tools and presentation separately.

The LLM returns RequestIntent, never executable tool flags. compile_intent is
the single policy boundary used by normal, shared, supplied and fallback plans.
A map is a presentation. Whole-image classification, general-scene LULC and agricultural field evidence
authorize the complete LULC vocabulary. A focused request must name its classes.
"""

import re
from typing import Dict, List

import config
from llm_client import chat_json
from schemas import Plan, RequestIntent, RequestTarget
from tools.segmentation import is_building_class


class PlanningError(ValueError):
    pass


BACKGROUND_CLASSES = {"background", "unclassified", "background/unclassified"}


ROUTER_SYSTEM_PROMPT = """\
Interpret the user's remote-sensing request. Return a RequestIntent, not a tool
plan and not an answer. Separate these independent dimensions:
- subjects and contextual dependencies (targets);
- the operation to perform;
- scope of semantic classes;
- presentation (boxes, binary mask, or classification_map);
- conditions requiring reasoning.

SUBJECT CONTRACT
Each target has kind land_cover, object, building, or agricultural_field; a
category as worded by the user; classes; and role target/context.
A classification request with a named subject classifies THAT SUBJECT.
Whole-image classification without a narrower subject uses general_scene and the
configured scene taxonomy. "Classify image", "classify this scene" and a general
classification map mean scene land-cover classification. Do not return empty
targets for these; use a land_cover subject with category scene and no classes.
For explicitly named classes, preserve those classes. For an open category,
choose a compact meaningful set of subclasses OF THAT CATEGORY using your
language understanding. Never copy unrelated default LULC categories.
Use English lowercase classes; comma-separated synonyms form one class.
Separate class identity from colour, damage, size or spatial constraints.
Put the requested classification dimension in classification_basis and requested
result categories in output_classes. Attribute categories do not replace the
subject's tool classes: the VLM classifies measured subject proposals afterwards.
Do not invent botanical species or other fine labels unsupported by the imagery.

CAPABILITIES
- land_cover: continuous semantic cover/areas: roads, water, forest/tree cover,
  grass, farmland, bare soil, sand and parking areas. Roads are land cover even
  when called features, objects, networks or infrastructure. NEVER route these
  to an object detector. Pixel coverage uses this kind too.
- object: discrete non-building instances and their types/attributes; vehicles,
  ships, aircraft and sports facilities belong here even when a map is requested.
- building: building/house/roof instances, counts, attributes and type classification.
- agricultural_field: individual agricultural field polygons/boundaries, not
  agricultural pixel coverage. Include only contextual subjects actually needed.
The compiler maps these kinds to SegEarth, LAE-DINO, UniBuilding and Delineate.
UniBuilding also runs first as a mandatory building exclusion dependency for
semantic cover. Do not add it as a user target unless buildings are requested.
Do not treat any request for 'classification', 'map' or 'mask' as a request for
land cover or for extra tools.

SCOPE AND DECISION
land_cover_scope=general_scene for unrestricted whole-image classification or a
general land-use/land-cover map, including a classification map with no named
subject. The compiler supplies configured classes; leave output_classes empty.
For a particular category or named subset use
requested_classes and supply that category's classes. No semantic subject -> none.
Plain semantic class maps need no reasoning. Attributes, spatial relationships,
comparison, selection, percentages and instance interpretation need reasoning.
Object classification and building attributes/types need the VLM after their
instance tools. Simply combining native class identities (building and road,
for example) requires no reasoning. Preserve conditions in constraints.
Agricultural instances always need semantic evidence plus geometric proposals
and VLM reasoning. Agricultural coverage needs semantic evidence only.

PRESENTATION
classification_map means a dense categorical pixel mask of the requested result
classes, with background/unclassified elsewhere. Use semantic pixels, UniBuilding
pixels or original field polygons; detector subjects need a subsequent pixel
extraction step. Bounding boxes and image overlays are not classification maps.
Never add unrelated classes. Background/unclassified is reserved class ID 0,
not a foreground output class or an inference prompt. Choose mask for a binary mask. Respect
negation. Rendering words must not change subjects, classes or reasoning needs.
Output no tool booleans, default vocabulary, or per-question special cases.
The user's language may differ from the English class labels.
"""


DETECTION_OBJECTS: Dict[str, str] = {
    "small vehicle": "small vehicle", "car": "small vehicle", "vehicle": "small vehicle",
    "large vehicle": "large vehicle", "truck": "large vehicle", "bus": "large vehicle",
    "lorry": "large vehicle", "ship": "ship", "boat": "ship", "vessel": "ship",
    "plane": "plane", "airplane": "plane", "aircraft": "plane", "helicopter": "helicopter",
    "storage tank": "storage tank", "tank": "storage tank", "swimming pool": "swimming pool",
    "pool": "swimming pool", "tennis court": "tennis court", "basketball court": "basketball court",
    "soccer": "soccer ball field", "football field": "soccer ball field",
    "football pitch": "soccer ball field", "sports facilities": "tennis court",
    "sports facility": "tennis court",
    "baseball": "baseball field", "ground track field": "ground track field",
    "bridge": "bridge", "roundabout": "roundabout", "windmill": "windmill",
    "wind turbine": "windmill", "chimney": "chimney", "container": "container",
    "stadium": "stadium", "sport field": "tennis court", "sports field": "tennis court",
    "court": "tennis court",
    "خودرو": "small vehicle", "ماشین": "small vehicle", "کامیون": "large vehicle",
    "اتوبوس": "large vehicle", "کشتی": "ship", "قایق": "ship", "هواپیما": "plane",
    "بالگرد": "helicopter", "هلیکوپتر": "helicopter", "مخزن": "storage tank",
    "استخر": "swimming pool", "زمین تنیس": "tennis court", "زمین ورزشی": "tennis court",
}
# a vague "vehicle" can be either size
_SPORT = ["basketball court", "soccer ball field", "baseball field", "ground track field"]
_ALSO = {"vehicle": ["large vehicle"], "خودرو": ["large vehicle"], "court": ["basketball court"],
         "sport field": _SPORT, "sports field": _SPORT, "زمین ورزشی": _SPORT}

_ALSO.update({"sports facilities": _SPORT, "sports facility": _SPORT})

_BUILDING_Q = re.compile(r"\b(building|buildings|house|houses|home|homes|roof|roofs|rooftops?)\b"
                         r"|ساختمان|خانه|منزل|بام|سقف", re.IGNORECASE)


def _keyword_re(word: str):
    if re.search(r"[؀-ۿ]", word):          # Persian: plain substring
        return re.compile(re.escape(word))
    return re.compile(r"\b%ss?\b" % re.escape(word), re.IGNORECASE)


_OBJECT_RES = [(k, _keyword_re(k), v)
               for k, v in sorted(DETECTION_OBJECTS.items(), key=lambda kv: -len(kv[0]))]


# an object word used as a PLACE modifier is not an object request:
# "car parking", "bus station", "boat dock", "پارکینگ خودرو" -> no detection
_PLACE_COMPOUNDS = re.compile(
    r"\b(car|cars|vehicle|vehicles|truck|trucks|bus|buses|boat|boats|ship|ships|plane|planes"
    r"|aircraft)[\s-]*(park|parks|parking|parkings|lot|lots|stop|stops|station|stations"
    r"|terminal|terminals|depot|depots|yard|yards|ramp|ramps|storage|dock|docks|hangar|hangars"
    r"|wash|washes|garage|garages|dealership|dealerships)\b"
    r"|(پارکینگ|توقفگاه|ایستگاه|پایانه|تعمیرگاه|اسکله|آشیانه)\S*\s*(\u200c?\s*(های|ها)\s*)?(ی\s*)?"
    r"(خودرو|ماشین|اتوبوس|کامیون|قایق|کشتی|هواپیما)\S*",
    re.IGNORECASE,
)


def objects_in_question(question: str) -> List[str]:
    """Detector classes the question asks about (longest keyword first).
    Place names built from an object word ("car parking") are ignored."""
    found, text = [], _PLACE_COMPOUNDS.sub(" ", question)
    for word, pattern, cls in _OBJECT_RES:
        if pattern.search(text):
            text = pattern.sub(" ", text)   # "large vehicle" must not also match "vehicle"
            for c in [cls] + _ALSO.get(word, []):
                if c not in found:
                    found.append(c)
    return found


def detection_class_of(seg_class: str):
    """The detector class for a (wrongly) segmentation-routed object, else None."""
    first = seg_class.split(",")[0].strip().lower()
    for _, pattern, cls in _OBJECT_RES:
        if pattern.fullmatch(first) or first == cls:
            return cls
    return None


def semantic_class_of(name):
    """Recognized continuous-cover capabilities, independent of router kinds.

    Reuse the configured semantic vocabulary and V3's continuous-cover subjects.
    This is a tool capability guard, not a request-to-class template: the actual
    requested classes are preserved, including arbitrary open vocabulary.
    """
    entries = [c for c in config.SEMANTIC_CLASSES.values()
               if not is_building_class(c) and not detection_class_of(c)]
    entries += ["road,street,pavement,highway,freeway,motorway,road network",
                "grass", "vegetation", "sand", "parking lot,car park,parking area",
                "wetland,marsh", "snow", "ice", "beach", "rock"]
    for part in name.split(","):
        for entry in entries:
            if any(_keyword_re(alias.strip()).fullmatch(part.strip())
                   for alias in entry.split(",")):
                return entry
    return None


_PRODUCT = re.compile(      # the user wants a land-cover / segmentation map as the result
    r"\b(land[\s-]?use|land[\s-]?cover|lulc|segment|segments|segmentation|segmented)\b"
    r"|(?<!on the )(?<!in the )\bmaps?\b"
    r"|نقشه|قطعه\s*‌?\s*بندی|سگمنت",
    re.IGNORECASE,
)
_NEEDS_REASONING = re.compile(   # anything to find, select, judge or answer
    # questions / analysis
    r"\b(how|what|where|which|why|when|who|whose|count|number\s+of|percentage|percent"
    r"|is\s+there|are\s+there|does|describe|explain|analy[sz]e|interpret|compare|tell"
    r"|detect|locate|find|identify|list|select|choose|suitable|good|best|worst|should"
    # superlatives / selection
    r"|largest|biggest|smallest|highest|lowest|tallest|longest|shortest|nearest|closest"
    r"|widest|most|least|first|second|third|last|except|excluding"
    # attributes: colour, size, state, orientation
    r"|red|green|blue|yellow|white|black|gr[ae]y|orange|brown|purple|pink|silver|dark|bright"
    r"|colou?red|colou?r|big|large|small|tiny|huge|damaged|destroyed|flooded|burn(ed|t)"
    r"|new|old|empty|occupied|parked|moving|horizontal\w*|vertical\w*|oriented|diagonal\w*"
    # relations
    r"|near|next\s+to|beside|around|inside|outside|between|along|adjacent|close\s+to"
    r"|north\w*|south\w*|east\w*|west\w*)\b"
    r"|چند|چقدر|چه\s*تعداد|کجا|کدام|چرا|آیا|درصد|توضیح|توصیف|تحلیل|مقایسه|شناسایی|پیدا"
    r"|بزرگترین|کوچکترین|بلندترین|نزدیکترین|زرد|قرمز|سفید|سیاه|آبی|سبز|نارنجی|رنگ"
    r"|آسیب|تخریب|پارک|افقی|عمودی|نزدیک|کنار|اطراف|داخل|بین|شمال|جنوب|شرق|غرب|مناسب",
    re.IGNORECASE,
)



_AGRICULTURE = re.compile(
    r"\b(agricultur\w*|crop(?:land|s)?|farm(?:land|s)?|cultivated|arable|orchards?|vineyards?|pastures?)\b"
    r"|کشاورز|مزارع|مزرعه|زراعی|باغات|تاکستان", re.I)
_COVERAGE = re.compile(r"\b(percent\w*|coverage|proportion|fraction|how much|how many hectares)\b|درصد|مساحت", re.I)
_BOUNDARIES = re.compile(r"\b(boundar\w*|delineat\w*|polygons?|parcels?|instances?|outlines?)\b|مرز|قطعات|قطعه", re.I)
_FIELD_GEOMETRY = re.compile(r"\b(fields?|boundar\w*|delineat\w*|polygons?|parcels?|instances?|outlines?)\b|مرز|محدوده|قطعات|قطعه", re.I)
_CLASSIFY = re.compile(r"\b(classif\w*|categor\w*|types?|species|multiclass)\b|طبقه|دسته", re.I)


def _clean(names):
    out = []
    for name in names:
        entry = ",".join(p.strip().lower() for p in name.split(",") if p.strip())
        if entry and entry != "background" and entry not in out:
            out.append(entry)
    return out


def _has_condition(question, classes=()):
    # Category words are not conditions. Remove requested vocabulary before
    # checking legacy questions, so a class name cannot also be a colour filter.
    text = question
    for entry in sorted(classes, key=len, reverse=True):
        for word in entry.split(","):
            if word.strip():
                text = _keyword_re(word.strip()).sub(" ", text)
    return bool(_NEEDS_REASONING.search(text))


def is_plain_map_request(question):
    return bool(_PRODUCT.search(question)) and not _has_condition(question) and not objects_in_question(question)


def field_request(question):
    """Conservative offline hint only. Typed intent determines actual execution."""
    if not _AGRICULTURE.search(question):
        return False
    if _COVERAGE.search(question) and not _BOUNDARIES.search(question):
        return False
    return bool(_FIELD_GEOMETRY.search(question) or re.search(
        r"\b(find|show|identify|extract|locate|select|containing|near)\b|پیدا|شناسایی|استخراج", question, re.I))


def _general_lulc_authorized(question):
    # Scope authorization depends on the subject, never on rendering vocabulary.
    return bool(re.search(r"\b(land[\s-]?use|land[\s-]?cover|lulc)\b|کاربری\s*اراضی|پوشش\s*زمین", question, re.I)
                or _unrestricted_classification(question))


def _unrestricted_classification(question):
    """Whole-scene grammar, not a dictionary of per-subject exceptions.

    Match only a classification operation with an omitted or image/scene/map
    subject. Additional category words fail the match: vegetation, vehicle type,
    minerals, constraints and attributes cannot silently acquire default classes.
    Multilingual semantic routing supplies general_scene for equivalent intents.
    """
    text = question.strip().lower().replace("\u200c"," ")
    if re.search(r"\b(?:not|don't|never)\s+classif",text):
        return False
    if not re.search(r"\bclassif(?:y|ication|ying|ied)\b|طبقه\s*بندی",text):
        return False
    words=set(re.findall(r"[^\W_]+",text,re.UNICODE))
    # Grammar/presentation words only. Any additional content noun/adjective
    # belongs to a focused subject or condition and prevents implicit defaults.
    grammar=set(("a an the this that these those my our your of for on in to into use using with from and as "
        "please can could would will you i we want need like kindly create make generate produce "
        "give show provide me us do perform carry out get default predefined configured standard usual "
        "classification classify classifying classified class classes map maps mask masks binary "
        "image images scene scenes input uploaded satellite remote sensing picture pictures photo photos "
        "این آن تصویر تصاویر عکس نقشه طبقه بندی را کن کنید بساز بسازید لطفا لطفاً یک تهیه ایجاد").split())
    return bool(words) and words <= grammar


def _default_entry_for(name):
    return next((entry for entry in config.SEMANTIC_CLASSES.values()
                 if name in [p.strip() for p in entry.split(",")]), name)


def _known_cover_in_question(question):
    return [entry for entry in config.SEMANTIC_CLASSES.values()
            if not is_building_class(entry) and any(
                _keyword_re(word.strip()).search(question) for word in entry.split(","))]


def _legacy_intent(plan, question):
    """Compatibility for saved/API plans; never infer defaults from empty lists.

    Live routing uses RequestIntent directly. This bridge retains existing
    recognized building/object guards without any per-topic class dictionaries.
    """
    generic = _general_lulc_authorized(question)
    field = field_request(question)
    coverage = bool(_COVERAGE.search(question)) and not _BOUNDARIES.search(question)
    det = _clean(plan.detection_classes) if plan.use_detection and not generic else []
    seg = _clean(plan.segmentation_classes) if plan.use_segmentation else []
    # A stale full default vocabulary cannot authorize its own scope.
    full_default = set(seg) == set(config.DEFAULT_LULC_CLASSES[1:])
    if full_default and not generic and not field:
        seg = []
    if field:
        # Full semantic evidence will be supplied by the compiler, not interpreted
        # as explicit sports/building instance targets.
        seg = []
        det = []
    for cls in objects_in_question(question):
        if cls not in det:
            det.append(cls)
    if not seg and not generic and not field:
        seg = _known_cover_in_question(question)
        if coverage and _AGRICULTURE.search(question) and not seg:
            seg = [_default_entry_for("cropland")]
    targets = []
    if field:
        targets.append(RequestTarget(kind="agricultural_field", category="agricultural field"))
    if det:
        targets.append(RequestTarget(kind="object", classes=det, role="context" if field else "target"))
    if seg or generic:
        targets.append(RequestTarget(kind="land_cover", classes=seg,
                                     category="land cover" if generic else "requested land cover"))
    if (_BUILDING_Q.search(question) or (plan.use_unibuilding and not generic)):
        targets.append(RequestTarget(kind="building", category="building", role="context" if field else "target"))
    names = [c for t in targets for c in t.classes]
    reasoning = _has_condition(question, names)
    operation = ("delineate" if field else "coverage" if coverage else
                 "classify" if (_PRODUCT.search(question) or _CLASSIFY.search(question)) else "analyze")
    return RequestIntent(operation=operation, targets=targets,
        land_cover_scope="general_scene" if generic else "requested_classes" if seg else "none",
        presentation=plan.output_kind, requires_reasoning=reasoning,
        reasoning=plan.reasoning)


def compile_intent(request, question):
    """One idempotent compiler. Scope, subject, decision and rendering stay separate."""
    request = RequestIntent.model_validate(request)
    request = request.model_copy(update={"output_classes": [name for name in request.output_classes
        if name.strip().lower() not in BACKGROUND_CLASSES]})
    if _unrestricted_classification(question):
        request = request.model_copy(update={"operation": "classify", "land_cover_scope": "general_scene",
            "targets": [RequestTarget(kind="land_cover", category="scene")],
            "output_classes": [], "constraints": [], "requires_reasoning": False,
            "classification_basis": "land cover",
            "presentation": "classification_map" if request.presentation == "boxes" else request.presentation})
    generic = request.land_cover_scope == "general_scene"
    if generic and not _general_lulc_authorized(question):
        raise PlanningError("Only an explicit general scene land-use/land-cover request authorizes default classes.")
    semantic_subjects = [t for t in request.targets if t.kind == "land_cover" and t.role == "target"]
    if generic and semantic_subjects and not any(
            not t.classes or re.sub(r"[\s-]", "", t.category.lower()) in
            ("", "landcover", "landuse", "lulc", "landuselandcover", "scene", "image", "wholeimage", "wholescene") for t in semantic_subjects):
        raise PlanningError("General scene scope conflicts with a targeted semantic category.")
    notes, det, seg = [], [], []
    use_uni = False
    field = any(t.kind == "agricultural_field" for t in request.targets)
    if request.operation == "coverage":
        field = False
    # Persist the repaired subject kinds, not just repaired tool booleans. The
    # executor/VLM/renderer must see the same target/context contract as the tools.
    resolved_targets = []
    for target in request.targets:
        if target.kind == "agricultural_field" and request.operation == "coverage":
            resolved_targets.append(target.model_copy(update={"kind": "land_cover",
                "classes": [_default_entry_for("cropland")]}))
            continue
        groups = {}
        # An authorized general LULC subject may mention built-up cover without
        # requesting building instances. Field evidence keeps its complete scheme.
        # Explicit building subjects still require UniBuilding.
        semantic_evidence = (target.kind == "land_cover" and (
            (generic and not _BUILDING_Q.search(question)) or
            (field and set(_clean(target.classes)) == set(config.SEMANTIC_CLASSES.values()))))
        for cls in _clean(target.classes):
            kind, value = target.kind, cls
            if kind != "agricultural_field" and not semantic_evidence:
                if is_building_class(cls):
                    kind = "building"
                elif detection_class_of(cls):
                    kind = "object"
                    if target.kind != "object":
                        value = cls.split(",")[0] if request.operation == "classify" else detection_class_of(cls)
                elif semantic_class_of(cls) or (kind == "object" and semantic_class_of(target.category)):
                    kind = "land_cover"
            if kind != target.kind:
                note = "%s: %s -> %s capability" % (cls, target.kind, kind)
                if note not in notes:
                    notes.append(note)
            groups.setdefault(kind, []).append(value)
        if groups:
            resolved_targets += [target.model_copy(update={"kind": kind, "classes": values})
                                 for kind, values in groups.items()]
        else:
            resolved_targets.append(target)
    request = request.model_copy(update={"targets": resolved_targets})
    if not generic:
        request = request.model_copy(update={"land_cover_scope": "requested_classes"
            if any(t.kind == "land_cover" for t in resolved_targets) else "none"})
    for target in request.targets:
        classes = _clean(target.classes)
        if target.kind == "building":
            use_uni = True
            continue
        if target.kind == "agricultural_field":
            if request.operation == "coverage":
                seg.append(_default_entry_for("cropland"))
            continue
        if target.kind == "object":
            if not classes:
                raise PlanningError("An object subject must supply detector classes.")
            for cls in classes:
                if is_building_class(cls):
                    use_uni = True
                    notes.append("Building instance class -> UniBuilding")
                elif cls not in det:
                    det.append(cls)
            continue
        if not classes and not generic and not field:
            raise PlanningError("A targeted semantic request has no classes; refusing to use the full LULC defaults.")
        if (set(classes) == set(config.DEFAULT_LULC_CLASSES[1:]) and not generic and not field):
            raise PlanningError("The full LULC vocabulary is outside this targeted request's scope.")
        for cls in classes:
            if cls not in seg:
                seg.append(cls)

    # Invariant guards for recognized explicit subjects, independent of 'map'.
    if _BUILDING_Q.search(question):
        use_uni = True
    if not det and not generic:
        det = objects_in_question(question)
    if field:
        seg = list(config.SEMANTIC_CLASSES.values())
        notes.append("Agricultural instances require full semantic evidence + Delineate + VLM on the full scene")
    elif generic:
        # General-scene scope uses exactly the configured default product.
        # Named categories outside it use requested_classes instead.
        seg = list(config.DEFAULT_LULC_CLASSES[1:])
        # The predefined scene map includes building pixels supplied by UniBuilding.
        use_uni = True
        seg = [c for c in seg if not is_building_class(c)]
        if not any(t.kind == "building" and t.role == "target" for t in request.targets):
            request = request.model_copy(update={"targets": request.targets + [
                RequestTarget(kind="building",category="buildings",classes=["buildings"])]})
        request = request.model_copy(update={"output_classes": [c.split(",")[0] for c in config.DEFAULT_LULC_CLASSES[1:]]})
        notes.append("Explicit general scene LULC scope authorizes the default scheme")
    seg, det = _clean(seg), _clean(det)
    if len(seg) > config.SEG_MAX_CLASSES:
        raise PlanningError("Requested semantic classes exceed SEG_MAX_CLASSES; refusing silent truncation.")
    if len(det) > config.DET_MAX_CLASSES:
        raise PlanningError("Requested object classes exceed DET_MAX_CLASSES; refusing silent truncation.")
    if request.operation == "classify" and not (seg or det or use_uni or field):
        raise PlanningError("Classification requires a defined subject and class scope; no default map is implied.")
    semantic_names = {word for entry in seg for word in [entry] + entry.split(",")}
    native_names = semantic_names | ({"building", "buildings", "house", "houses", "home", "homes",
                                      "roof", "roofs", "rooftop", "rooftops"} if use_uni else set())
    direct_native = ((bool(seg) or use_uni) and not det and not field
                       and request.operation == "classify"
                       and not request.requires_reasoning and not request.constraints
                       and not any(t.role == "context" for t in request.targets)
                       and all(name.lower() in native_names for name in request.output_classes)
                       # Preserve V3's VLM building default for legacy plans and
                       # for unspecified building attribute classifications.
                       and (not use_uni or bool(request.output_classes)))
    protect_buildings = bool(seg)
    if protect_buildings:
        notes.append("UniBuilding first; semantic building exclusion on original pixels before decisions/field filtering")
    notes.append("Subject determines tools; operation/conditions determine decision; presentation determines rendering")
    return Plan(request=request, use_detection=bool(det), detection_classes=det,
        use_segmentation=bool(seg), segmentation_classes=["background"] + seg if seg else [],
        use_unibuilding=use_uni or protect_buildings, use_delineate=field,
        protect_semantic_buildings=protect_buildings,
        output="tool_output" if direct_native else "vlm_answer",
        output_kind=request.presentation, reasoning=request.reasoning, rule_notes=notes)


def apply_rules(plan, question):
    """Normalize live intents and all saved/caller-supplied plans in one place."""
    if isinstance(plan, RequestIntent):
        return compile_intent(plan, question)
    request = plan.request if plan.request is not None else _legacy_intent(plan, question)
    compiled = compile_intent(request, question)
    compiled.rule_notes = list(dict.fromkeys(plan.rule_notes + compiled.rule_notes))
    return compiled


def enforce_building_routing(plan, question):
    """Compatibility entry point; there is no separate execution policy anymore."""
    return apply_rules(plan, question)


def make_plan(question):
    """Interpret semantics, validate, then compile. Repair invalid scope once."""
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    for attempt in range(2):
        intent, raw = chat_json(config.ROUTER_MODEL, messages, RequestIntent,
                               json_mode=True, temperature=0, retries=2)
        try:
            return apply_rules(intent, question)
        except PlanningError as exc:
            if attempt:
                raise
            messages += [{"role": "assistant", "content": raw or intent.model_dump_json()},
                         {"role": "user", "content": "Repair the request interpretation. " + str(exc)}]
    raise PlanningError("Could not resolve a valid request.")
