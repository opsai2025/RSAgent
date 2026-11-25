# RSAgent

RSAgent is a research-oriented assistant for interpreting aerial and satellite
imagery. A natural-language question is converted into a plan that combines
remote-sensing models and, when needed, a vision-language model (VLM). The
Streamlit interface presents the answer, model outputs, processing trace and
downloadable spatial artifacts.

The agent is an integration of independently developed models. Model code,
weights, Python environments, input imagery and API credentials are **not**
included in this repository; they must be obtained and configured separately.

## What it can do

| Capability | Model or component | Example question |
| --- | --- | --- |
| Open-vocabulary object detection | LAE-DINO | “Find vehicles in this image.” |
| Semantic land-cover segmentation | SegEarth-OV-3 / SAM 3 | “Map roads and water.” |
| Building extraction | UniBuild | “Show building footprints.” |
| Agricultural field proposals | Delineate Anything v2, with semantic evidence from SegEarth and UniBuild | “Delineate agricultural fields and show uncertain candidates.” |
| Interpretation and selection | Configured router and VLM | “Find fields near buildings.” |

The planner selects tools based on the question; not every run uses every model.
The general scene map uses buildings, grass, tree, meadow, road, water and
bareland, with background/unclassified assigned ID 0. Buildings are supplied by
UniBuild. Focused requests can use a different class scope.

## Method and interpretation

The application routes a question to a structured plan, invokes models in
separate worker processes, and combines the returned detections, masks and
geometry. For field requests, it uses UniBuild to identify building pixels,
SegEarth for land-cover evidence and Delineate for candidate field geometry. It
measures class composition inside candidate polygons; a VLM reviews eligible
proposals and their contextual evidence. These checks are heuristics: semantic
fractions are **not calibrated confidence scores**, and uncertain fields remain
uncertain. A VLM answer is an interpretation, not ground truth.

Field requests run through Delineate's full-scene native tiling and merging;
other large images can be split into independent tiles. Tile answers are not
automatically merged. The application's visualizations and class labels should
be checked against the source image and, for research, independent reference
data. Accuracy varies with geography, spatial resolution, season, imaging
conditions, vocabulary, checkpoints and model settings. The supplied smoke tests
check behavior; they are not an accuracy benchmark.

## Requirements

- Windows or Linux, including WSL 2, with Git and Conda/Miniconda. Use native
  Linux interpreters and Linux paths when running in WSL; Windows `.exe`
  interpreters and Windows virtual environments cannot be reused as Linux
  environments. GPU drivers, CUDA and PyTorch must be compatible if using a GPU.
- An OpenAI-compatible endpoint with text and image support, a credential, and
  model names available at that endpoint. A run may transmit the image and
  contextual information to that service; check its data handling policy before
  using sensitive imagery. A local compatible endpoint can also be configured.
- Four external model repositories and their checkpoints, described below.
  Only the models used by a particular request need to be running, but a planned
  operation will fail if its required worker or checkpoint is unavailable.
- Several separate Python environments. LAE-DINO's older PyTorch/MMDetection
  stack conflicts with the modern SegEarth/SAM and UniBuild stacks; Delineate
  also requires GDAL. Do not assume one `pip install` creates all four.
- Significant disk space and memory for the models and imagery. GPU acceleration
  depends on each model and its installed PyTorch build; CPU inference can be
  slow and may not be supported by every upstream configuration.

## External models and weights

Follow each upstream installation guide and license. Use a trusted checkpoint
from the model author, and review the upstream model's terms independently.
The repository versions are not all locked; for reproducible work, record the
exact source commit, checkpoint checksum, installed package versions, hardware,
endpoint model version and configuration for each experiment.

| Component | Source and weights | Local inputs required by this integration |
| --- | --- | --- |
| LAE-DINO | [source](https://github.com/jaychempan/LAE-DINO), [checkpoint](https://huggingface.co/ML4Sustain/LAE-DINO) | Repository, LAE-1M Swin-T checkpoint, and local [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) files. Follow upstream's Python 3.8/MMDetection setup. |
| SegEarth-OV-3 | [source](https://github.com/earth-insights/SegEarth-OV-3), [SAM 3 weights](https://huggingface.co/facebook/sam3) | Repository, its SAM 3 dependencies and `sam3.pt`. Hugging Face access may require approval and a token. |
| UniBuild | [source](https://github.com/zhu-xlab/UniBuild), [checkpoint](https://drive.google.com/drive/folders/1YZ-bbXoZ1rOKQayRL79bLdhIE_CcU3K9) | Repository and `unibuild_dinov3_base_hrdpt.pth`; tested local source identified itself as commit `eeb56ff95514ef61e4c4e4098d55e3a69c764f9e`. |
| Delineate Anything v2 | [source](https://github.com/Lavreniuk/Delineate-Anything), [v2 checkpoint](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnythingv2.pt) | Repository and `DelineateAnythingv2.pt`; inspected local source identified itself as commit `653e261c856356b9614cd15e964cf437bf8b846e`. The upstream project is AGPL-3.0. |

The local copy of LAE-DINO and SegEarth used during development was downloaded
without Git metadata; its exact revisions cannot be established. Upstream
installation instructions and dependencies can change, so a new installation
must be validated against the worker tests below.

## Installation

### 1. Clone the agent

```bash
git clone https://github.com/opsai2025/RSAgent.git
cd RSAgent
```

On Windows, use PowerShell or a Conda prompt. On WSL, run these commands in the
Linux terminal. Keeping the source and environments on the WSL Linux filesystem
is generally convenient; use absolute Linux paths in `.env`. The commands below
illustrate the WSL/Linux form. They are also applicable in Windows Conda after
using Windows paths in `.env`.

The defaults look for sibling directories such as `../LAE-DINO-main`,
`../SegEarth-OV-3/SegEarth-OV-3-main`, `../UniBuild/unibuild_repo`, and
`../Delineate_AnythingV2/upstream`. These are external repositories, not files
downloaded automatically by the agent. You can place them elsewhere and set
the individual paths below. For example, from a separate model directory:

```bash
mkdir -p ~/rsagent-models
cd ~/rsagent-models
git clone https://github.com/jaychempan/LAE-DINO.git
git clone https://github.com/earth-insights/SegEarth-OV-3.git
git clone https://github.com/zhu-xlab/UniBuild.git
git clone https://github.com/Lavreniuk/Delineate-Anything.git
```

Set the respective repository paths in `.env` and separately download each
checkpoint. Return to the agent directory before running the commands below.

### 2. Create model environments

Install the LAE-DINO, SegEarth-OV-3 and UniBuild repositories and weights from
their upstream documentation. Create a separate Conda environment for each.
For example, the upstream LAE instructions use Python 3.8 and a compatible
PyTorch/MMDetection build. Place the downloaded BERT files in the directory
configured by `LAE_BERT_DIR`.

The agent uses these repository-specific requirement files in addition to
upstream setup:

```bash
# In the modern Python environment that runs Streamlit and the SegEarth worker:
python -m pip install -r requirements_agent.txt
python -m pip install -r requirements_segearth.txt

# In the separate UniBuild worker environment:
python -m pip install -r requirements_unibuilding.txt

# For field delineation, create a separate environment with GDAL first:
conda create -n rsagent-delineate -c conda-forge python=3.12 gdal
conda activate rsagent-delineate
python -m pip install -r requirements_delineate.txt
```

Select a PyTorch build supported by your OS and GPU before installing worker
dependencies, following the relevant upstream guide. GDAL's native library and
Python bindings must match; a conda-forge installation avoids a typical local
GDAL source build. This is a starting point, not a lockfile: upstream projects
and platform-specific binary dependencies may require additional setup. See
`requirements_*.txt` for the integration's Python dependencies.

### 3. Configure paths and endpoint

Copy `.env.example` to `.env`, then edit it. In WSL/Linux:

```bash
cp .env.example .env
```

In PowerShell: `Copy-Item .env.example .env`. Use paths appropriate to the OS
running Streamlit. If Windows and WSL share the same checkout, they also share
its `.env`; update the paths when switching OS, or use separate checkouts and
environments. The key settings are:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Credential and optional OpenAI-compatible API base URL. |
| `ROUTER_MODEL`, `VLM_MODEL` | Endpoint model IDs; the VLM must support images. |
| `LAE_PYTHON`, `SAM_PYTHON`, `UNIBUILD_PYTHON`, `DELINEATE_PYTHON` | Absolute interpreter paths for the four workers. Set explicitly when automatic Conda discovery does not find them. |
| `LAE_DINO_ROOT`, `LAE_CONFIG_FILE`, `LAE_CHECKPOINT`, `LAE_BERT_DIR` | LAE source, model configuration, weights and BERT directory. |
| `SEGEARTH_ROOT`, `SAM3_CHECKPOINT` | SegEarth repository and local SAM 3 checkpoint. |
| `UNIBUILD_ROOT`, `UNIBUILD_REPO`, `UNIBUILD_CHECKPOINT` | UniBuild workspace, source and checkpoint. |
| `DELINEATE_ROOT`, `DELINEATE_UPSTREAM`, `DELINEATE_CHECKPOINT` | Delineate workspace, source and checkpoint. |
| `RGB_BANDS` | One-based image bands used as RGB for TIFF input; default `1,2,3`. |

For example, `/home/user/miniconda3/envs/LAE/bin/python` is a Linux worker
interpreter; `C:\Miniconda3\envs\LAE\python.exe` is a Windows interpreter. Do
not commit `.env` or credentials. Model checkpoints and external source trees
must be acquired separately; copying only this repository does not install them.

For the example clone locations above, a WSL `.env` could contain the following
after replacing `/home/user`, interpreter names, checkpoint paths and model IDs
with real values:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=
ROUTER_MODEL=your-text-model
VLM_MODEL=your-image-capable-model
LAE_DINO_ROOT=/home/user/rsagent-models/LAE-DINO
LAE_CHECKPOINT=/home/user/rsagent-models/weights/lae_dino_swint_lae1m-28ca3a15.pth
LAE_BERT_DIR=/home/user/rsagent-models/LAE-DINO/weights/bert-base-uncased
SEGEARTH_ROOT=/home/user/rsagent-models/SegEarth-OV-3
SAM3_CHECKPOINT=/home/user/rsagent-models/weights/sam3.pt
UNIBUILD_ROOT=/home/user/rsagent-models/UniBuild
UNIBUILD_REPO=/home/user/rsagent-models/UniBuild
UNIBUILD_CHECKPOINT=/home/user/rsagent-models/weights/unibuild_dinov3_base_hrdpt.pth
DELINEATE_ROOT=/home/user/rsagent-models/Delineate-Anything
DELINEATE_UPSTREAM=/home/user/rsagent-models/Delineate-Anything
DELINEATE_CHECKPOINT=/home/user/rsagent-models/weights/DelineateAnythingv2.pt
LAE_PYTHON=/home/user/miniconda3/envs/LAE/bin/python
SAM_PYTHON=/home/user/miniconda3/envs/SAM/bin/python
UNIBUILD_PYTHON=/home/user/miniconda3/envs/geopython/bin/python
DELINEATE_PYTHON=/home/user/miniconda3/envs/rsagent-delineate/bin/python
```

### 4. Launch

Activate the modern environment containing the agent requirements and SegEarth
dependencies, then run:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Upload an aerial or satellite image,
enter a question, inspect the proposed processing and run it. The interface
shows the response, model evidence and errors; available outputs can be
downloaded. First use of a model can take time while its worker loads.

## Data, coordinates and outputs

Supported uploads include PNG, JPEG, WebP and TIFF. GeoTIFF input retains its
source CRS and affine transform for compatible exported artifacts. TIFF channels
are selected with `RGB_BANDS`; non-byte imagery is display-stretched using valid
pixel percentiles. PNG/JPEG lack geographic metadata and use a local pixel
coordinate system; their outputs do not imply a real-world CRS or metre-scale
measurements.

Runs are saved under `runs/`; worker logs are in `logs/`. Outputs vary with the
request and available tools. Classification can produce a color map, class-ID
PNG and GeoTIFF, and JSON mapping. Field processing can produce `fields.gpkg`
with native, semantic and selected layers, instance and binary GeoTIFF masks,
`field_proposals.json`, semantic evidence and trace files. Field proposal JSON
contains pixel geometry and source metadata; it is not RFC 7946 GeoJSON.
Internal Delineate staging files use a private pixel grid; public exports use the
source transform where available. Review results before using them in GIS or
scientific analysis.

## Configuration and troubleshooting

- The sidebar displays worker interpreter paths. A missing interpreter, model
  repository, checkpoint, incompatible native library or failed model load is
  reported in the run trace or `logs/`; configure the path in `.env` and restart
  Streamlit after changes.
- API failures require a working endpoint, a valid key and compatible model IDs.
  Image-capable inference is needed when the VLM is selected. Endpoint usage can
  incur charges.
- Field settings include `DELINEATE_CONFIDENCE`,
  `DELINEATE_REMAINING_AREA`, `FIELD_NONAG_REJECT_FRACTION` and
  `FIELD_CROPLAND_SUPPORT_FRACTION`. These are operational thresholds, not
  universally validated scientific parameters. See `config.py` for defaults.
- Change `TILE_SIZE` and `TILE_STRATEGY` for large images; field requests use
  Delineate's native full-scene workflow instead of external agent tiling.
- A failed required segmentation or model step should not be interpreted as a
  valid empty prediction. Inspect the trace and worker log.

## Validation and reproducibility

With the agent environment active:

```bash
python selftest.py
python -m unittest -v test_fields
```

Additional scripts include `smoketest_detection.py`, `smoketest_unibuilding.py`
and `smoketest_fields.py`; these need their respective configured model files
and environments. The field smoke uses a real Delineate checkpoint but defaults
to controlled semantic and VLM stubs. Its `--real-segearth` and `--live-vlm`
flags enable those dependencies explicitly. Inspect each script's arguments
and outputs before running it. Use independent annotated data, an appropriate
spatial split and task-specific metrics to evaluate scientific performance.

For each experiment, retain the original input, source metadata, question, plan,
trace, parameters and software/model versions. Avoid treating a successful test
or visualization as evidence of accuracy on a new landscape.

## Project layout

```text
app.py                  Streamlit interface
planner.py, schemas.py  Request interpretation and tool plans
orchestrator.py, vlm.py Execution, evidence and VLM review
tools/, workers/        Model adapters and isolated worker processes
field_geometry.py       Image and coordinate handling
tiling.py, masks.py     Tiling and output masks
requirements_*.txt      Environment-specific integration dependencies
selftest.py, test_fields.py, smoketest_*.py  Validation
```

## Research references

Please cite the original methods and model repositories when using their
outputs: [LAE-DINO](https://github.com/jaychempan/LAE-DINO),
[SegEarth-OV-3](https://github.com/earth-insights/SegEarth-OV-3) and
[SAM 3](https://huggingface.co/facebook/sam3),
[UniBuild](https://github.com/zhu-xlab/UniBuild), and
[Delineate Anything v2](https://arxiv.org/abs/2607.19069). Consult each source
for the recommended citation and license. No accuracy figures in an upstream
paper establish the accuracy of this integrated agent on your imagery.
