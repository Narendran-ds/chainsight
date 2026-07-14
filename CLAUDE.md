# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ChainSight is a warehouse safety reasoning pipeline: single fixed camera, offline batch
processing of recorded video (not real-time). It detects objects/people (YOLOv8), tracks
them (ByteTrack), reasons about zones and proximity (Shapely), builds a scene graph
(NetworkX), and evaluates deterministic safety rules against it. See `docs/architecture.md`
for a narrative walkthrough of the pipeline, `docs/scope_and_limitations.md` for the
authoritative scope/limitations writeup (system boundaries, known weak classes, the
correlation-vs-causation framing for rule output), `docs/demo_scenarios.md` for what to say
when walking someone through the demo, and `docs/progress.md` for a running checklist of
what's completed vs. remaining.

**Only the YOLOv8 detector is a trained/learned model.** Every downstream stage — tracking,
spatial reasoning, world-graph construction, and the rule engine — is deterministic,
non-learned Python. This is a deliberate design choice (see `docs/scope_and_limitations.md`
§3): it keeps the reasoning chain fully auditable back to explicit code logic rather than
compounding uncertainty across multiple learned models. Preserve this property when
extending the pipeline — don't quietly introduce a second learned model into the
spatial/graph/rules stages.

## Implementation status

Read this before assuming a file works. A meaningful chunk of the repo is currently **empty
stub files (0 bytes)** — they exist as placeholders for planned modules but contain no code:

- `src/vision/detector.py` (Stage 1 detection-only wrapper — `tracker.py` currently does
  detection+tracking together via `model.track()`)
- `src/narration/nim_client.py` — the NVIDIA NIM narration provider is not yet implemented
  (Gemini, via `gemini_client.py`, is — see Narration Layer below)
- `scripts/run_finetune.py`, `scripts/annotate_clips.py`
- `configs/rules_config.yaml`, `configs/zones_config.yaml` — rule/zone config is currently
  wired via CLI args and dataclass defaults (`RuleEngineConfig`, `SpatialConfig`), not these
  YAML files

**Implemented and working:** `src/vision/tracker.py`, `src/vision/train.py`,
`src/spatial/*`, `src/world_graph/*`, `src/rules/*`, `src/utils/{track_quality,video_io,visualization}.py`,
`src/pipeline.py` (single-entry orchestrator, `scripts/run_pipeline.py` is its CLI, and now
also writes `outputs/manifest_<run>.json`), `src/narration/{gemini_client,narrator,prompt_templates}.py`
(`scripts/run_narration.py` is its CLI — see Narration Layer below), `demo/app.py` +
`demo/components/*` (Streamlit demo — see Demo Layer below), `README.md`,
`docs/{architecture,demo_scenarios,scope_and_limitations,progress}.md`, and the data-prep/CLI
scripts listed below.

## Commands

```
pip install -r requirements.txt
```

### Data preparation (source datasets → training-ready split)

Order matters: prepare → split → verify, before training.

```
python scripts/prepare_data.py --report      # dry run: print merge stats only
python scripts/prepare_data.py                # merge data/raw/{arvist,forklift,pallet,exit_dataset_1,exit_dataset_2}
                                                # into data/processed/chainsight_dataset/, per data/class_mapping.yaml
python scripts/split_three_way.py --val-ratio 0.15 --test-ratio 0.08 --seed 42
                                                # leakage-safe: groups by source image (strips Roboflow .rf.<hash>
                                                # suffix) before splitting, so augmented variants of one source
                                                # image can't land in different splits
python scripts/verify_no_leakage.py            # confirms zero source-image overlap across train/val/test
```

### Training

```
python src/vision/train.py --run-name <name> [--epochs 100 --imgsz 640 --batch 16]
python src/vision/train.py --run-name check --dry-run     # validate dataset/device/config, no training
python src/vision/train.py --run-name <name> --resume     # only if models/finetuned/<name>/weights/last.pt exists
```

Requires `data/processed/chainsight_dataset/data.yaml` with an **absolute** `path:` (relative
paths resolve against CWD, not the yaml's location — see `train.py` module docstring).
Appends a row to `models/model_registry.md` on completion; logs to W&B if installed and
`wandb login` has been run.

### Zone setup

```
python scripts/define_zones.py --video <clip.mp4> --frame 0 --out configs/zones.json
```

Interactive OpenCV point-and-click tool. Zone `type` (`restricted` / `exit` / other) drives
rule behavior — see Rule Engine below.

### Reasoning pipeline (run in order, chained through `outputs/*.json`)

`scripts/run_pipeline.py` (backed by `src/pipeline.py`'s `ChainSightPipeline`) chains all
stages in one command:

```
python scripts/run_pipeline.py --model <weights.pt> --video <clip.mp4> --zones configs/zones.json --run-name <name> [--narrate]
```

`--narrate` additionally runs the Gemini narration stage (requires `GEMINI_API_KEY` — see
`.env.example`), writing `outputs/narration_<run-name>.json`. Each stage is still independently
runnable by hand via its own CLI script, e.g. for iterating on one stage without re-running
the (slow) tracker:

```
python src/vision/tracker.py --model <weights.pt> --video <clip.mp4> --out outputs/tracks.json [--annotated outputs/annotated.mp4]
python scripts/run_spatial.py --tracks outputs/tracks.json --zones configs/zones.json --out outputs/spatial_events.json
python scripts/run_world_graph.py --tracks outputs/tracks.json --spatial outputs/spatial_events.json --out outputs/world_graph_summary.json
python scripts/run_rules.py --tracks outputs/tracks.json --spatial outputs/spatial_events.json --zones configs/zones.json --out outputs/rule_events.json
python scripts/run_narration.py --rule-events outputs/rule_events.json --out outputs/narration.json
```

### Demo

```
streamlit run demo/app.py
```

Browses whichever runs already exist under `outputs/` (a run only gets `outputs/manifest_<run>.json`
— needed to locate its source video — if it was produced via `scripts/run_pipeline.py`, not the
individual `scripts/run_*.py` stages by hand).

### Tests

```
pytest
```

Collects real regression tests from `test_spatial.py`, `test_rules.py`, `test_pipeline.py`,
`test_narration.py`, `test_visualization.py`, and `test_run_data.py` (58 tests total). Run a
single file with `pytest tests/test_rules.py` or a single test with
`pytest tests/test_rules.py::test_name`.

## Architecture

Five-stage pipeline, each stage reading the previous stage's JSON output from `outputs/`:

```
YOLOv8 detection + ByteTrack   ->  tracks.json
        (tracker.py)
              |
              v
Shapely zone/proximity analysis -> spatial_events.json
        (spatial/analyzer.py)
              |
              v
NetworkX scene graph            -> world_graph_summary.json
        (world_graph/graph_builder.py)
              |
              v
Deterministic rule engine       -> rule_events.json
        (rules/engine.py)
              |
              v
Narration (Gemini, optional)    -> narration_<run>.json
        (narration/narrator.py)
              |
              v
Streamlit demo (browses precomputed runs — demo/app.py)
```

**Vision layer** (`src/vision/`): `tracker.py`'s `ChainSightTracker` wraps
`ultralytics.YOLO(...).track()` (built-in ByteTrack) to produce persistent per-object
`Track`s with velocity, lifecycle status (`alive`/`lost`/`reappeared`/`dead`), and occlusion
recovery, exported to `tracks.json`. `tracks.json` also carries a top-level `"_meta"` key
(`{"frame_width", "frame_height"}`, captured once via `cv2.VideoCapture` at track time) — any
new consumer iterating `tracks.json`'s track-id keys must pop/skip `"_meta"` first (see
`SpatialAnalyzer.load_tracks` / `GraphBuilder._load_tracks` for the pattern). `train.py`
fine-tunes YOLOv8 on the merged dataset and auto-appends results to `models/model_registry.md`.

**Spatial layer** (`src/spatial/`): `zones.py` loads zone polygons from `zones.json`
(`Zone.zone_type` is the string that rules key off of — e.g. `"restricted"`, `"exit"`).
`overlap.py` computes bbox-vs-zone containment (bbox-overlap-ratio by default, or
centroid-in-polygon via `--centroid-only-zones`). `analyzer.py` orchestrates per-frame zone
containment, `entered`/`left` zone-transition events, and deduplicated proximity pairs (via
Shapely `STRtree`, squared-distance until the final sqrt) into `spatial_events.json`.
`SpatialConfig.proximity_threshold_px` is calibrated at `video_io.REFERENCE_FRAME_WIDTH`
(1920px) and scaled per-clip by `utils/video_io.py`'s `resolution_scale()` against the actual
`frame_width` from `tracks.json`'s `"_meta"` — this is what keeps a fixed pixel threshold from
silently never firing on 4K footage (or over-firing on low-res footage). See
`docs/scope_and_limitations.md` §5 for the known remaining limitation (resolution scaling
doesn't correct for camera framing/distance-to-scene, only sensor resolution).

**World graph layer** (`src/world_graph/`): `graph_builder.py`'s `GraphBuilder` consumes
`tracks.json` + `spatial_events.json` and builds `nx.Graph`s — either per-frame (nodes =
tracks + zones, edges = `inside_zone` / `near`) or one aggregate summary graph for the whole
clip. Ghost/churned tracks (short-lived tracker-ID fragments) are filtered via
`utils/track_quality.py`'s `filter_valid_tracks` (`min_track_frames`, default 15) — this is
the single shared definition of "valid track" for any consumer of `tracks.json`, not
reimplemented per layer. `query.py`'s `WorldQuery` is a thin domain-language façade over the
graph (`objects_in_zone`, `neighbors`, `zone_history`, ...) — this is the only interface the
rule engine should need into `world_graph`; avoid reaching into `GraphBuilder`/NetworkX
internals from `rules/`.

**Rule engine** (`src/rules/`): `engine.py`'s `RuleEngine` evaluates five deterministic
rules against `WorldQuery` + zone-type metadata:

| Rule | Fires when |
|---|---|
| R1 Restricted Zone Intrusion | person enters a `restricted`-type zone |
| R2 Forklift-Pedestrian Near-Miss | person+forklift stay within `near_miss_distance_px` inside a restricted zone for N consecutive frames |
| R3 PPE Violation | a `no_vest`/`no_helmet` detection's bbox overlaps a person's bbox above threshold (asymmetric overlap ratio, not IoU — see `rule_definitions.bbox_overlap_ratio` docstring for why) |
| R4 Exit Blockage | a non-person object occupies an `exit`-type zone continuously past a duration threshold (computed from continuous per-frame presence, not entered/left pairs, since a blocker's track can go dead while still inside the zone) |
| R5 Loitering in Restricted Zone | person stays inside a restricted zone past a duration threshold |

Every fired rule produces a `RuleEvent` (`rule_id`, `trigger`, `evidence`, `conclusion`) —
never a bare boolean — per the project's explainability requirement (`events.py`). Rule
thresholds live in `RuleEngineConfig` (`rule_definitions.py`), wired from CLI flags in
`scripts/run_rules.py`. Note `near_miss_distance_px` must stay `<=` spatial layer's
`proximity_threshold_px`, since a `near` edge only exists in `spatial_events.json` within
that threshold at all — `PipelineConfig.__post_init__` (`src/pipeline.py`) enforces this at
construction time. Both are calibrated in the same reference-resolution units; `RuleEngine`
scales `near_miss_distance_px` by the same `resolution_scale()` factor the spatial layer used
(via `GraphBuilder.frame_width`), so the invariant holds regardless of the clip's actual
resolution.

R1/R2 have no way to distinguish a vehicle **operator** from a **pedestrian** — both are
just the `person` class — so a forklift driver visible through an open cab can be detected as
a second `person` track nested inside the forklift's own bbox, firing R1 (and inflating R2
proximity counts) for the driver rather than a real bystander. Confirmed on a validation clip
(`aisle_test`, not one of the two primary demo clips: `nearmiss`/`blocked_exit`, where this
doesn't occur) — see `docs/scope_and_limitations.md` §7 before trusting an R1/R2 event on any
new clip with a visibly-seated driver.

**Narration layer** (`src/narration/`): `narrator.py`'s `Narrator` consumes `rule_events.json`
and produces plain-English narration via Gemini (`gemini_client.py`, using the `google-genai`
SDK — see requirements.txt for why not the deprecated `google-generativeai`) — one sentence
per fired `RuleEvent` plus one aggregate summary paragraph for the clip, written to
`outputs/narration_<run>.json` (never merged into `rule_events.json`, so that file's schema is
unaffected). `prompt_templates.py` holds the prompt strings and the shared system instruction:
narration is **constrained rephrasing** of what the deterministic rule engine already decided
— no causal claims, blame, or severity judgments beyond the input — preserving this project's
single-learned-model design principle (only YOLOv8 trains; narration doesn't reason, it
rephrases). `gemini_client.py` retries only on transient errors (HTTP 429/5xx) with exponential
backoff; anything else fails immediately. `nim_client.py` (an alternate NVIDIA NIM provider) is
not yet implemented.

`GeminiClientConfig.model` defaults to `"gemini-flash-latest"`, not a pinned version like
`gemini-2.5-flash` — verified live against a real API key: `gemini-2.5-flash`/`-lite` returned
404 ("no longer available to new users") and `gemini-2.0-flash` variants returned 429 with a
hard `limit: 0` free-tier quota (not a transient rate limit) on that project. The `-latest`
alias and the `gemma-4-*` models were the ones with actual free-tier quota. If narration calls
start failing with 404/429-limit-0, check which model names currently have quota for the
configured key before assuming the code is broken.

**Demo layer** (`demo/`, built 2026-07-14): Streamlit app (`streamlit run demo/app.py`),
browsing **precomputed runs only** — it never triggers a live pipeline run from the UI
(tracker.py's YOLO+ByteTrack pass is the slow, GPU-bound stage; re-running it live during a
demo/interview is unnecessary risk for zero benefit over pre-generating outputs).

- **`demo/components/run_data.py`**: pure data-loading/discovery (no Streamlit import) —
  `discover_runs()` scans `outputs/` for `tracks_<run>.json` (a bare `tracks.json` with no
  suffix is the `""`/"(default)" run); `resolve_run_paths()` returns `None` for any of
  manifest/tracks/spatial_events/world_graph_summary/rule_events/narration that don't exist
  for a run, so the UI can degrade gracefully instead of crashing. `index_tracks_by_frame()`
  builds a `frame_idx -> [tracks]` dict once per run (mirrors `analyzer.py`'s `frame_buckets`
  pattern) so the frame scrubber is an O(1) lookup per slider move, not a full history rescan.
  `derive_fps()` recovers the clip's fps from `tracks.json`'s existing `frame`/`timestamp`
  relationship (`timestamp = round(frame / fps, 4)`, set once by `tracker.py`) rather than
  needing a dedicated `fps` field in `"_meta"` — works on `tracks.json` files written before
  the frame<->time jump widgets (below) needed it.
- **`demo/components/clip_picker.py`**: sidebar run selector, headed by a bold "⚙️ Controls"
  heading + divider (matches a reference layout provided during the build). `st.stop()`s with
  a clear message if `outputs/` has no runs yet.
- **`demo/components/video_player.py`**: the frame-index scrubber (`st.session_state["current_frame"]`
  is the single source of truth, shared with `event_timeline.py`) plus two number inputs —
  "Jump to frame" (int) and "Jump to time (s)" (float, via `derive_fps()`) — kept in lockstep
  with the slider. Streamlit doesn't allow multiple widgets to share one key, so each of the
  three has its own (`frame_slider`/`frame_number_input`/`time_number_input`); an `on_change`
  callback on whichever widget fired writes the canonical `current_frame` plus all three
  widgets' own keys via a shared `_sync()` helper, and `render()` re-runs that same `_sync()`
  once at the top (before any widget is instantiated) so an *external* jump — `event_timeline.py`
  setting `current_frame` directly from a row click — also propagates to all three. Verified
  with Streamlit's `AppTest` in every direction (slider->inputs, each input->slider+other input,
  external jump->all three). Overlays (bboxes, track IDs, zone polygons) are drawn **live** from
  `tracks.json` + `zones.json` via `src/utils/visualization.py` — deliberately *not* replaying
  `tracker.py`'s pre-rendered `--annotated` video, so a rule event's frame can be jumped to
  directly with correct zone overlays regardless of whether an annotated video was ever
  generated for that run. Falls back to a blank canvas (`visualization.blank_canvas`) with a
  caption if the run's manifest is missing or the source video file can't be opened, rather
  than erroring.
- **`demo/components/event_timeline.py`**: rule-event table using Streamlit's native
  `st.dataframe(..., on_select="rerun", selection_mode="single-row")` — clicking a row sets
  `st.session_state["current_frame"]` before `video_player.py` instantiates its slider in the
  same script run, jumping the scrubber to that event's frame. Chosen over the
  `streamlit-timeline` package in `requirements.txt` (now commented out) because that package
  is a very early (0.0.2), thinly-maintained component; native selection is simpler. Pinned
  `streamlit==1.50.0` as the known-good, tested version. Neither `st.dataframe` nor `st.image`
  pass any width-related kwarg (no `width="stretch"`, no `use_container_width`) — a real,
  differently-installed Streamlit hit two different crashes in succession trying both: first
  `TypeError: 'str' object cannot be interpreted as an integer` on `width="stretch"` (that
  string API postdates the version installed there), then `TypeError: ImageMixin.image() got
  an unexpected keyword argument 'use_container_width'` on the very same env for `st.image`
  (already past that param's removal there). The two Streamlit APIs for "full width" have each
  been renamed/removed on different timelines per element, so don't reintroduce either without
  confirming the target Streamlit version first — omitting the param entirely is what's
  actually version-safe.
- **`demo/components/report_card.py`**: narration panel — clip summary + per-event narration
  from `narration_<run>.json`, or the exact `scripts/run_narration.py` command to generate it
  if that file is absent for the selected run.
- **`demo/components/internals.py`**: collapsed-by-default expander with raw tracks/spatial/
  world-graph JSON — supporting detail, not the headline (rules + narration are front-and-center
  in `demo/app.py`'s layout).
- **`src/utils/visualization.py`**: pure OpenCV drawing helpers (`draw_tracks_on_frame`,
  `draw_zones_on_frame`, `blank_canvas`) — no file I/O, no Streamlit, so drawing logic is
  unit-tested independently of a real video file. Colors are BGR (matching `cv2` convention);
  fixed for `person`/`forklift`, hash-derived (but deterministic) for any other class so all 17
  dataset classes render consistently without hardcoding each one.
- **Manifest artifact**: `src/pipeline.py` now writes `outputs/manifest_<run>.json`
  (`video_path`, `zones_path`, `model_path`) at the start of every `ChainSightPipeline.run()` —
  this is what `video_player.py` needs to locate the source video and `zones.json` for a given
  `run_name`, since neither existed anywhere in `outputs/` before. Only the full pipeline run
  produces one; a run assembled by hand from individual `scripts/run_*.py` calls won't have
  one, and the demo falls back to a zone-only canvas rather than erroring.

**Config dataclass pattern**: every layer follows the same shape — a `*Config` dataclass
(`TrackerConfig`, `SpatialConfig`, `WorldGraphConfig`, `RuleEngineConfig`,
`TrackQualityConfig`, `GeminiClientConfig`) holding tunables, constructed once, passed into the stage's main class,
with the corresponding `scripts/run_*.py` CLI translating `argparse` flags into it. Follow
this pattern for new tunables rather than adding ad-hoc function parameters.

**Data prep pipeline** (`scripts/`, `data/`): three Roboflow-sourced datasets
(`arvist`, `forklift`, `pallet`) plus two exit-sign datasets (`exit_dataset_1/2`) are merged
via `data/class_mapping.yaml` (per-source `original_classes` → 17-class master list) into
`data/processed/chainsight_dataset/` (`prepare_data.py`). Arvist requires filtering:
images whose only annotations are traffic classes (car/bus/truck/etc.) are dropped entirely
rather than remapped. `split_three_way.py` then recombines any existing train/val/test back
into one pool and re-splits by **source-image group** (not by file) so Roboflow-augmented
`.rf.<hash>` variants of one source image can't leak across splits — `verify_no_leakage.py`
confirms this. Current best/reference checkpoint: `run2_exit_marker` (see
`models/model_registry.md`, `docs/scope_and_limitations.md` for known weak classes).

## Conventions

- Every stage-level module (`tracker.py`, `analyzer.py`, `graph_builder.py`, `engine.py`)
  opens with a docstring stating its pipeline position (what it consumes/produces) and any
  non-obvious design decisions — read that docstring before modifying the module.
- File I/O boundaries (loading `tracks.json`, `zones.json`, `spatial_events.json`) validate
  required keys up front and raise loudly (`FileNotFoundError`/`ValueError`) rather than
  producing a silently empty/wrong downstream result — follow this when adding new loaders.
- `logging` (not `print`) is used inside `src/`; CLI scripts under `scripts/` use `print`
  for user-facing summaries. Match the existing style within whichever file you're editing.
- Scripts add the repo root to `sys.path` (`sys.path.append(str(Path(__file__).resolve().parent.parent))`)
  to import `src.*` without installing the package — there is no `setup.py`/`pyproject.toml`.
