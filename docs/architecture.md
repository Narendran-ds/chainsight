# ChainSight — Architecture

This document describes how ChainSight turns a recorded warehouse video into a set of
explainable safety-rule findings and plain-English narration. See
[`scope_and_limitations.md`](scope_and_limitations.md) for what the system deliberately does
*not* do, and [`progress.md`](progress.md) for current build status.

---

## 1. The five-stage pipeline

ChainSight is five deterministic reasoning stages chained through JSON files on disk, fed by
one trained model at the very front:

```
                     ┌─────────────────────┐
   video.mp4  ─────► │  1. Vision           │  YOLOv8 detection + ByteTrack tracking
                     │     tracker.py       │  (the ONLY trained/learned stage)
                     └──────────┬───────────┘
                                 │ tracks.json
                                 ▼
                     ┌─────────────────────┐
                     │  2. Spatial          │  zone containment + proximity (Shapely)
                     │     spatial/analyzer │
                     └──────────┬───────────┘
                                 │ spatial_events.json
                                 ▼
                     ┌─────────────────────┐
                     │  3. World Graph       │  per-frame + summary scene graphs (NetworkX)
                     │     world_graph/*     │
                     └──────────┬───────────┘
                                 │ world_graph_summary.json
                                 ▼
                     ┌─────────────────────┐
                     │  4. Rule Engine       │  5 deterministic safety rules (R1-R5)
                     │     rules/engine.py   │
                     └──────────┬───────────┘
                                 │ rule_events.json
                                 ▼
                     ┌─────────────────────┐
                     │  5. Narration         │  Gemini — constrained rephrasing only
                     │     narration/*       │  (optional, --narrate)
                     └──────────┬───────────┘
                                 │ narration_<run>.json
                                 ▼
                     ┌─────────────────────┐
                     │  Streamlit Demo       │  browses precomputed runs, draws overlays
                     │     demo/app.py       │  live — never re-runs the pipeline itself
                     └─────────────────────┘
```

Every arrow is a JSON file written to `outputs/`, not an in-memory handoff — each stage's own
CLI script (`scripts/run_*.py`) can be run in isolation against the previous stage's output, or
`scripts/run_pipeline.py` chains all five in one call. Both paths go through the same file
boundary, so schema drift between stages is caught the same way either way (see `src/pipeline.py`'s
module docstring).

### Stage 1 — Vision (`src/vision/tracker.py`)

Wraps `ultralytics.YOLO(...).track()`, which runs YOLOv8 detection and ByteTrack tracking
together in one call. Produces a persistent `Track` per object with velocity, lifecycle status
(`alive` / `lost` / `reappeared` / `dead`), and occlusion recovery. Exports `tracks.json`: a
dict keyed by track ID, plus a `"_meta"` key (`frame_width`, `frame_height`) captured once from
the source video — every downstream consumer of `tracks.json` pops/skips that key before
iterating tracks.

### Stage 2 — Spatial (`src/spatial/`)

Loads hand-defined zone polygons (`zones.json`, each tagged `restricted` / `exit` / other) and,
per frame, computes: which zones each track is inside (bbox-overlap ratio by default), zone
`entered`/`left` transition events, and deduplicated proximity pairs between nearby tracks (via
a Shapely `STRtree` spatial index, squared-distance comparison until the final `sqrt`).
Pixel-distance thresholds here are calibrated against a 1920px reference width and scaled by
the clip's actual resolution — see `scope_and_limitations.md` §5 for why, and its limits.

### Stage 3 — World Graph (`src/world_graph/`)

Builds `networkx.Graph`s from `tracks.json` + `spatial_events.json` — either one graph per
frame (nodes = tracks + zones, edges = `inside_zone` / `near`) or one aggregate summary graph
for the whole clip. Ghost/churned tracks (short-lived tracker-ID fragments from ByteTrack ID
switches) are filtered by a single shared `filter_valid_tracks()` helper
(`src/utils/track_quality.py`), so every downstream consumer of `tracks.json` agrees on what
counts as a "real" track rather than each reimplementing its own filter. A thin query façade
(`query.py`'s `WorldQuery`) is the only interface the rule engine uses into this layer —
domain-language methods like `objects_in_zone()`/`neighbors()`/`zone_history()` instead of raw
graph traversal, so a rule change never needs to know NetworkX internals.

### Stage 4 — Rule Engine (`src/rules/engine.py`)

Evaluates five deterministic rules against `WorldQuery` plus zone-type metadata:

| Rule | Fires when |
|---|---|
| **R1** Restricted Zone Intrusion | a person enters a `restricted`-type zone |
| **R2** Forklift-Pedestrian Near-Miss | a person and forklift stay within a (resolution-normalized) proximity threshold, inside a restricted zone, for N consecutive frames |
| **R3** PPE Violation | a `no_vest`/`no_helmet` detection's bbox overlaps a person's bbox above a threshold |
| **R4** Exit Blockage | a non-person object occupies an `exit`-type zone continuously past a duration threshold |
| **R5** Loitering in Restricted Zone | a person stays inside a restricted zone past a duration threshold |

Every fired rule produces a `RuleEvent` — `rule_id`, `trigger`, `evidence`, `conclusion` — never
a bare boolean. This is a deliberate explainability requirement: a reviewer (or the narration
stage) can read exactly what fired and why without re-deriving it from raw graph state.

### Stage 5 — Narration (`src/narration/`, optional)

Consumes `rule_events.json` and produces plain-English text via Gemini: one sentence per fired
event, plus one clip-level summary. This stage is **constrained rephrasing, not reasoning** —
the shared system instruction explicitly forbids adding causal claims, blame, or severity
judgments beyond what the rule engine already decided (see §2 below for why this constraint
exists). Written to a separate `narration_<run>.json` file rather than merged into
`rule_events.json`, so the rule engine's own output schema is never touched by this optional
stage.

### The demo (`demo/`)

A Streamlit app that browses **already-computed** runs under `outputs/` — it never triggers a
live pipeline run itself. Overlays (bboxes, zone polygons) are drawn live from `tracks.json` +
`zones.json` for whichever frame is selected, rather than replaying a pre-rendered annotated
video, so a rule event's frame can be jumped to directly with correct overlays regardless of
whether an annotated video was ever generated for that run.

---

## 2. Design principle: only one learned model in the whole chain

**YOLOv8 (Stage 1's detector) is the only trained/learned component in ChainSight.** Tracking,
spatial reasoning, world-graph construction, the rule engine, and narration are all
deterministic, non-learned Python — plain geometry, graph traversal, and threshold
comparisons, with narration constrained to rephrasing rather than reasoning.

This is a deliberate architectural choice, not an accident of what got built first. It matters
for two concrete reasons:

1. **Auditability.** Every decision downstream of detection can be traced back to explicit code
   logic, not a second model's learned weights. If R2 fires, you can point at the exact
   `if near["distance_px"] > self.near_miss_distance_px` comparison that fired it — there's no
   "the model decided" black box past the bounding boxes themselves.
2. **No compounding uncertainty.** Chaining multiple learned models (e.g. a learned tracker, a
   learned zone classifier, an LLM making the actual safety judgment) means each stage's error
   rate compounds into the next. Keeping everything after detection deterministic means the
   system's overall reliability is bounded by one model's accuracy, not the product of several.

Extending the pipeline should preserve this: a new capability that requires "an ML model to
decide X" belongs either inside the detector's class list (add a class, retrain) or as a
narrowly-scoped narration constraint — not as a new learned component in the
spatial/graph/rules stages.

---

## 3. The config dataclass pattern

Every stage follows the same shape, and new tunables should follow it too rather than adding
ad-hoc function parameters:

- A `*Config` dataclass holds every tunable for that stage — `TrackerConfig`, `SpatialConfig`,
  `WorldGraphConfig`, `RuleEngineConfig`, `TrackQualityConfig`, `GeminiClientConfig`.
- The stage's main class takes a `Config` instance once at construction (`ChainSightTracker`,
  `SpatialAnalyzer`, `GraphBuilder`, `RuleEngine`, `GeminiClient`).
- The corresponding `scripts/run_*.py` CLI translates `argparse` flags into that `Config`,
  keeping command-line ergonomics separate from the stage's actual logic.

This keeps every stage's tunables discoverable in one place (the dataclass definition) and
means `src/pipeline.py`'s orchestrator can validate cross-stage invariants at construction time
— e.g. `PipelineConfig.__post_init__` rejects a `near_miss_distance_px` greater than
`proximity_threshold_px` before any stage even runs, because a `near` edge can't exist in
`spatial_events.json` above that threshold in the first place.

---

## 4. Data artifacts per run

Every full pipeline run (`scripts/run_pipeline.py --run-name <name>`) writes, under `outputs/`:

| File | Written by | Contents |
|---|---|---|
| `manifest_<run>.json` | pipeline orchestrator | source `video_path`/`zones_path`/`model_path` — lets the demo locate a run's source video |
| `tracks_<run>.json` | Stage 1 | per-track history + `"_meta"` (frame_width/height) |
| `spatial_events_<run>.json` | Stage 2 | per-frame zone containment, transitions, proximity pairs |
| `world_graph_summary_<run>.json` | Stage 3 | aggregate scene graph (node-link JSON) |
| `rule_events_<run>.json` | Stage 4 | fired `RuleEvent`s |
| `narration_<run>.json` | Stage 5 (optional) | per-event narration + clip summary |

A run assembled by hand from individual `scripts/run_*.py` calls (rather than the full
orchestrator) may be missing `manifest_<run>.json` or `narration_<run>.json` — the demo
degrades gracefully in both cases rather than erroring (see `demo/components/run_data.py`).
