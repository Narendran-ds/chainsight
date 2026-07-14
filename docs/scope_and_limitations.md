# ChainSight — Scope and Limitations

This document records known limitations, design boundaries, and honest caveats
for the ChainSight warehouse safety reasoning pipeline. It is intended to be
read alongside `model_registry.md` and referenced directly in interviews or
demos — the goal is transparency, not marketing.

---

## 1. System Scope

ChainSight is designed and validated for:

- **Single fixed camera**, static mount, consistent field of view
- **Offline batch processing** of recorded video (not a real-time streaming system)
- **Indoor warehouse environments** resembling the training distribution
  (forklifts, pallets, PPE, exit signage)

It is explicitly **not** designed for multi-camera fusion, live/real-time
alerting, outdoor environments, or camera angles substantially different from
the training data. These would require additional data collection and are
out of scope for the current portfolio version.

### 1.1 Real-time / streaming is a non-goal, not a missing feature

ChainSight processes a complete, already-recorded clip — every stage assumes
the full frame range is available before it runs. This is a deliberate
architectural boundary, not something left unfinished:

- **Rule evaluation is stateless per run, not per frame.** `RuleEngine.run()`
  is called once against a fully-built `WorldQuery`/spatial-events dataset
  for the whole clip. There is no notion of "the pipeline is currently
  watching frame N and must decide right now" — every rule looks at
  data that already exists on disk.
- **R2/R4/R5 all require full-clip lookback.** R2's consecutive-frame
  counter, R4's continuous-presence duration, and R5's enter/leave duration
  all need to see a track's *entire* history (or at least look ahead to
  where a streak ends) to decide whether a threshold was met. In a batch
  pipeline this is a simple loop over `spatial_events.json`; in a live
  stream there is no "end of the streak" to look ahead to yet.
- **Tracker-level occlusion recovery (`lost`/`reappeared`) is easier
  offline.** `tracker.py` can be more forgiving about gaps because it never
  has to commit to a decision under a real-time deadline.

**What genuine real-time support would require** is a different
architecture, not a flag on this one: stateful sliding-window processing
(bounded lookback instead of full-clip access), live frame ingestion instead
of a file on disk, and persistent per-track occupancy timers that update
incrementally per frame rather than being computed once over the whole
history. That's a rewrite of the spatial/rules layers' core assumptions, not
an extension of them — hence a scope boundary rather than a roadmap item.

---

## 2. Detection Model (YOLOv8) — Known Limitations

Reference run: `run2_exit_marker` (2026-07-09), best checkpoint epoch 53.
Overall mAP50: 0.7678, mAP50-95: 0.5140, across 17 classes.

### 2.1 `exit_zone_marker` — resolved gap

Earlier versions of the dataset had insufficient `exit_zone_marker` instances,
producing a persistent detection gap for this class. This was resolved by
merging two additional Roboflow exit-sign datasets, adding 689 real instances
across train/val/test. In `run2_exit_marker`, this class reached **AP50 =
0.8892**, exceeding the overall mAP50 — the gap is considered resolved.

### 2.2 Low-support classes — not statistically meaningful

Two classes have very low instance counts in the validation set and their
AP50 numbers should **not** be read as reliable signal in either direction:

| Class | Val instances | AP50 | Interpretation |
|---|---|---|---|
| `open_box` | 1 | 0.0120 | Single instance — one miss produces a near-zero score. Not evidence the model can't detect this class. |
| `package` | 2 | 0.9950 | Near-perfect score is not evidence of robust generalization — too few examples to trust. |

**Decision:** these numbers are deliberately not "chased" (e.g. by
oversampling or rebalancing) since doing so risks overfitting to a handful of
val images rather than genuinely improving the model. This is flagged as a
known data limitation rather than a modeling failure.

### 2.3 Real, higher-support weak classes

Two classes have reasonable support (>25 val instances) and meaningfully
lower AP50 than the rest of the model — these are genuine weaknesses, not
statistical noise:

| Class | Val instances | AP50 | Likely cause |
|---|---|---|---|
| `glove` | 98 | 0.4753 | Small object, frequently occluded by hands/tools, fast relative motion |
| `no_vest` | 29 | 0.5576 | Lower instance count than most PPE classes; visually similar to partial/angled `vest` views |

**Recommended next step (not yet done):** additional labeled data for these
two classes, and/or targeted augmentation (motion blur, partial occlusion)
if this project is extended beyond portfolio scope.

### 2.4 Label quality caveat — mixed box/segment annotations

During validation, ultralytics logged a warning that the merged dataset
contains a mix of segmentation polygons and plain bounding boxes
(26,490 segments vs. 28,707 boxes). Since ChainSight trains a detection-only
model, segments are automatically dropped and only boxes are used. This means
a subset of annotations — likely from one of the merged exit-sign datasets —
may have auto-derived boxes (tightest rectangle around a polygon) rather than
hand-drawn boxes. This is not expected to meaningfully affect the metrics
above, but is noted here for completeness since it wasn't a deliberate design
choice.

---

## 3. Pipeline-Wide Design Note

Only the YOLOv8 detector (Section 2) is a trained/learned model. Every
downstream stage — tracking (ByteTrack), spatial reasoning (Shapely),
world-state modeling (NetworkX), the rule engine, and narration — is
deterministic and does not require training. This is a deliberate scope
decision: it keeps the reasoning chain fully explainable and avoids
compounding uncertainty from multiple learned models. It also means system
behavior beyond detection accuracy is fully auditable — every downstream
decision can be traced to explicit code logic, not learned weights.

---

## 5. Rule Engine — R2 Near-Miss Distance Calibration

`proximity_threshold_px` (spatial layer) and `near_miss_distance_px` (R2) are
now normalized against the clip's actual frame resolution (`src/utils/video_io.py`'s
`resolution_scale()`, calibrated at a 1920px reference width — see
`tracker.py`'s `"_meta"` export and `analyzer.py`/`engine.py`'s use of it).
This was needed because the original absolute-pixel thresholds (150px / 100px)
were tuned against ~1080p footage and silently never fired on 4K clips, where
the same real-world distance spans far more pixels. The fix is confirmed
working symmetrically: it scales thresholds *up* for higher-resolution clips
and *down* for lower-resolution ones (e.g. 0.40x on the 768px-wide
`blocked_exit.mp4`).

**Known remaining limitation:** resolution scaling only corrects for sensor
resolution, not camera framing/distance-to-scene. On
`forklift_pedestrian_nearmiss.mp4` (4K, `run2_exit_marker`), R2 still does not
fire — verified by inspecting the actual frames
(`data/staged_clips/extracted_frames/nearmiss_wide/frame_00410.jpg`, the
clip's closest tracked approach): the forklift and tracked pedestrian sit on
opposite sides of an open aisle, never in genuine near-collision, across all
525 frames. The minimum tracked centroid distance anywhere in the clip is
~1060–1270px, well beyond even the resolution-corrected 300px threshold. This
is a wide establishing shot where the forklift occupies roughly 1/4 of frame
width, so a real ~1–1.5m close call still spans 1000+ raw pixels — a gap
resolution normalization alone cannot close without also producing false
positives on more tightly-framed footage.

**Decision:** this is treated as a camera-framing/test-clip limitation, not a
pipeline bug — proximity observations went from 0 (pre-fix, unscaled) to 24
(post-fix) on this clip, confirming the scaling itself behaves correctly; R2
just has no genuine near-miss instant to catch in this specific recording.
Thresholds are deliberately not inflated further to force a fire, since that
would risk false near-misses on footage where the camera sits closer to the
action.

**Recommended next step (not yet done):** normalize `near_miss_distance_px`
against detected object size (e.g. a multiple of the forklift/person bbox
diagonal) instead of frame width, so R2 is robust to camera zoom/distance as
well as resolution. Deferred as a larger design change outside current scope.

---

## 6. Correlation vs. Causation

ChainSight's rule engine is explicitly designed to describe **observed
spatial-temporal patterns** (e.g. "person entered restricted zone while
forklift was active nearby") rather than assert causal claims about why an
incident occurred. This distinction is deliberate and should be preserved in
any narration or demo framing — the system flags patterns worth human review,
it does not determine fault or root cause.

---

## 7. Vehicle Operator vs. Pedestrian Ambiguity

The pipeline has no class distinguishing a vehicle **operator** from a
**pedestrian** — both are simply detected as `person`. Normally this is
harmless (an operator sitting inside a forklift's cab is rarely detected as a
separate bounding box), but on footage where the driver is clearly visible
through an open-sided cab, YOLO can produce a second `person` track nested
entirely inside the forklift's own bounding box.

**Confirmed on a validation clip** (`forklift_person_aisle.mp4`, run name
`aisle_test`, not one of the two primary demo clips): track 294 ("person")
exists for only 18 frames (490-507) and its bbox has a 1.0 overlap ratio with
the forklift's bbox for its entire lifespan — i.e. it's the driver, not a
separate person. This fired R1 (Restricted Zone Intrusion) as if a pedestrian
had entered the zone, when in fact it was the forklift's own operator driving
through the zone they operate in. The same effect would inflate R2 near-miss
proximity counts (a driver is always ~0 real-world distance from their own
vehicle), producing spurious "near-miss" signal between a forklift and
itself.

**Checked against the two primary demo clips (`nearmiss`, `blocked_exit`):
not present.** In `nearmiss`, every `person` track has zero bbox overlap with
the forklift at any frame — the operator was never detected as a separate
track in that footage, likely because the cab is less visually open. This is
specific to footage/camera-angle, not a universal bug, but it's a real gap:
**there is currently no way to distinguish "vehicle operator" from
"pedestrian" in the class taxonomy**, so any new clip should be checked for
this before trusting an R1/R2 event involving a forklift.

**Mitigation path (not yet done):** the cleanest fix is a dedicated
`forklift_operator` class added to the dataset in a future training pass, so
the detector itself distinguishes "person driving a forklift" from "person
on foot" — this is a data/training problem, not a rules-engine one. A
cheaper interim option is a heuristic filter in `rules/engine.py`: treat a
`person` track whose bbox stays highly overlapped with a `forklift` track
for its entire lifespan as the operator and exclude it from R1/R2 evaluation.
Both are deferred as changes outside current scope.

---

## 8. R4/R5 — Duration-Rule Frame Reporting

R4 (Exit Blockage) and R5 (Loitering) both report the **last** frame where
their duration condition was confirmed, not the frame where the threshold
was first crossed. Concretely: `engine.py`'s `_run_r4` accumulates a
track's continuous per-frame presence in a zone across the whole clip, and
only emits the `RuleEvent` once, using the *final* observed frame
(`obs[-1]`) as `frame`/`timestamp` — the same pattern R5 uses at its `left`
zone-transition frame. This is internally consistent between R4 and R5, but
different from R2, which fires the moment its consecutive-frame counter
first crosses `near_miss_min_consecutive_frames`, mid-stream.

**Why this matters in practice:** for a condition that already existed
before the clip started recording — e.g. an exit blocked by an object
present from frame 0 — the reported "frame" is wherever the clip happens to
end (or the track's last observation), not the moment it became a
violation. Jumping to that frame in the demo shows a frame that looks
visually identical to frame 0, which reads as confusing or broken even
though the rule fired correctly. Confirmed on `outdoor_exit_test`: the
barrier is present for all 126 frames, and R4 reports `frame=125` — the
clip's last frame — with nothing visually different there from frame 0.

**Decision:** left as-is. Changing which frame gets reported would change
evaluated semantics for two rules simultaneously and touches existing
tests/expectations; not pursued in this pass. Documented here so it isn't
mistaken for a bug the next time it's noticed in a demo.

---

## 9. Deferred Enhancements (Out of Current Scope)

Two extensions were deliberately not pursued, each for a specific reason
rather than lack of time:

- **NVIDIA NIM as an alternate narration provider** (`src/narration/nim_client.py`,
  currently an empty stub). Gemini (`gemini_client.py`) already satisfies the
  narration layer's requirements — constrained rephrasing of rule-engine
  output, live-verified against a real API key. Adding a second provider
  behind the same interface would be straightforward but wasn't necessary
  for this project's scope.
- **R2 object-size-relative distance normalization** — see §5's "Recommended
  next step." Deferred after the resolution-normalization fix already
  closed the main gap (thresholds silently never firing on 4K footage);
  the remaining camera-framing/zoom sensitivity is a smaller, separate
  problem not required for the two primary demo clips to work correctly.

---

*Last updated: 2026-07-14, following the documentation cleanup pass (real-time
scope note, R4/R5 frame-reporting behavior, and deferred-enhancements list).*