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

## 4. Correlation vs. Causation

ChainSight's rule engine is explicitly designed to describe **observed
spatial-temporal patterns** (e.g. "person entered restricted zone while
forklift was active nearby") rather than assert causal claims about why an
incident occurred. This distinction is deliberate and should be preserved in
any narration or demo framing — the system flags patterns worth human review,
it does not determine fault or root cause.

---

*Last updated: 2026-07-10, following run2_exit_marker evaluation.*