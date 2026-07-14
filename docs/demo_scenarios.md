# ChainSight — Demo Scenarios

This is the walkthrough script for the two clips the demo is actually tuned around, plus notes
on the two validation-only clips that also happen to be browsable. See
[`architecture.md`](architecture.md) for how the pipeline works and
[`scope_and_limitations.md`](scope_and_limitations.md) for the caveats referenced below.

---

## Primary demo clip 1: `nearmiss` — R1 fires, R2 deliberately doesn't

**Source:** `forklift_pedestrian_nearmiss.mp4` (4K warehouse aisle, forklift + two pedestrians).

**What fires:** R1 (Restricted Zone Intrusion), twice — both at frame 0:

| Frame | Time | Trigger |
|---|---|---|
| 0 | 0.0s | Person 2 entered restricted zone `restricted_forklift_pedestrian_aisle` |
| 0 | 0.0s | Person 3 entered restricted zone `restricted_forklift_pedestrian_aisle` |

**What to say walking through it:** Pick `nearmiss` in the sidebar, let the rule-event table
show both R1 events, then use the frame scrubber (or the "Jump to frame"/"Jump to time"
inputs) to move through the clip and show the live bbox/zone overlay tracking the forklift and
both people as they move through the restricted aisle. The Gemini-generated summary at the
bottom reads the two events back in plain English — point out that it's *rephrasing* the rule
engine's output, not adding its own judgment (no severity language beyond what R1 already
assigned).

**Why R2 (Forklift-Pedestrian Near-Miss) does not fire here — and why that's the point, not a
bug:** this is a wide establishing shot where the forklift occupies roughly 1/4 of frame width.
The closest the forklift and a tracked pedestrian ever get, across all 525 frames, is
~1060-1270px — well beyond even the resolution-corrected proximity threshold. Verified
visually (see `data/staged_clips/extracted_frames/nearmiss_wide/frame_00410.jpg`, the clip's
closest tracked approach — the two are still clearly on opposite sides of an open floor).
**This is worth demonstrating deliberately**: it shows the system doesn't force a near-miss
signal just because two relevant classes share a frame — R2 has a real, specific proximity
condition, and this clip's camera framing genuinely doesn't produce one. Full details in
`scope_and_limitations.md` §5.

---

## Primary demo clip 2: `blocked_exit` — zero events fire, and that's the demonstration

**Source:** `blocked_exit.mp4` (indoor exit door, roller shutter, boxes stacked nearby).

**What fires:** nothing. `rule_events_blocked_exit.json` is an empty list.

**What to say walking through it:** this clip is staged so that an object occupies the exit
zone for **2.96 seconds** — just under R4's 3.0-second `min_exit_block_seconds` threshold. The
demo will show the frame scrubber tracking the object sitting in the exit zone, the zone
overlay correctly drawn, but the rule-event table stays empty. This is the natural place to
talk about threshold calibration as a design decision: R4 requires *sustained* blockage, not a
single frame of overlap, specifically to avoid flagging someone briefly walking through an
exit zone as a "blockage." 2.96s vs. 3.0s is a genuine, honest borderline case — not a bug, and
deliberately not "fixed" by lowering the threshold just to make this specific clip fire (same
don't-force-a-signal principle as R2 above).

If you want to show R4 **actually firing**, that's the validation clip `outdoor_exit_test`
below, with a caveat.

---

## Validation-only clips: `aisle_test` and `outdoor_exit_test`

These exist to stress-test the pipeline against real footage it wasn't tuned on — not to look
good in a demo. Both are browsable in the Streamlit app, but neither should be used as a
primary walkthrough clip.

### `aisle_test` (`forklift_person_aisle.mp4`)

R1 fires once (frame 490), but **the event is not a genuine safety signal**: the "person" track
that triggered it has a 1.0 bbox-overlap ratio with the forklift for its entire 18-frame
lifespan — it's the driver visible through an open cab, not a separate pedestrian. On top of
that, the source camera **pans** to follow the forklift down the aisle, so any zone polygon
drawn on one frame stops being meaningful once the framing changes. Two independent,
disqualifying issues, neither fixable by threshold tuning. Full detail in
`scope_and_limitations.md` §7.

**Why not a demo clip:** showing this would require explaining two limitations before getting
to anything the pipeline does correctly — it's useful evidence that the pipeline correctly
tracks a forklift and person through a busy aisle, but not a clean R1/R2 story.

### `outdoor_exit_test` (`_reserve_blocked_exit_outdoor.mp4`)

R4 fires correctly — an object occupies the exit zone continuously for 5.2s against the 3.0s
threshold, on a genuinely fixed camera (unlike `aisle_test`). **But the object is
misclassified**: it's an orange traffic barrier, detected as `small_load_carrier` because the
17-class detector has no barrier/cone class to map it to. The rule's mechanical trigger is
correct; the narration text naming the object is not.

**Why not a demo clip:** the rule logic is a clean positive example, but the narration output
would actively mislead anyone reading it about what's actually blocking the exit — not
something to put in front of a reviewer without a caveat attached every time.

---

## Note on this document vs. the original request

`blocked_exit` was originally described as the R4-firing example. Checking the actual
`rule_events_blocked_exit.json` output while writing this doc, that's not accurate — R4 does
not fire in that clip (2.96s vs. 3.0s, see above). `outdoor_exit_test` is where R4 genuinely
fires, but it's validation-only due to the misclassification caveat. This doc reflects the
real data rather than the original framing.
