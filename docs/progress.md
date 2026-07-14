# ChainSight — Progress Tracker

A running checklist of what's done vs. what's left. See `CLAUDE.md` for how each piece works
and `docs/scope_and_limitations.md` for known limitations/caveats on the completed work.

---

## Completed

1. **Vision layer** — YOLOv8 detection + ByteTrack tracking, one combined stage
   (`src/vision/tracker.py`), plus fine-tuning (`src/vision/train.py`). Reference checkpoint:
   `run2_exit_marker`.
2. **Spatial layer** — zone containment + proximity via Shapely (`src/spatial/*`).
3. **World graph layer** — per-frame and summary `nx.Graph`s (`src/world_graph/*`), with a
   shared ghost/churn track filter (`src/utils/track_quality.py`).
4. **Rule engine** — 5 deterministic rules (R1-R5) over `WorldQuery`, each producing an
   explainable `RuleEvent` (`src/rules/*`).
5. **Single-entry pipeline orchestrator** — `src/pipeline.py` (`ChainSightPipeline`) +
   `scripts/run_pipeline.py`, chaining all stages through one command instead of four
   hand-run CLI scripts.
6. **R2 near-miss threshold — resolution normalization** — `proximity_threshold_px` /
   `near_miss_distance_px` were hardcoded pixel values that silently broke on 4K footage.
   Fixed via `src/utils/video_io.py`'s `resolution_scale()`, `tracker.py`'s new `tracks.json`
   `"_meta"` field (frame_width/height), and scaling in `analyzer.py` + `rules/engine.py`.
   Verified symmetrically on both real clips (2.00x up on the 4K near-miss clip, 0.40x down on
   the 768px blocked-exit clip). Remaining known limitation (camera framing/distance, not
   sensor resolution) documented in `docs/scope_and_limitations.md` §5 — deliberately not
   "fixed" further by inflating thresholds.
7. **Narration layer (Gemini-only)** — `src/narration/{gemini_client,narrator,prompt_templates}.py`
   + `scripts/run_narration.py`, wired into the pipeline behind `--narrate` (off by default).
   Per-event narration + one clip-level summary, both constrained to rephrasing the
   deterministic rule engine's output (no causal claims, blame, or severity judgments beyond
   the input). Uses the current `google-genai` SDK (the older `google-generativeai` is fully
   deprecated by Google). **Live-verified** against a real API key — default model is
   `gemini-flash-latest` (not `gemini-2.5-flash`/`gemini-2.0-flash`, which 404'd or hit a
   hard 0-quota 429 on the tested project/key).
8. **Test suite** — `tests/test_spatial.py`, `test_rules.py`, `test_pipeline.py`,
   `test_narration.py`, `test_visualization.py`, `test_run_data.py` (58 tests total, all
   passing). Narration tests are fully mocked — no real API calls in CI/local test runs.
9. **Docs housekeeping** — `CLAUDE.md` corrected in several places where it had gone stale
   (claimed `run_pipeline.py`, `.env.example`, and most test files were empty stubs; they
   weren't). `.env.example` filled in with `GEMINI_API_KEY`. `.gitignore` updated to exclude
   `.env` (it wasn't excluded before — flagged and fixed before any key was written to disk).
10. **Streamlit demo** (scoped and built 2026-07-14 — full design in `CLAUDE.md`'s "Demo
    layer" section):
    - `src/utils/visualization.py` — pure OpenCV drawing helpers (bboxes/track IDs/zone
      polygons), unit-tested independently of any real video.
    - `src/pipeline.py` now writes `outputs/manifest_<run>.json` (video_path/zones_path/
      model_path) at the start of every run — the piece the demo needed to locate a run's
      source video, which nothing previously recorded.
    - `demo/components/run_data.py` — pure run-discovery/data-loading (no Streamlit import),
      fully unit-tested.
    - `demo/components/{clip_picker,video_player,event_timeline,report_card,internals}.py` +
      `demo/app.py` — browses **precomputed runs only** (no live pipeline trigger from the
      UI); frame-by-frame scrubber with overlays drawn **live** (not a replay of `tracker.py`'s
      pre-rendered `--annotated` video); rules + narration front-and-center (clip picker ->
      rule-event timeline -> synced frame scrubber -> narration panel), raw per-stage JSON in
      one secondary "pipeline internals" expander.
    - Used Streamlit's native `st.dataframe` row-selection for the timeline instead of the
      `streamlit-timeline` package pinned in `requirements.txt` (very early/thinly-maintained;
      now commented out there) — this needed bumping the pinned `streamlit` version to 1.50.0
      (1.37.0 predates the dataframe selection API).
    - **Verified end-to-end** with Streamlit's `AppTest` framework (runs the script headlessly,
      no browser needed) against real run data: default/no-manifest run (blank-canvas
      fallback + caption), `nearmiss` run (real video frame + narration summary/per-event
      text), and `blocked_exit` run (0 rule events + "narration not generated" fallback
      message) — zero exceptions across all three paths.
    - **Visually verified** on top of that: launched the app for real (`streamlit run
      demo/app.py`), drove it headlessly with Playwright (installed Chromium via
      `playwright install`), and screenshotted it — confirmed the zone polygon renders
      correctly aligned over the real video frame, not just that the script runs without
      throwing.
11. **`README.md`** — written now that the pipeline + demo are both done and demoable. Covers
    quickstart (the repo ships with committed sample outputs + trained weights, so the demo
    runs immediately after `pip install`), running the pipeline on new video, training a new
    detector, the 5 rules, and project structure.
12. **Two validation runs onboarded** (`aisle_test`, `outdoor_exit_test`) — hand-authored
    `zones_<run>.json` for both (`define_zones.py` is interactive-only, can't run headlessly),
    ran the full pipeline with `--narrate` for each, without touching `nearmiss`/`blocked_exit`.
    Surfaced two genuine, previously-undocumented limitations rather than confirming a clean
    result:
    - `aisle_test`'s camera **pans** to follow the forklift (verified by diffing frames at
      0/390/700) — invalidates zone-based reasoning regardless of thresholds, since a zone
      polygon is only valid for one camera position.
    - The R1 event that fired on `aisle_test` is **not a real safety signal** — the "person"
      track has a 1.0 bbox-overlap with the forklift for its entire 18-frame lifespan, i.e.
      it's the driver visible through the cab, not a pedestrian. Checked and confirmed absent
      from `nearmiss`/`blocked_exit`. Documented in `docs/scope_and_limitations.md` §7.
    - `outdoor_exit_test`'s R4 event fired correctly (genuinely fixed camera, 5.2s continuous
      presence vs. 3.0s threshold), but the blocking object is misclassified as
      `small_load_carrier` — it's actually an orange traffic barrier, and the 17-class detector
      has no barrier/cone class. The rule's mechanical trigger is valid; the narration text
      naming the object is not.
13. **Demo UI pass** (2026-07-14): title now leads with 🏭; sidebar gets a bold "⚙️ Controls"
    heading + divider above the run picker (matching a reference layout); Frame viewer gained
    "Jump to frame" (int) and "Jump to time (s)" (float) number inputs alongside the slider,
    all three kept in sync via a shared `_sync()` helper and per-widget `on_change` callbacks
    (`demo/components/video_player.py`) — `run_data.derive_fps()` recovers fps from existing
    `tracks.json` data for the time<->frame conversion, no schema change needed. Verified with
    `AppTest` in every sync direction (each widget -> the other two, plus the existing
    rule-event-table jump).
14. **Documentation cleanup pass** (2026-07-14):
    - `docs/scope_and_limitations.md` gained 3 new sections: real-time/streaming explicitly
      framed as a non-goal (§1.1 — why, and what genuine real-time support would require
      architecturally), R4/R5's last-frame (not first-crossed-threshold) reporting behavior
      (§8 — explains why a duration-rule's frame can look identical to frame 0 in the demo),
      and a Deferred Enhancements list (§9 — NIM provider, R2 object-size normalization).
      §7's operator-vs-pedestrian mitigation path was sharpened to name a `forklift_operator`
      class specifically.
    - `docs/architecture.md` written: narrative walkthrough of the 5-stage pipeline with an
      ASCII data-flow diagram, the config-dataclass pattern, the single-learned-model
      principle and why it matters, and a table of per-run output artifacts.
    - `docs/demo_scenarios.md` written — and in writing it, found the original assumption that
      `blocked_exit` demonstrates R4 firing was wrong: `rule_events_blocked_exit.json` is an
      empty list (2.96s vs. 3.0s threshold, a real borderline case). Documented `blocked_exit`
      accurately as the "zero events, and that's the point" demonstration instead, with
      `outdoor_exit_test` noted as where R4 actually fires (validation-only, due to the
      barrier-misclassification caveat).
    - Audited all 0-byte files/dirs in the repo and compared `.env`/`.env.example` — see the
      conversation this ran in for the full per-file table; nothing was deleted, pending
      explicit confirmation. Also surfaced a stray locked git worktree
      (`.claude/worktrees/docs-cleanup/`, checked out at an old commit) unrelated to any
      action taken this session — flagged, not touched.

---

## Not started (no fixed priority — pick up as needed)

- `src/narration/nim_client.py` — alternate NVIDIA NIM provider. Explicitly deferred; Gemini-only
  was the chosen scope for the narration layer.
- R2 object-size-relative distance normalization — normalizing near-miss distance against
  detected bbox size (not just frame resolution), so it's robust to camera zoom/distance too.
  Explicitly deferred after the resolution-normalization work (see
  `docs/scope_and_limitations.md` §5's "Recommended next step").
- Vehicle operator vs. pedestrian distinction for R1/R2 — either a dedicated `operator` class
  (new labeled data) or a heuristic filter excluding a `person` track that stays highly
  bbox-overlapped with a `forklift` track for its whole lifespan. Surfaced by the `aisle_test`
  validation run; deferred as a larger design change (see `docs/scope_and_limitations.md` §7).
- `src/vision/detector.py` — cosmetic split of detection out of `tracker.py`'s combined
  detect+track stage. No functional gap; `tracker.py` already does both.
- `configs/rules_config.yaml`, `configs/zones_config.yaml` — YAML config surface. CLI args +
  dataclass defaults already cover this; would just be ergonomics.
- `scripts/run_finetune.py`, `scripts/annotate_clips.py` — training/data-prep tooling, unrelated
  to the reasoning pipeline being demoable.

---

*Last updated: 2026-07-14, after the documentation cleanup pass (item 14 below).*
