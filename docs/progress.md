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
   `test_narration.py`, `test_visualization.py`, `test_run_data.py` (55 tests total, all
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

---

## Not started (no fixed priority — pick up as needed)

- `src/narration/nim_client.py` — alternate NVIDIA NIM provider. Explicitly deferred; Gemini-only
  was the chosen scope for the narration layer.
- R2 object-size-relative distance normalization — normalizing near-miss distance against
  detected bbox size (not just frame resolution), so it's robust to camera zoom/distance too.
  Explicitly deferred after the resolution-normalization work (see
  `docs/scope_and_limitations.md` §5's "Recommended next step").
- `src/vision/detector.py` — cosmetic split of detection out of `tracker.py`'s combined
  detect+track stage. No functional gap; `tracker.py` already does both.
- `configs/rules_config.yaml`, `configs/zones_config.yaml` — YAML config surface. CLI args +
  dataclass defaults already cover this; would just be ergonomics.
- `scripts/run_finetune.py`, `scripts/annotate_clips.py` — training/data-prep tooling, unrelated
  to the reasoning pipeline being demoable.
- `docs/architecture.md`, `docs/demo_scenarios.md` — deeper portfolio-facing docs (`README.md`
  is now written, covering the essentials of both).

---

*Last updated: 2026-07-14, after building/verifying the Streamlit demo and writing README.md.*
