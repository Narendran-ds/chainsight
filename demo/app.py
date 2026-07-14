"""
app.py — ChainSight Streamlit demo, entry point
Usage:
    streamlit run demo/app.py

Browses precomputed pipeline runs only — it never triggers a live
tracker/spatial/rules/narration run from the UI (see CLAUDE.md's Demo
layer section for why: the tracker stage is slow/GPU-bound, and a demo
should not risk a live YOLO pass failing on stage).

Layout: clip picker (sidebar) -> rule-event timeline -> frame scrubber
(synced to the selected event's frame) -> narration panel -> a collapsed
"pipeline internals" expander for raw per-stage JSON. Rules + narration are
front-and-center per the demo's scoped design; raw JSON is supporting detail.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))  # for `components.*`
sys.path.append(str(Path(__file__).resolve().parent.parent))  # for `src.*`

from components import clip_picker, event_timeline, video_player, report_card, internals, run_data

OUTPUT_DIR = "outputs"

st.set_page_config(page_title="ChainSight", layout="wide")
st.title("🏭 ChainSight — Warehouse Safety Review")

run_name = clip_picker.render(OUTPUT_DIR)

# Reset the frame scrubber when switching runs — a frame index from a
# previous, differently-sized run isn't meaningful here.
if st.session_state.get("_active_run") != run_name:
    st.session_state["_active_run"] = run_name
    st.session_state.pop("current_frame", None)

paths = run_data.resolve_run_paths(OUTPUT_DIR, run_name)
manifest = run_data.load_json(paths["manifest"])
tracks = run_data.load_tracks(paths["tracks"])
frame_meta = run_data.load_frame_meta(paths["tracks"])
zones_path = Path(manifest["zones_path"]) if manifest and manifest.get("zones_path") else None
zones = run_data.load_zones(zones_path) if zones_path and zones_path.exists() else []
spatial_events = run_data.load_json(paths["spatial_events"])
world_graph_summary = run_data.load_json(paths["world_graph_summary"])
rule_events = run_data.load_rule_events(paths["rule_events"])
narration = run_data.load_json(paths["narration"])

frame_index = run_data.index_tracks_by_frame(tracks)
min_frame, max_frame = run_data.frame_range(tracks)
fps = run_data.derive_fps(tracks)

event_timeline.render(rule_events)
video_player.render(frame_index, zones, manifest, frame_meta, min_frame, max_frame, fps)
report_card.render(narration, run_name)
internals.render(tracks, spatial_events, world_graph_summary)
