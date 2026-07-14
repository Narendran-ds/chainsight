"""
video_player.py — ChainSight demo, frame-by-frame scrubber
Renders a frame-index slider and the corresponding frame with bboxes/zone
polygons drawn live via src/utils/visualization.py (never a replay of
tracker.py's pre-rendered --annotated video — see CLAUDE.md's Demo layer
section for why). Falls back to a blank canvas if the run has no manifest
(video_path unknown) or the source video file can't be opened — a run
assembled from the individual scripts/run_*.py calls may never have had a
manifest written.

st.session_state["current_frame"] is the single source of truth for the
selected frame, shared with event_timeline.py (clicking a rule event
updates it before this slider is instantiated in the same script run).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.visualization import blank_canvas, draw_tracks_on_frame, draw_zones_on_frame  # noqa: E402


@st.cache_resource(show_spinner=False)
def _open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    return cap if cap.isOpened() else None


def _read_frame(video_path: Optional[str], frame_idx: int, width: int, height: int):
    """Returns a BGR frame (matching visualization.py's BGR color tuples) —
    either the real video frame, or a blank canvas if unavailable."""
    if video_path and Path(video_path).exists():
        cap = _open_video(video_path)
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok:
                return frame
    return blank_canvas(width, height)


def render(
    frame_index: Dict[int, List[dict]],
    zones: List[dict],
    manifest: Optional[dict],
    frame_meta: Dict[str, Optional[int]],
    min_frame: int,
    max_frame: int,
) -> None:
    st.subheader("Frame viewer")

    st.session_state.setdefault("current_frame", min_frame)
    # Clamp in case a previous run's selection (e.g. from event_timeline.py)
    # is out of range for a differently-sized run.
    st.session_state["current_frame"] = max(min_frame, min(st.session_state["current_frame"], max_frame))

    st.slider("Frame", min_value=min_frame, max_value=max(max_frame, min_frame), key="current_frame")
    frame_idx = st.session_state["current_frame"]

    width = frame_meta.get("frame_width") or 1280
    height = frame_meta.get("frame_height") or 720
    video_path = manifest.get("video_path") if manifest else None

    frame = _read_frame(video_path, frame_idx, width, height)
    frame = draw_zones_on_frame(frame, zones)
    frame = draw_tracks_on_frame(frame, frame_index.get(frame_idx, []))

    if not (video_path and Path(video_path).exists()):
        st.caption("Source video not found for this run — showing zone/track overlay only.")

    st.image(frame, caption=f"Frame {frame_idx}", channels="BGR")
