"""
video_player.py — ChainSight demo, frame-by-frame scrubber
Renders a frame-index slider (plus a "jump to frame" and "jump to time"
number input, kept in sync with it) and the corresponding frame with
bboxes/zone polygons drawn live via src/utils/visualization.py (never a
replay of tracker.py's pre-rendered --annotated video — see CLAUDE.md's
Demo layer section for why). Falls back to a blank canvas if the run has
no manifest (video_path unknown) or the source video file can't be opened
— a run assembled from the individual scripts/run_*.py calls may never
have had a manifest written.

st.session_state["current_frame"] is the single source of truth for the
selected frame, shared with event_timeline.py (clicking a rule event sets
it before this module's widgets are instantiated in the same script run).
The slider/frame-input/time-input each have their own widget key
(Streamlit doesn't allow two widgets to share one key); on_change
callbacks below keep all three, plus "current_frame", in lockstep — see
the module docstring in run_data.py's derive_fps() for how fps (needed
for the time<->frame conversion) is recovered without a schema change.
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


def _sync(frame: int, min_frame: int, max_frame: int, fps: float) -> None:
    """Canonical update: clamps to range, writes "current_frame" plus all
    three widgets' own keys, so whichever widget didn't trigger this still
    picks up the new value when it's instantiated on the next run."""
    frame = max(min_frame, min(int(round(frame)), max_frame))
    st.session_state["current_frame"] = frame
    st.session_state["frame_slider"] = frame
    st.session_state["frame_number_input"] = frame
    st.session_state["time_number_input"] = round(frame / fps, 2) if fps else 0.0


def render(
    frame_index: Dict[int, List[dict]],
    zones: List[dict],
    manifest: Optional[dict],
    frame_meta: Dict[str, Optional[int]],
    min_frame: int,
    max_frame: int,
    fps: float,
) -> None:
    st.subheader("Frame viewer")

    st.session_state.setdefault("current_frame", min_frame)
    # Re-propagate to each widget's own key before instantiating them, so an
    # external jump (event_timeline.py setting "current_frame" directly from
    # a rule-event row click) reaches the slider and both number inputs too.
    _sync(st.session_state["current_frame"], min_frame, max_frame, fps)

    def _on_slider():
        _sync(st.session_state["frame_slider"], min_frame, max_frame, fps)

    def _on_frame_input():
        _sync(st.session_state["frame_number_input"], min_frame, max_frame, fps)

    def _on_time_input():
        _sync(st.session_state["time_number_input"] * fps, min_frame, max_frame, fps)

    st.slider(
        "Frame", min_value=min_frame, max_value=max(max_frame, min_frame),
        key="frame_slider", on_change=_on_slider,
    )

    jump_col1, jump_col2 = st.columns(2)
    with jump_col1:
        st.number_input(
            "Jump to frame", min_value=min_frame, max_value=max(max_frame, min_frame),
            step=1, key="frame_number_input", on_change=_on_frame_input,
        )
    with jump_col2:
        st.number_input(
            "Jump to time (s)", min_value=0.0, max_value=round(max(max_frame, min_frame) / fps, 2),
            step=0.1, format="%.2f", key="time_number_input", on_change=_on_time_input,
        )

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
