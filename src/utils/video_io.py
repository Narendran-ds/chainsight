"""
video_io.py — shared video-resolution scaling helper.
Used by the spatial/rules layers to keep pixel-distance thresholds like
proximity_threshold_px / near_miss_distance_px resolution-independent.

Why this exists: those thresholds were originally tuned against ~1080p
footage. Run the same pipeline on a 4K clip and a physically-close
forklift/person pair can sit 1000+px apart in raw pixel terms — the
threshold silently never fires. resolution_scale() converts a threshold
calibrated at REFERENCE_FRAME_WIDTH into the equivalent threshold for
whatever resolution the current clip actually is. Frame width itself comes
from tracker.py, which captures it once from the source video and stores it
in tracks.json's "_meta" key.
"""

from typing import Optional

# Pixel thresholds throughout spatial/rules configs (proximity_threshold_px,
# near_miss_distance_px, ...) are calibrated against this frame width.
REFERENCE_FRAME_WIDTH = 1920.0


def resolution_scale(frame_width: Optional[float], reference_width: float = REFERENCE_FRAME_WIDTH) -> float:
    """
    Factor to convert a px threshold calibrated at reference_width into the
    equivalent threshold for frame_width. Falls back to 1.0 (no scaling) when
    frame_width is unknown/zero, so tracks.json produced before this field
    existed still behaves exactly as before.
    """
    if not frame_width:
        return 1.0
    return frame_width / reference_width
