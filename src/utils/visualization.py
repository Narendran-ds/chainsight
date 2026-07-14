"""
visualization.py — ChainSight demo drawing helpers
Pure OpenCV drawing functions: given a frame (as a numpy array) plus already-
resolved track/zone data for that frame, draw bboxes/labels/zone polygons on
top and return the annotated frame. No file I/O, no Streamlit — this module
only knows pixels, so it can be unit-tested without a display or a real
video file.

Pipeline position:
    demo/components/frame_viewer.py resolves "which tracks/zones are active
    at frame N" from tracks.json/zones.json, then calls into THIS module to
    render them. Kept separate from that resolution logic so drawing can be
    tested independently of tracks.json's schema.

Design principle: colors are assigned deterministically (fixed for the two
safety-critical classes the rule engine cares about most — person, forklift
— hash-based for everything else) so the same class always renders the same
color across frames and runs, without needing to hardcode all 17 dataset
classes here.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BGRColor = Tuple[int, int, int]


@dataclass
class VisualizationConfig:
    box_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    zone_thickness: int = 2
    zone_colors: Dict[str, BGRColor] = field(default_factory=lambda: {
        "restricted": (0, 0, 255),   # red (BGR)
        "exit": (0, 200, 0),         # green
    })
    default_zone_color: BGRColor = (0, 200, 200)  # yellow-ish, for zone types not listed above
    track_colors: Dict[str, BGRColor] = field(default_factory=lambda: {
        "person": (0, 165, 255),     # orange
        "forklift": (255, 0, 0),     # blue
    })


def _color_for_class(class_name: str, config: VisualizationConfig) -> BGRColor:
    if class_name in config.track_colors:
        return config.track_colors[class_name]
    # Deterministic fallback so any of the other 17 dataset classes still get
    # a stable (if arbitrary) color, without hardcoding all of them here.
    digest = hashlib.sha1(class_name.encode()).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def blank_canvas(width: int, height: int, color: BGRColor = (40, 40, 40)) -> np.ndarray:
    """A flat-color canvas for when no source video frame is available for a run."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = color
    return canvas


def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: List[dict],
    config: Optional[VisualizationConfig] = None,
) -> np.ndarray:
    """
    tracks: [{"track_id": int, "class_name": str, "bbox": [x1, y1, x2, y2]}, ...]
    Draws each bbox + an "ID {track_id} {class_name}" label. Returns a new
    array (does not mutate the input frame).
    """
    config = config or VisualizationConfig()
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        color = _color_for_class(t["class_name"], config)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, config.box_thickness)
        label = f"ID {t['track_id']} {t['class_name']}"
        cv2.putText(
            out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale, color, config.font_thickness, cv2.LINE_AA,
        )
    return out


def draw_zones_on_frame(
    frame: np.ndarray,
    zones: List[dict],
    config: Optional[VisualizationConfig] = None,
) -> np.ndarray:
    """
    zones: [{"name": str, "type": str, "polygon": [[x, y], ...]}, ...]
    (the same shape zones.json's "zones" list already uses). Draws each
    zone's outline plus its name near the first vertex.
    """
    config = config or VisualizationConfig()
    out = frame.copy()
    for zone in zones:
        color = config.zone_colors.get(zone["type"], config.default_zone_color)
        points = np.array(zone["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [points], isClosed=True, color=color, thickness=config.zone_thickness)
        x, y = zone["polygon"][0]
        cv2.putText(
            out, zone["name"], (int(x), max(0, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            config.font_scale, color, config.font_thickness, cv2.LINE_AA,
        )
    return out
