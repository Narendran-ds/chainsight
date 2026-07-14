"""
test_visualization.py — regression tests for src/utils/visualization.py

Pure pixel-drawing checks: shape/mutation invariants and that known classes
render with their fixed colors (so demo overlays stay visually consistent
across runs).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.visualization import (
    VisualizationConfig,
    blank_canvas,
    draw_tracks_on_frame,
    draw_zones_on_frame,
    _color_for_class,
)


def test_blank_canvas_shape_and_fill():
    canvas = blank_canvas(width=64, height=32, color=(10, 20, 30))
    assert canvas.shape == (32, 64, 3)
    assert tuple(canvas[0, 0]) == (10, 20, 30)
    assert tuple(canvas[-1, -1]) == (10, 20, 30)


def test_draw_tracks_does_not_mutate_input_frame():
    frame = blank_canvas(100, 100)
    original = frame.copy()

    draw_tracks_on_frame(frame, [{"track_id": 1, "class_name": "person", "bbox": [10, 10, 50, 50]}])

    assert np.array_equal(frame, original)


def test_draw_tracks_paints_bbox_border_with_class_color():
    frame = blank_canvas(100, 100, color=(0, 0, 0))
    config = VisualizationConfig(box_thickness=2)

    out = draw_tracks_on_frame(
        frame, [{"track_id": 1, "class_name": "person", "bbox": [10, 10, 50, 50]}], config,
    )

    expected_color = config.track_colors["person"]
    # top-left corner of the box should now be painted with person's color
    assert tuple(out[10, 10]) == expected_color


def test_color_for_class_is_deterministic_and_distinct_for_unknown_classes():
    config = VisualizationConfig()

    color_a = _color_for_class("pallet", config)
    color_b = _color_for_class("pallet", config)
    color_c = _color_for_class("box", config)

    assert color_a == color_b  # deterministic across calls
    assert color_a != color_c  # different unknown classes get different colors


def test_known_classes_use_fixed_colors_not_hash_fallback():
    config = VisualizationConfig()
    assert _color_for_class("person", config) == config.track_colors["person"]
    assert _color_for_class("forklift", config) == config.track_colors["forklift"]


def test_draw_zones_does_not_mutate_input_frame():
    frame = blank_canvas(100, 100)
    original = frame.copy()

    draw_zones_on_frame(frame, [{"name": "aisle", "type": "restricted", "polygon": [[5, 5], [50, 5], [50, 50], [5, 50]]}])

    assert np.array_equal(frame, original)


def test_draw_zones_paints_outline_with_type_color():
    frame = blank_canvas(100, 100, color=(0, 0, 0))
    config = VisualizationConfig(zone_thickness=2)
    zones = [{"name": "aisle", "type": "restricted", "polygon": [[5, 5], [80, 5], [80, 80], [5, 80]]}]

    out = draw_zones_on_frame(frame, zones, config)

    expected_color = config.zone_colors["restricted"]
    # top edge of the polygon should be painted with the restricted-zone color
    assert tuple(out[5, 40]) == expected_color


def test_draw_zones_falls_back_to_default_color_for_unknown_zone_type():
    frame = blank_canvas(100, 100, color=(0, 0, 0))
    config = VisualizationConfig(zone_thickness=2)
    zones = [{"name": "loading_dock", "type": "staging", "polygon": [[5, 5], [80, 5], [80, 80], [5, 80]]}]

    out = draw_zones_on_frame(frame, zones, config)

    assert tuple(out[5, 40]) == config.default_zone_color
