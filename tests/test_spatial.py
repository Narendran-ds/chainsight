"""
test_spatial.py — regression tests for the ChainSight spatial layer
(src/spatial/{overlap,zones,analyzer}.py).

The rule engine's test suite (tests/test_rules.py) leans heavily on
spatial_events.json's schema without verifying the layer that produces
it. These tests cover that layer directly: zone containment geometry
(overlap.py), zone-file loading/validation (zones.py), and the
per-frame orchestration that turns tracks.json into zone-transition and
proximity events (analyzer.py).
"""

import json
import sys
from pathlib import Path

import pytest
from shapely.geometry import Polygon

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.spatial import Zone, SpatialConfig, load_zones, zones_containing_bbox, SpatialAnalyzer


SQUARE_ZONE = Zone(name="z", zone_type="restricted", polygon=Polygon([[0, 0], [100, 0], [100, 100], [0, 100]]))


# =====================================================================
# overlap.py — zones_containing_bbox
# =====================================================================

def test_bbox_fully_inside_zone_is_contained():
    bbox = [10, 10, 50, 50]
    result = zones_containing_bbox(bbox, [SQUARE_ZONE], SpatialConfig())
    assert result == ["z"]


def test_bbox_fully_outside_zone_is_not_contained():
    bbox = [200, 200, 250, 250]
    result = zones_containing_bbox(bbox, [SQUARE_ZONE], SpatialConfig())
    assert result == []


def test_bbox_overlap_ratio_below_threshold_excluded():
    # overlap region [0,0]-[10,10] = area 100; bbox area = 100*100 = 10000
    # -> ratio 0.01, below the default 0.1 threshold
    bbox = [-90, -90, 10, 10]
    result = zones_containing_bbox(bbox, [SQUARE_ZONE], SpatialConfig(zone_overlap_threshold=0.1))
    assert result == []


def test_bbox_overlap_ratio_above_threshold_included():
    # same geometry as above, but with a threshold low enough to include it
    bbox = [-90, -90, 10, 10]
    result = zones_containing_bbox(bbox, [SQUARE_ZONE], SpatialConfig(zone_overlap_threshold=0.005))
    assert result == ["z"]


def test_centroid_only_mode_ignores_bbox_overlap():
    # bbox overlaps the zone by ~18% (would pass the default bbox-mode
    # threshold), but its centroid falls outside the zone polygon, so
    # centroid-only mode must exclude it.
    bbox = [-80, -80, 60, 60]
    config = SpatialConfig(use_bbox_for_zones=False)
    result = zones_containing_bbox(bbox, [SQUARE_ZONE], config)
    assert result == []


def test_zero_area_bbox_is_never_contained():
    bbox = [10, 10, 10, 10]
    result = zones_containing_bbox(bbox, [SQUARE_ZONE], SpatialConfig())
    assert result == []


def test_no_zones_returns_empty():
    result = zones_containing_bbox([10, 10, 50, 50], [], SpatialConfig())
    assert result == []


# =====================================================================
# zones.py — Zone.from_dict / load_zones
# =====================================================================

def write_zones_file(tmp_path: Path, zones: list) -> Path:
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": zones, "frame_width": 100, "frame_height": 100}))
    return path


def test_load_zones_valid_file(tmp_path):
    path = write_zones_file(tmp_path, [
        {"name": "restricted_1", "type": "restricted", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
    ])
    zones = load_zones(str(path))
    assert len(zones) == 1
    assert zones[0].name == "restricted_1"
    assert zones[0].zone_type == "restricted"


def test_load_zones_skips_malformed_zone_but_keeps_valid_ones(tmp_path):
    path = write_zones_file(tmp_path, [
        {"name": "good_zone", "type": "exit", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"name": "missing_type", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},  # no "type" key
    ])
    zones = load_zones(str(path))
    assert [z.name for z in zones] == ["good_zone"]


def test_load_zones_skips_zero_area_polygon(tmp_path):
    path = write_zones_file(tmp_path, [
        {"name": "degenerate", "type": "restricted", "polygon": [[0, 0], [0, 0], [0, 0]]},
    ])
    zones = load_zones(str(path))
    assert zones == []


def test_load_zones_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_zones(str(tmp_path / "does_not_exist.json"))


def test_load_zones_invalid_json_raises(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError):
        load_zones(str(path))


# =====================================================================
# analyzer.py — SpatialAnalyzer.analyze
# =====================================================================

def write_tracks_file(tmp_path: Path, tracks: dict) -> Path:
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(tracks))
    return path


def obs(frame, bbox, centroid, timestamp=None):
    return {"frame": frame, "timestamp": timestamp if timestamp is not None else frame / 25.0,
            "bbox": bbox, "centroid": centroid}


def test_zone_transitions_entered_and_left(tmp_path):
    # person track: outside zone (frame 0), inside (frames 1-2), outside again (frame 3)
    tracks = {
        "1": {
            "class_name": "person",
            "history": [
                obs(0, [200, 200, 220, 220], [210, 210]),
                obs(1, [10, 10, 50, 50], [30, 30]),
                obs(2, [10, 10, 50, 50], [30, 30]),
                obs(3, [200, 200, 220, 220], [210, 210]),
            ],
        }
    }
    zones_path = write_zones_file(tmp_path, [
        {"name": "restricted_zone", "type": "restricted", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
    ])
    tracks_path = write_tracks_file(tmp_path, tracks)

    analyzer = SpatialAnalyzer(zones_path=str(zones_path))
    events_by_frame, transitions = analyzer.analyze(str(tracks_path))

    entered = [t for t in transitions if t.event_type == "entered"]
    left = [t for t in transitions if t.event_type == "left"]
    assert len(entered) == 1 and entered[0].frame == 1 and entered[0].zone_name == "restricted_zone"
    assert len(left) == 1 and left[0].frame == 3 and left[0].zone_name == "restricted_zone"
    assert events_by_frame[1][0].zones_inside == ["restricted_zone"]
    assert events_by_frame[0][0].zones_inside == []


def test_proximity_pairs_within_threshold(tmp_path):
    # track 1 and 2 are 5px apart (within threshold 10), track 3 is far away
    tracks = {
        "1": {"class_name": "person", "history": [obs(0, [0, 0, 10, 10], [0, 0])]},
        "2": {"class_name": "forklift", "history": [obs(0, [3, 4, 13, 14], [3, 4])]},
        "3": {"class_name": "person", "history": [obs(0, [500, 500, 510, 510], [500, 500])]},
    }
    zones_path = write_zones_file(tmp_path, [])
    tracks_path = write_tracks_file(tmp_path, tracks)

    analyzer = SpatialAnalyzer(zones_path=str(zones_path), config=SpatialConfig(proximity_threshold_px=10))
    events_by_frame, _ = analyzer.analyze(str(tracks_path))

    events_by_track = {e.track_id: e for e in events_by_frame[0]}
    assert events_by_track[3].nearby_tracks == []

    near_1 = events_by_track[1].nearby_tracks
    assert len(near_1) == 1
    assert near_1[0]["track_id"] == 2
    assert near_1[0]["distance_px"] == 5.0  # 3-4-5 triangle

    # symmetric: track 2 must also see track 1 as nearby
    near_2 = events_by_track[2].nearby_tracks
    assert len(near_2) == 1 and near_2[0]["track_id"] == 1


def test_proximity_excludes_pairs_beyond_threshold(tmp_path):
    tracks = {
        "1": {"class_name": "person", "history": [obs(0, [0, 0, 10, 10], [0, 0])]},
        "2": {"class_name": "forklift", "history": [obs(0, [3, 4, 13, 14], [3, 4])]},  # distance 5
    }
    zones_path = write_zones_file(tmp_path, [])
    tracks_path = write_tracks_file(tmp_path, tracks)

    analyzer = SpatialAnalyzer(zones_path=str(zones_path), config=SpatialConfig(proximity_threshold_px=4))
    events_by_frame, _ = analyzer.analyze(str(tracks_path))

    for e in events_by_frame[0]:
        assert e.nearby_tracks == []


def test_analyze_skips_malformed_track_without_crashing(tmp_path):
    tracks = {
        "1": {"class_name": "person", "history": [obs(0, [0, 0, 10, 10], [0, 0])]},
        "2": {"class_name": "forklift"},  # missing "history" -> malformed, should be skipped
    }
    zones_path = write_zones_file(tmp_path, [])
    tracks_path = write_tracks_file(tmp_path, tracks)

    analyzer = SpatialAnalyzer(zones_path=str(zones_path))
    events_by_frame, _ = analyzer.analyze(str(tracks_path))

    assert len(events_by_frame[0]) == 1
    assert events_by_frame[0][0].track_id == 1


def test_analyze_missing_tracks_file_raises(tmp_path):
    zones_path = write_zones_file(tmp_path, [])
    analyzer = SpatialAnalyzer(zones_path=str(zones_path))
    with pytest.raises(FileNotFoundError):
        analyzer.analyze(str(tmp_path / "does_not_exist.json"))
