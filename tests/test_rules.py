"""
test_rules.py — regression tests for src/rules/engine.py

Covers two behaviors that were silently wrong until fixed:
  - R1 fired restricted-zone-intrusion events for ghost/churn tracks
    (tracker-ID fragments alive fewer than min_track_frames), because it
    read raw zone_transitions instead of going through the same
    "valid track" filter the world graph already applies.
  - R2 reset its consecutive-proximity counter on any single missed
    frame, so one sustained near-miss with a 1-2 frame tracker/detector
    flicker in the middle got reported as several separate events.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.world_graph import GraphBuilder, WorldGraphConfig, WorldQuery
from src.rules import RuleEngine, RuleEngineConfig


ZONE_NAME = "restricted_zone"
ZONES = {
    "zones": [
        {"name": ZONE_NAME, "type": "restricted", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
    ],
    "frame_width": 100,
    "frame_height": 100,
}


def make_track(class_name: str, first_frame: int, last_frame: int) -> dict:
    """Minimal track record satisfying GraphBuilder.REQUIRED_TRACK_KEYS."""
    return {
        "class_name": class_name,
        "status": "alive",
        "first_seen_frame": first_frame,
        "last_seen_frame": last_frame,
        "age_frames": last_frame - first_frame + 1,
    }


def make_engine(tmp_path: Path, tracks: dict, frames: dict, zone_transitions: list,
                 config: RuleEngineConfig = None) -> RuleEngine:
    tracks_path = tmp_path / "tracks.json"
    spatial_path = tmp_path / "spatial_events.json"
    zones_path = tmp_path / "zones.json"

    tracks_path.write_text(json.dumps(tracks))
    spatial_path.write_text(json.dumps({"frames": frames, "zone_transitions": zone_transitions}))
    zones_path.write_text(json.dumps(ZONES))

    builder = GraphBuilder(
        tracks_path=str(tracks_path),
        spatial_events_path=str(spatial_path),
        config=WorldGraphConfig(min_track_frames=15),
    )
    query = WorldQuery(builder)
    return RuleEngine(query=query, zones_path=str(zones_path), config=config or RuleEngineConfig())


# --- R1: ghost/churn tracks must not fire a restricted-zone intrusion ---

def test_r1_ignores_ghost_track_below_min_track_frames(tmp_path):
    tracks = {
        "1": make_track("person", 0, 5),  # 6 frames, below min_track_frames=15 -> ghost
    }
    zone_transitions = [
        {"frame": 0, "timestamp": 0.0, "track_id": 1, "class_name": "person",
         "zone_name": ZONE_NAME, "event_type": "entered"},
    ]
    engine = make_engine(tmp_path, tracks, frames={}, zone_transitions=zone_transitions)

    events = engine.run()

    r1_events = [e for e in events if e.rule_id == "R1_RESTRICTED_ZONE_INTRUSION"]
    assert r1_events == []


def test_r1_fires_for_track_meeting_min_track_frames(tmp_path):
    tracks = {
        "1": make_track("person", 0, 20),  # 21 frames, meets min_track_frames=15
    }
    zone_transitions = [
        {"frame": 0, "timestamp": 0.0, "track_id": 1, "class_name": "person",
         "zone_name": ZONE_NAME, "event_type": "entered"},
    ]
    engine = make_engine(tmp_path, tracks, frames={}, zone_transitions=zone_transitions)

    events = engine.run()

    r1_events = [e for e in events if e.rule_id == "R1_RESTRICTED_ZONE_INTRUSION"]
    assert len(r1_events) == 1
    assert r1_events[0].track_ids == [1]


# --- R2: gap tolerance for the sustained-proximity counter ---

def _person_forklift_frame(frame_idx: int, distance_px: float) -> list:
    """One frame's worth of spatial 'frames' events: person 1 inside the
    restricted zone, forklift 2 nearby at the given distance."""
    return [
        {
            "track_id": 1, "class_name": "person", "timestamp": frame_idx / 25.0,
            "centroid": [10, 10], "zones_inside": [ZONE_NAME],
            "nearby_tracks": [{"track_id": 2, "class_name": "forklift", "distance_px": distance_px}],
        },
        {
            "track_id": 2, "class_name": "forklift", "timestamp": frame_idx / 25.0,
            "centroid": [20, 20], "zones_inside": [], "nearby_tracks": [],
        },
    ]


def test_r2_tolerates_brief_gap_as_one_sustained_event(tmp_path):
    tracks = {
        "1": make_track("person", 0, 29),
        "2": make_track("forklift", 0, 29),
    }
    near_px = 50.0
    far_px = 9999.0
    # Two 15-frame near stretches separated by a single flicker frame.
    # Each stretch is independently >= near_miss_min_consecutive_frames
    # (10), so under the old reset-on-any-gap logic this fires TWICE
    # (once per stretch) even though it's one continuous interaction with
    # a 1-frame tracker blip in the middle.
    frames = {}
    for f in range(15):
        frames[str(f)] = _person_forklift_frame(f, near_px)
    frames["15"] = _person_forklift_frame(15, far_px)
    for f in range(16, 30):
        frames[str(f)] = _person_forklift_frame(f, near_px)

    config = RuleEngineConfig(
        near_miss_distance_px=near_px,
        near_miss_min_consecutive_frames=10,
        near_miss_gap_tolerance_frames=3,
    )
    engine = make_engine(tmp_path, tracks, frames=frames, zone_transitions=[], config=config)

    events = engine.run()

    r2_events = [e for e in events if e.rule_id == "R2_FORKLIFT_PEDESTRIAN_NEAR_MISS"]
    assert len(r2_events) == 1, "a single 1-frame gap should not fragment one sustained near-miss into two events"


def test_r2_does_not_bridge_gap_longer_than_tolerance(tmp_path):
    tracks = {
        "1": make_track("person", 0, 39),
        "2": make_track("forklift", 0, 39),
    }
    near_px = 50.0
    far_px = 9999.0
    # 15 near frames, a genuine 10-frame separation, then 15 more near frames
    frames = {}
    for f in range(15):
        frames[str(f)] = _person_forklift_frame(f, near_px)
    for f in range(15, 25):
        frames[str(f)] = _person_forklift_frame(f, far_px)
    for f in range(25, 40):
        frames[str(f)] = _person_forklift_frame(f, near_px)

    config = RuleEngineConfig(
        near_miss_distance_px=near_px,
        near_miss_min_consecutive_frames=10,
        near_miss_gap_tolerance_frames=3,
    )
    engine = make_engine(tmp_path, tracks, frames=frames, zone_transitions=[], config=config)

    events = engine.run()

    r2_events = [e for e in events if e.rule_id == "R2_FORKLIFT_PEDESTRIAN_NEAR_MISS"]
    assert len(r2_events) == 2, "a genuine 10-frame separation (beyond gap tolerance) should be two episodes"

