"""
test_pipeline.py — regression tests for src/pipeline.py (the single-entry
orchestrator).

The tracker stage (ChainSightTracker) needs real YOLO weights and a real
video, which unit tests shouldn't depend on. So these tests mock only
ChainSightTracker (patched where pipeline.py looks it up) and let every
downstream stage — SpatialAnalyzer, GraphBuilder, RuleEngine — run for
real against synthetic track data, on real temp-file zones/output paths.
This exercises the actual stage-wiring and file hand-off, not just
config plumbing.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import ChainSightPipeline, PipelineConfig
from src.spatial import SpatialConfig
from src.rules import RuleEngineConfig


ZONE_NAME = "restricted_zone"


def write_zones_file(tmp_path: Path) -> Path:
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({
        "zones": [
            {"name": ZONE_NAME, "type": "restricted", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
        ],
        "frame_width": 100,
        "frame_height": 100,
    }))
    return path


def make_history(n_frames: int, bbox, centroid):
    return [
        {"frame": f, "timestamp": f / 25.0, "bbox": bbox, "centroid": centroid, "conf": 0.9, "vx": 0.0, "vy": 0.0, "speed": 0.0}
        for f in range(n_frames)
    ]


PERSON_INSIDE_ZONE_TRACKS = {
    "1": {
        "class_id": 0,
        "class_name": "person",
        "status": "alive",
        "first_seen_frame": 0,
        "last_seen_frame": 19,
        "age_frames": 20,
        "duration_seconds": 0.76,
        "path_length": 0.0,
        "average_speed": 0.0,
        "average_confidence": 0.9,
        "times_lost": 0,
        # stays fully inside the [0,0]-[100,100] restricted zone for all 20 frames
        "history": make_history(20, bbox=[10, 10, 50, 50], centroid=[30, 30]),
    }
}


# =====================================================================
# PipelineConfig validation
# =====================================================================

def test_config_rejects_near_miss_distance_above_proximity_threshold():
    with pytest.raises(ValueError):
        PipelineConfig(
            model_path="dummy.pt",
            video_path="dummy.mp4",
            zones_path="dummy_zones.json",
            spatial=SpatialConfig(proximity_threshold_px=100.0),
            rules=RuleEngineConfig(near_miss_distance_px=200.0),
        )


def test_config_accepts_near_miss_distance_at_or_below_proximity_threshold():
    config = PipelineConfig(
        model_path="dummy.pt",
        video_path="dummy.mp4",
        zones_path="dummy_zones.json",
        spatial=SpatialConfig(proximity_threshold_px=150.0),
        rules=RuleEngineConfig(near_miss_distance_px=150.0),
    )
    assert config.rules.near_miss_distance_px == 150.0


def test_config_builds_tracker_config_from_model_path():
    config = PipelineConfig(model_path="weights/best.pt", video_path="dummy.mp4", zones_path="dummy_zones.json")
    assert config.tracker.model_path == "weights/best.pt"


# =====================================================================
# ChainSightPipeline.run() — full stage wiring, tracker mocked
# =====================================================================

def test_pipeline_run_produces_all_stage_outputs_and_fires_r1(tmp_path, mocker):
    zones_path = write_zones_file(tmp_path)

    mock_tracker_cls = mocker.patch("src.pipeline.ChainSightTracker")
    mock_tracker = mock_tracker_cls.return_value
    mock_tracker.summary.return_value = "1 track (mocked)"
    mock_tracker.to_json_serializable.return_value = PERSON_INSIDE_ZONE_TRACKS

    config = PipelineConfig(
        model_path="dummy.pt",
        video_path="dummy.mp4",
        zones_path=str(zones_path),
        output_dir=str(tmp_path / "outputs"),
        run_name="testclip",
    )
    pipeline = ChainSightPipeline(config)

    result = pipeline.run()

    # tracker stage was invoked through the mock, not a real model/video
    mock_tracker.run.assert_called_once_with("dummy.mp4", save_annotated=None)

    # every stage's output file exists, with the run_name suffix applied
    assert Path(result.manifest_path).name == "manifest_testclip.json"
    assert Path(result.tracks_path).name == "tracks_testclip.json"
    assert Path(result.spatial_events_path).name == "spatial_events_testclip.json"
    assert Path(result.world_graph_summary_path).name == "world_graph_summary_testclip.json"
    assert Path(result.rule_events_path).name == "rule_events_testclip.json"
    for p in (result.manifest_path, result.tracks_path, result.spatial_events_path,
              result.world_graph_summary_path, result.rule_events_path):
        assert Path(p).exists()

    # manifest records what the demo needs to locate this run's source video/zones
    with open(result.manifest_path) as f:
        manifest = json.load(f)
    assert manifest == {"video_path": "dummy.mp4", "zones_path": str(zones_path), "model_path": "dummy.pt"}

    # the person track (fully inside the restricted zone for all 20 frames)
    # should have flowed through spatial -> world_graph -> rules and fired R1
    assert len(result.rule_events) == 1
    event = result.rule_events[0]
    assert event.rule_id == "R1_RESTRICTED_ZONE_INTRUSION"
    assert event.track_ids == [1]
    assert event.frame == 0

    # rule_events.json on disk matches the in-memory result
    with open(result.rule_events_path) as f:
        saved_events = json.load(f)
    assert len(saved_events) == 1
    assert saved_events[0]["rule_id"] == "R1_RESTRICTED_ZONE_INTRUSION"


def test_pipeline_run_with_no_zone_activity_fires_nothing(tmp_path, mocker):
    zones_path = write_zones_file(tmp_path)

    tracks_outside_zone = {
        "1": {
            **PERSON_INSIDE_ZONE_TRACKS["1"],
            "history": make_history(20, bbox=[500, 500, 540, 540], centroid=[520, 520]),
        }
    }

    mock_tracker_cls = mocker.patch("src.pipeline.ChainSightTracker")
    mock_tracker = mock_tracker_cls.return_value
    mock_tracker.summary.return_value = "1 track (mocked)"
    mock_tracker.to_json_serializable.return_value = tracks_outside_zone

    config = PipelineConfig(
        model_path="dummy.pt",
        video_path="dummy.mp4",
        zones_path=str(zones_path),
        output_dir=str(tmp_path / "outputs"),
    )
    result = ChainSightPipeline(config).run()

    assert result.rule_events == []
