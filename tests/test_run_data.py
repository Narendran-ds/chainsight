"""
test_run_data.py — regression tests for demo/components/run_data.py

Pure data-loading/discovery logic, no Streamlit dependency — tests run
against real tmp_path files just like the other stage tests.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "demo"))

from components import run_data


def test_discover_runs_finds_suffixed_and_bare_tracks_files(tmp_path):
    (tmp_path / "tracks_nearmiss.json").write_text("{}")
    (tmp_path / "tracks_blocked_exit.json").write_text("{}")
    (tmp_path / "tracks.json").write_text("{}")
    (tmp_path / "spatial_events_nearmiss.json").write_text("{}")  # not a tracks_* file, ignored

    runs = run_data.discover_runs(str(tmp_path))

    assert set(runs) == {"nearmiss", "blocked_exit", ""}


def test_discover_runs_empty_when_output_dir_missing(tmp_path):
    assert run_data.discover_runs(str(tmp_path / "does_not_exist")) == []


def test_resolve_run_paths_none_for_missing_files(tmp_path):
    (tmp_path / "tracks_nearmiss.json").write_text("{}")
    (tmp_path / "rule_events_nearmiss.json").write_text("[]")

    paths = run_data.resolve_run_paths(str(tmp_path), "nearmiss")

    assert paths["tracks"] is not None
    assert paths["rule_events"] is not None
    assert paths["narration"] is None
    assert paths["manifest"] is None


def test_load_tracks_pops_meta_key(tmp_path):
    path = tmp_path / "tracks_x.json"
    path.write_text(json.dumps({
        "_meta": {"frame_width": 1920, "frame_height": 1080},
        "1": {"class_name": "person", "history": []},
    }))

    tracks = run_data.load_tracks(path)

    assert "_meta" not in tracks
    assert set(tracks.keys()) == {"1"}


def test_load_tracks_returns_empty_dict_for_missing_path():
    assert run_data.load_tracks(None) == {}


def test_load_frame_meta_defaults_when_absent(tmp_path):
    path = tmp_path / "tracks_x.json"
    path.write_text(json.dumps({"1": {"class_name": "person", "history": []}}))

    meta = run_data.load_frame_meta(path)

    assert meta == {"frame_width": None, "frame_height": None}


def test_load_zones_returns_zone_list(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": [{"name": "aisle", "type": "restricted", "polygon": [[0, 0]]}]}))

    zones = run_data.load_zones(path)

    assert zones == [{"name": "aisle", "type": "restricted", "polygon": [[0, 0]]}]


def test_load_rule_events_empty_list_for_missing_path():
    assert run_data.load_rule_events(None) == []


def test_index_tracks_by_frame_groups_observations_per_frame():
    tracks = {
        "1": {"class_name": "person", "history": [
            {"frame": 0, "bbox": [0, 0, 10, 10]},
            {"frame": 1, "bbox": [1, 1, 11, 11]},
        ]},
        "2": {"class_name": "forklift", "history": [
            {"frame": 0, "bbox": [20, 20, 40, 40]},
        ]},
    }

    index = run_data.index_tracks_by_frame(tracks)

    assert len(index[0]) == 2
    assert {e["track_id"] for e in index[0]} == {1, 2}
    assert index[1] == [{"track_id": 1, "class_name": "person", "bbox": [1, 1, 11, 11]}]


def test_frame_range_spans_all_tracks():
    tracks = {
        "1": {"history": [{"frame": 5}, {"frame": 10}]},
        "2": {"history": [{"frame": 0}, {"frame": 3}]},
    }
    assert run_data.frame_range(tracks) == (0, 10)


def test_frame_range_zero_for_no_tracks():
    assert run_data.frame_range({}) == (0, 0)
