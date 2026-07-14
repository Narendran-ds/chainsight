"""
run_data.py — ChainSight demo, run discovery + data loading
Pure data access for the Streamlit demo: which runs exist under outputs/,
and loading each stage's JSON for a selected run. No Streamlit dependency,
so this can be unit-tested without launching the app.

Pipeline position:
    outputs/*.json (all stages) -> demo/components/run_data.py (THIS)
        -> demo/components/{video_player,event_timeline,report_card,internals}.py

Design principle: every load_* function tolerates a missing file by
returning None/empty rather than raising — a run assembled by hand from the
individual scripts/run_*.py calls (rather than the full pipeline) may be
missing world_graph/narration/manifest outputs, and the UI should degrade
gracefully (e.g. "narration not generated for this run") instead of crashing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def discover_runs(output_dir: str) -> List[str]:
    """Run names with a tracks_<run>.json (or bare tracks.json) in output_dir.
    Bare tracks.json (no run_name suffix) is represented as "" (default run)."""
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return []
    runs = []
    for p in sorted(out_dir.glob("tracks*.json")):
        stem = p.stem  # "tracks" or "tracks_<run>"
        if stem == "tracks":
            runs.append("")
        elif stem.startswith("tracks_"):
            runs.append(stem[len("tracks_"):])
    return runs


def _stage_path(output_dir: str, stem: str, run_name: str) -> Path:
    suffix = f"_{run_name}" if run_name else ""
    return Path(output_dir) / f"{stem}{suffix}.json"


def resolve_run_paths(output_dir: str, run_name: str) -> Dict[str, Optional[Path]]:
    """Path for each stage's output for run_name, or None if that file doesn't exist."""
    stems = ("manifest", "tracks", "spatial_events", "world_graph_summary", "rule_events", "narration")
    return {stem: (p if (p := _stage_path(output_dir, stem, run_name)).exists() else None) for stem in stems}


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def load_tracks(path: Optional[Path]) -> Dict[str, dict]:
    """track_id -> track dict, with tracker.py's "_meta" key popped out."""
    data = load_json(path)
    if not data:
        return {}
    data = dict(data)  # don't mutate a cached load
    data.pop("_meta", None)
    return data


def load_frame_meta(path: Optional[Path]) -> Dict[str, Optional[int]]:
    """The "_meta" (frame_width/frame_height) tracker.py wrote for this run, if any."""
    data = load_json(path)
    if not data or "_meta" not in data:
        return {"frame_width": None, "frame_height": None}
    return data["_meta"]


def load_zones(path: Optional[Path]) -> List[dict]:
    data = load_json(path)
    return data.get("zones", []) if data else []


def load_rule_events(path: Optional[Path]) -> List[dict]:
    return load_json(path) or []


def index_tracks_by_frame(tracks: Dict[str, dict]) -> Dict[int, List[dict]]:
    """frame_idx -> [{"track_id", "class_name", "bbox"}, ...], built once per
    run so the frame scrubber does an O(1) lookup per slider move instead of
    rescanning every track's full history on every frame change."""
    index: Dict[int, List[dict]] = {}
    for track_id_str, track in tracks.items():
        class_name = track.get("class_name")
        for obs in track.get("history", []):
            index.setdefault(obs["frame"], []).append({
                "track_id": int(track_id_str),
                "class_name": class_name,
                "bbox": obs["bbox"],
            })
    return index


def frame_range(tracks: Dict[str, dict]) -> Tuple[int, int]:
    """(min_frame, max_frame) across all tracks' history; (0, 0) if empty."""
    frames = [obs["frame"] for track in tracks.values() for obs in track.get("history", [])]
    return (min(frames), max(frames)) if frames else (0, 0)


def derive_fps(tracks: Dict[str, dict], default: float = 25.0) -> float:
    """
    Recovers the clip's fps from tracks.json's frame/timestamp relationship
    (tracker.py sets timestamp = round(frame / fps, 4) for every observation)
    instead of requiring a dedicated fps field in "_meta" — lets frame<->time
    conversion work on tracks.json files written before this was needed.
    Uses the observation with the largest frame index for best precision
    against the timestamp's 4-decimal rounding.
    """
    best: Optional[Tuple[int, float]] = None
    for track in tracks.values():
        for obs in track.get("history", []):
            frame, ts = obs.get("frame", 0), obs.get("timestamp", 0)
            if frame > 0 and ts > 0 and (best is None or frame > best[0]):
                best = (frame, ts)
    if best is None:
        return default
    frame, ts = best
    return frame / ts
