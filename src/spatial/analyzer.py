"""
analyzer.py — ChainSight Spatial Layer, orchestration
Consumes tracker.py's tracks.json + zones.py's loaded Zone list and computes,
per frame, per track: zone containment (via overlap.py), zone transition events
(entered/left), and deduplicated nearby-object proximity pairs (via STRtree).

Pipeline position:
    tracker.py (tracks.json) -> spatial/{zones,overlap,analyzer}.py -> world_graph -> rule engine

Improvements implemented here (continuing from zones.py / overlap.py):
  6  Zone transition events (entered / left), not just repeated per-frame state
  2  Squared-distance comparison (avoid sqrt until needed for output)
  9  Deduplicated proximity pairs (unordered pair stored once)
  1  STRtree spatial index for proximity search (avoids O(N^2) brute force)
  14 Error handling around file loads
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from shapely.geometry import Point

from .zones import Zone, SpatialConfig, load_zones
from .overlap import zones_containing_bbox
from ..utils.video_io import resolution_scale

try:
    from shapely.strtree import STRtree
except ImportError:  # very old shapely fallback, shouldn't happen but just in case
    STRtree = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chainsight.spatial.analyzer")


@dataclass
class SpatialEvent:
    """A single frame's spatial facts for one track."""
    frame: int
    timestamp: float
    track_id: int
    class_name: str
    centroid: Tuple[float, float]
    zones_inside: List[str] = field(default_factory=list)
    nearby_tracks: List[dict] = field(default_factory=list)  # [{track_id, class_name, distance_px}]


@dataclass
class ZoneTransitionEvent:
    """Fired when a track's zone membership changes between consecutive frames."""
    frame: int
    timestamp: float
    track_id: int
    class_name: str
    zone_name: str
    event_type: str  # "entered" | "left"


class SpatialAnalyzer:
    """
    Loads tracks.json + zones (via zones.load_zones) and computes zone containment
    (bbox-overlap based, via overlap.py), zone transition events, and deduplicated
    proximity pairs (via STRtree), frame by frame.
    """

    def __init__(self, zones_path: str, config: Optional[SpatialConfig] = None):
        self.config = config or SpatialConfig()
        self.zones: List[Zone] = load_zones(zones_path)
        logger.info(f"Loaded {len(self.zones)} zone(s): {[z.name for z in self.zones]}")

    # --- loading (14) ---
    def load_tracks(self, tracks_path: str) -> dict:
        path = Path(tracks_path)
        if not path.exists():
            raise FileNotFoundError(f"Tracks file not found: {tracks_path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"tracks.json is not valid JSON: {e}") from e
        if not isinstance(data, dict) or not data:
            logger.warning("tracks.json is empty or malformed — no tracks to analyze.")
        return data

    @staticmethod
    def _pop_meta(tracks_data: dict) -> dict:
        """Split tracker.py's "_meta" (frame_width/frame_height) out of the
        track-id-keyed dict. Missing on tracks.json produced before this
        field existed — callers must tolerate an empty dict here."""
        return tracks_data.pop("_meta", {}) if isinstance(tracks_data, dict) else {}

    # --- main analysis ---
    def analyze(self, tracks_path: str) -> Tuple[Dict[int, List[SpatialEvent]], List[ZoneTransitionEvent]]:
        tracks_data = self.load_tracks(tracks_path)
        meta = self._pop_meta(tracks_data)
        scale = resolution_scale(meta.get("frame_width"))
        if scale != 1.0:
            logger.info(
                f"Scaling proximity_threshold_px by {scale:.2f}x for frame_width="
                f"{meta.get('frame_width')} (reference width used to calibrate the default)."
            )

        # frame_idx -> list of (track_id, class_name, timestamp, bbox, centroid)
        frame_buckets: Dict[int, List[tuple]] = {}
        for track_id_str, track in tracks_data.items():
            try:
                track_id = int(track_id_str)
                class_name = track["class_name"]
                for obs in track["history"]:
                    frame_idx = obs["frame"]
                    timestamp = obs.get("timestamp", frame_idx)  # defensive default (14)
                    bbox = obs["bbox"]
                    centroid = tuple(obs["centroid"])
                    frame_buckets.setdefault(frame_idx, []).append(
                        (track_id, class_name, timestamp, bbox, centroid)
                    )
            except (KeyError, TypeError) as e:
                logger.error(f"Skipping malformed track '{track_id_str}': {e}")
                continue

        events_by_frame: Dict[int, List[SpatialEvent]] = {}
        zone_transitions: List[ZoneTransitionEvent] = []
        prev_zones_by_track: Dict[int, Set[str]] = {}

        proximity_threshold_px = self.config.proximity_threshold_px * scale
        threshold_sq = proximity_threshold_px ** 2  # (2) avoid sqrt in the hot loop

        for frame_idx in sorted(frame_buckets.keys()):
            entries = frame_buckets[frame_idx]

            # --- proximity via spatial index (1) ---
            centroid_points = [Point(e[4]) for e in entries]
            pair_distances_sq: Dict[Tuple[int, int], float] = {}  # (dedup, 9)

            if len(entries) > 1 and STRtree is not None:
                tree = STRtree(centroid_points)
                for i, pt in enumerate(centroid_points):
                    track_id_i = entries[i][0]
                    candidate_idxs = tree.query(pt.buffer(proximity_threshold_px))
                    for j in candidate_idxs:
                        j = int(j)
                        if j <= i:
                            continue  # dedup: only consider each unordered pair once (9)
                        track_id_j = entries[j][0]
                        dx = entries[i][4][0] - entries[j][4][0]
                        dy = entries[i][4][1] - entries[j][4][1]
                        dist_sq = dx * dx + dy * dy
                        if dist_sq <= threshold_sq:
                            key = (track_id_i, track_id_j) if track_id_i < track_id_j else (track_id_j, track_id_i)
                            pair_distances_sq[key] = dist_sq
            elif len(entries) > 1:
                # fallback brute-force if STRtree unavailable
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        dx = entries[i][4][0] - entries[j][4][0]
                        dy = entries[i][4][1] - entries[j][4][1]
                        dist_sq = dx * dx + dy * dy
                        if dist_sq <= threshold_sq:
                            id_a, id_b = entries[i][0], entries[j][0]
                            key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                            pair_distances_sq[key] = dist_sq

            class_by_id = {e[0]: e[1] for e in entries}
            nearby_by_track: Dict[int, List[dict]] = {e[0]: [] for e in entries}
            for (id_a, id_b), dist_sq in pair_distances_sq.items():
                dist_px = round(dist_sq ** 0.5, 2)  # sqrt only here, once per real pair (2)
                nearby_by_track[id_a].append({"track_id": id_b, "class_name": class_by_id[id_b], "distance_px": dist_px})
                nearby_by_track[id_b].append({"track_id": id_a, "class_name": class_by_id[id_a], "distance_px": dist_px})

            # --- zones + transitions (6) ---
            frame_events = []
            for track_id, class_name, timestamp, bbox, centroid in entries:
                zones_inside = set(zones_containing_bbox(bbox, self.zones, self.config))
                prev_zones = prev_zones_by_track.get(track_id, set())

                for entered_zone in zones_inside - prev_zones:
                    zone_transitions.append(ZoneTransitionEvent(
                        frame=frame_idx, timestamp=timestamp, track_id=track_id,
                        class_name=class_name, zone_name=entered_zone, event_type="entered",
                    ))
                for left_zone in prev_zones - zones_inside:
                    zone_transitions.append(ZoneTransitionEvent(
                        frame=frame_idx, timestamp=timestamp, track_id=track_id,
                        class_name=class_name, zone_name=left_zone, event_type="left",
                    ))
                prev_zones_by_track[track_id] = zones_inside

                frame_events.append(SpatialEvent(
                    frame=frame_idx,
                    timestamp=timestamp,
                    track_id=track_id,
                    class_name=class_name,
                    centroid=centroid,
                    zones_inside=sorted(zones_inside),
                    nearby_tracks=sorted(nearby_by_track.get(track_id, []), key=lambda n: n["distance_px"]),
                ))

            events_by_frame[frame_idx] = frame_events

        return events_by_frame, zone_transitions

    # --- export ---
    def to_json_serializable(
        self,
        events_by_frame: Dict[int, List[SpatialEvent]],
        zone_transitions: List[ZoneTransitionEvent],
    ) -> dict:
        frames_out = {}
        for frame_idx, events in events_by_frame.items():
            frames_out[str(frame_idx)] = [
                {
                    "timestamp": e.timestamp,
                    "track_id": e.track_id,
                    "class_name": e.class_name,
                    "centroid": list(e.centroid),
                    "zones_inside": e.zones_inside,
                    "nearby_tracks": e.nearby_tracks,
                }
                for e in events
            ]

        transitions_out = [
            {
                "frame": t.frame,
                "timestamp": t.timestamp,
                "track_id": t.track_id,
                "class_name": t.class_name,
                "zone_name": t.zone_name,
                "event_type": t.event_type,
            }
            for t in zone_transitions
        ]

        return {"frames": frames_out, "zone_transitions": transitions_out}

    def summary(self, events_by_frame: Dict[int, List[SpatialEvent]], zone_transitions: List[ZoneTransitionEvent]) -> str:
        zone_entries = 0
        proximity_events = 0
        classes_in_zones: Dict[str, set] = {}

        for events in events_by_frame.values():
            for e in events:
                if e.zones_inside:
                    zone_entries += 1
                    for z in e.zones_inside:
                        classes_in_zones.setdefault(z, set()).add(e.class_name)
                if e.nearby_tracks:
                    proximity_events += 1

        entered_count = sum(1 for t in zone_transitions if t.event_type == "entered")
        left_count = sum(1 for t in zone_transitions if t.event_type == "left")

        lines = [
            f"Analyzed {len(events_by_frame)} frames.",
            f"Zone-containment observations: {zone_entries}",
            f"Proximity observations: {proximity_events}",
            f"Zone transitions: {entered_count} entered, {left_count} left",
        ]
        for zone_name, classes in classes_in_zones.items():
            lines.append(f"  Zone '{zone_name}' visited by classes: {sorted(classes)}")
        return "\n".join(lines)