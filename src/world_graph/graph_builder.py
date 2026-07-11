"""
graph_builder.py — ChainSight World Graph Layer
Consumes tracker.py's tracks.json and spatial.py's spatial_events.json,
builds a NetworkX graph representation of the scene for the rule engine
to query.

Pipeline position:
    tracker.py (tracks.json) + spatial/*.py (spatial_events.json) -> world_graph (THIS) -> rules

Design principle (per ChainSight scope): deterministic, no training.
Ghost/churned tracks (tracker ID fragmentation on occluded static
objects — see tracker.py ID-churn caveat) are filtered out before the
graph is built, using a minimum track-lifespan threshold, so rules
don't fire on tracking noise.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Iterator, Set, Tuple

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chainsight.world_graph.builder")


@dataclass
class WorldGraphConfig:
    min_track_frames: int = 15       # drop tracks alive fewer frames than this (ghost/churn filter)
    include_zone_nodes: bool = True  # add zone nodes wherever a track is inside one


class GraphBuilder:
    """
    Builds NetworkX graphs of the warehouse scene.

    Two outputs:
      - frame graphs (iter_frame_graphs / build_frame_graph): one nx.Graph
        per analyzed frame, nodes = tracks + zones, edges = "inside_zone"
        (track->zone) and "near" (track<->track). This is the primary
        interface for the rule engine, which needs point-in-time state
        (e.g. "person in restricted zone AND forklift in same zone AND
        distance < threshold").
      - summary graph (build_summary_graph): one aggregate nx.Graph across
        the whole clip, with lifecycle/visit-duration attributes instead
        of per-frame state. Useful for narration context and debugging,
        not for rule evaluation.
    """

    def __init__(self, tracks_path: str, spatial_events_path: str, config: Optional[WorldGraphConfig] = None):
        self.config = config or WorldGraphConfig()
        self.tracks = self._load_tracks(tracks_path)
        self.spatial = self._load_spatial(spatial_events_path)
        self.valid_track_ids = self._filter_valid_tracks()
        logger.info(
            f"Loaded {len(self.tracks)} tracks, kept {len(self.valid_track_ids)} "
            f"after ghost-filter (min_track_frames={self.config.min_track_frames})"
        )

    # --- loading (mirrors error-handling style of spatial/analyzer.py) ---
    def _load_tracks(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"tracks.json not found: {path}")
        try:
            with open(p) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"tracks.json is not valid JSON: {e}") from e
        if not isinstance(data, dict) or not data:
            logger.warning("tracks.json is empty or malformed — no tracks to graph.")
        return data

    def _load_spatial(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"spatial_events.json not found: {path}")
        try:
            with open(p) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"spatial_events.json is not valid JSON: {e}") from e
        if "frames" not in data or "zone_transitions" not in data:
            raise ValueError("spatial_events.json missing 'frames' or 'zone_transitions' keys.")
        return data

    # --- ghost/churn filter ---
    def _filter_valid_tracks(self) -> Set[int]:
        valid = set()
        for track_id_str, track in self.tracks.items():
            age = track.get("age_frames", 0)
            if age >= self.config.min_track_frames:
                valid.add(int(track_id_str))
        return valid

    # --- per-frame graphs (primary rule-engine interface) ---
    def iter_frame_graphs(self) -> Iterator[nx.Graph]:
        """Yields one nx.Graph per frame present in spatial_events.json, in frame order."""
        frames = self.spatial["frames"]
        for frame_idx_str in sorted(frames.keys(), key=int):
            yield self.build_frame_graph(int(frame_idx_str))

    def build_frame_graph(self, frame_idx: int) -> nx.Graph:
        frame_events = self.spatial["frames"].get(str(frame_idx), [])
        G = nx.Graph()
        G.graph["frame"] = frame_idx

        zones_seen: Set[str] = set()

        for event in frame_events:
            track_id = event["track_id"]
            if track_id not in self.valid_track_ids:
                continue  # skip ghost/churned tracks

            G.add_node(
                track_id,
                node_type="track",
                class_name=event["class_name"],
                centroid=tuple(event["centroid"]),
                timestamp=event["timestamp"],
            )

            if self.config.include_zone_nodes:
                for zone_name in event.get("zones_inside", []):
                    if zone_name not in zones_seen:
                        G.add_node(zone_name, node_type="zone")
                        zones_seen.add(zone_name)
                    G.add_edge(track_id, zone_name, relation="inside_zone")

            for near in event.get("nearby_tracks", []):
                near_id = near["track_id"]
                if near_id not in self.valid_track_ids:
                    continue
                G.add_edge(track_id, near_id, relation="near", distance_px=near["distance_px"])

        return G

    # --- aggregate summary graph (narration / debugging) ---
    def build_summary_graph(self) -> nx.Graph:
        """
        One graph for the whole clip: track nodes carry lifecycle attrs
        (class, duration, avg_speed, status), zone nodes are visited by
        edges carrying frame_count / first_frame / last_frame instead of
        per-frame state. Not used by the rule engine directly.
        """
        G = nx.Graph()

        for track_id_str, track in self.tracks.items():
            track_id = int(track_id_str)
            if track_id not in self.valid_track_ids:
                continue
            G.add_node(
                track_id,
                node_type="track",
                class_name=track["class_name"],
                first_seen_frame=track.get("first_seen_frame"),
                last_seen_frame=track.get("last_seen_frame"),
                age_frames=track.get("age_frames"),
                duration_seconds=track.get("duration_seconds"),
                average_speed=track.get("average_speed"),
                status=track.get("status"),
            )

        zone_edge_stats: Dict[Tuple[int, str], dict] = {}
        near_edge_stats: Dict[Tuple[int, int], dict] = {}

        for frame_idx_str, events in self.spatial["frames"].items():
            frame_idx = int(frame_idx_str)
            for event in events:
                track_id = event["track_id"]
                if track_id not in self.valid_track_ids:
                    continue

                for zone_name in event.get("zones_inside", []):
                    G.add_node(zone_name, node_type="zone")
                    key = (track_id, zone_name)
                    stats = zone_edge_stats.setdefault(
                        key, {"frame_count": 0, "first_frame": frame_idx, "last_frame": frame_idx}
                    )
                    stats["frame_count"] += 1
                    stats["last_frame"] = max(stats["last_frame"], frame_idx)
                    stats["first_frame"] = min(stats["first_frame"], frame_idx)

                for near in event.get("nearby_tracks", []):
                    near_id = near["track_id"]
                    if near_id not in self.valid_track_ids:
                        continue
                    key = (track_id, near_id) if track_id < near_id else (near_id, track_id)
                    stats = near_edge_stats.setdefault(
                        key, {"frame_count": 0, "min_distance": near["distance_px"],
                              "first_frame": frame_idx, "last_frame": frame_idx}
                    )
                    stats["frame_count"] += 1
                    stats["min_distance"] = min(stats["min_distance"], near["distance_px"])
                    stats["last_frame"] = max(stats["last_frame"], frame_idx)
                    stats["first_frame"] = min(stats["first_frame"], frame_idx)

        for (track_id, zone_name), stats in zone_edge_stats.items():
            G.add_edge(track_id, zone_name, relation="visited_zone", **stats)

        for (id_a, id_b), stats in near_edge_stats.items():
            G.add_edge(id_a, id_b, relation="near", **stats)

        return G

    # --- export helpers ---
    def frame_graph_to_json(self, G: nx.Graph) -> dict:
        """node-link JSON representation of a single frame graph."""
        return nx.node_link_data(G, edges="edges")

    def summary(self) -> str:
        G = self.build_summary_graph()
        track_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "track"]
        zone_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "zone"]
        return (
            f"World graph: {len(track_nodes)} track nodes, {len(zone_nodes)} zone nodes, "
            f"{G.number_of_edges()} edges (ghost-filtered, min_track_frames="
            f"{self.config.min_track_frames})"
        )
