"""
graph_builder.py — ChainSight World Graph Layer
Consumes tracker.py's tracks.json and spatial.py's spatial_events.json,
builds a NetworkX graph representation of the scene for the rule engine
to query (via query.py's WorldQuery).

Pipeline position:
    tracker.py (tracks.json) + spatial/*.py (spatial_events.json)
        -> track_quality.py (valid track_ids)
        -> world_graph/graph_builder.py (THIS)
        -> world_graph/query.py -> rules

Design principle (per ChainSight scope): deterministic, no training.
Track-quality filtering (ghost/churned tracks from tracker ID
fragmentation) is delegated to utils/track_quality.py rather than done
inline here, so "what counts as a valid track" has one definition
shared by any future consumer of tracks.json, not just this layer.

Known scope decisions (documented, not bugs):
  - Undirected nx.Graph, single edge per node pair: sufficient for the
    zone-containment / proximity rules this project's rule engine
    currently needs. Directed/multi-edge relationships (e.g. "follows",
    "blocking") are deferred until a concrete rule requires them —
    see docs/scope_and_limitations.md.
  - Frame graphs are rebuilt independently per frame rather than
    incrementally updated. Fine at this project's scale (single clip,
    offline batch); would matter for real-time/streaming use.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Iterator, Set, Tuple

import networkx as nx

from ..utils.track_quality import TrackQualityConfig, filter_valid_tracks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chainsight.world_graph.builder")

# --- schema validation (#12) ---
# Minimum fields graph_builder relies on existing in each tracks.json entry.
# If tracker.py's export schema changes, this fails loudly here instead of
# silently producing an empty/wrong graph downstream.
REQUIRED_TRACK_KEYS = {"class_name", "age_frames"}
REQUIRED_SPATIAL_EVENT_KEYS = {"track_id", "class_name", "centroid", "timestamp"}


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
        (track->zone) and "near" (track<->track). Primary interface for
        the rule engine (via query.py) for point-in-time conditions.
      - summary graph (build_summary_graph): one aggregate nx.Graph across
        the whole clip, with lifecycle/visit-duration attributes instead
        of per-frame state. Useful for narration context and debugging.
    """

    def __init__(self, tracks_path: str, spatial_events_path: str, config: Optional[WorldGraphConfig] = None):
        self.config = config or WorldGraphConfig()
        self.tracks = self._load_tracks(tracks_path)
        self.frame_width: Optional[int] = self._meta.get("frame_width")
        self.frame_height: Optional[int] = self._meta.get("frame_height")
        self.spatial = self._load_spatial(spatial_events_path)
        self.valid_track_ids = filter_valid_tracks(
            self.tracks, TrackQualityConfig(min_track_frames=self.config.min_track_frames)
        )
        logger.info(
            f"Loaded {len(self.tracks)} tracks, kept {len(self.valid_track_ids)} "
            f"after track-quality filter (min_track_frames={self.config.min_track_frames})"
        )

    # --- loading + schema validation (#12) ---
    def _load_tracks(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"tracks.json not found: {path}")
        try:
            with open(p) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"tracks.json is not valid JSON: {e}") from e
        self._meta = data.pop("_meta", {}) if isinstance(data, dict) else {}

        if not isinstance(data, dict) or not data:
            logger.warning("tracks.json is empty or malformed — no tracks to graph.")
            return data

        sample_id, sample_track = next(iter(data.items()))
        missing = REQUIRED_TRACK_KEYS - set(sample_track.keys())
        if missing:
            raise ValueError(
                f"tracks.json schema mismatch: track '{sample_id}' is missing required "
                f"field(s) {missing}. tracker.py's export schema may have changed — "
                f"update REQUIRED_TRACK_KEYS in graph_builder.py if this is intentional."
            )
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

        for frame_events in data["frames"].values():
            if frame_events:
                missing = REQUIRED_SPATIAL_EVENT_KEYS - set(frame_events[0].keys())
                if missing:
                    raise ValueError(
                        f"spatial_events.json schema mismatch: event is missing required "
                        f"field(s) {missing}. spatial/analyzer.py's export schema may have "
                        f"changed — update REQUIRED_SPATIAL_EVENT_KEYS in graph_builder.py "
                        f"if this is intentional."
                    )
                break
        return data

    # --- per-frame graphs (primary rule-engine interface, via query.py) ---
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
        edges carrying frame_count / first_frame / last_frame / entry
        counts instead of per-frame state. Not used by the rule engine
        directly — see query.py for that.
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
                              "sum_distance": 0.0, "first_frame": frame_idx, "last_frame": frame_idx}
                    )
                    stats["frame_count"] += 1
                    stats["min_distance"] = min(stats["min_distance"], near["distance_px"])
                    stats["sum_distance"] += near["distance_px"]
                    stats["last_frame"] = max(stats["last_frame"], frame_idx)
                    stats["first_frame"] = min(stats["first_frame"], frame_idx)

        # zone_transitions -> per-track-per-zone entry counts (scoped #10)
        entry_counts: Dict[Tuple[int, str], int] = {}
        for t in self.spatial["zone_transitions"]:
            if t["event_type"] == "entered" and t["track_id"] in self.valid_track_ids:
                key = (t["track_id"], t["zone_name"])
                entry_counts[key] = entry_counts.get(key, 0) + 1

        for (track_id, zone_name), stats in zone_edge_stats.items():
            G.add_edge(
                track_id, zone_name, relation="visited_zone",
                transition_count=entry_counts.get((track_id, zone_name), 0),
                **stats,
            )

        for (id_a, id_b), stats in near_edge_stats.items():
            frame_count = stats.pop("frame_count")
            sum_distance = stats.pop("sum_distance")
            G.add_edge(
                id_a, id_b, relation="near",
                frame_count=frame_count,
                average_distance=round(sum_distance / frame_count, 2) if frame_count else None,
                total_time_near_frames=frame_count,
                **stats,
            )

        return G

    # --- export helpers ---
    def frame_graph_to_json(self, G: nx.Graph) -> dict:
        """
        node-link JSON representation of a single frame graph.
        NetworkX renamed the "links" key to "edges" (via the edges= kwarg)
        in 3.4+; older versions don't accept that kwarg at all. Try the
        modern call first, fall back for older installs so this works
        regardless of the installed networkx version.
        """
        try:
            return nx.node_link_data(G, edges="edges")
        except TypeError:
            return nx.node_link_data(G)

    def summary(self) -> str:
        G = self.build_summary_graph()
        track_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "track"]
        zone_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "zone"]
        return (
            f"World graph: {len(track_nodes)} track nodes, {len(zone_nodes)} zone nodes, "
            f"{G.number_of_edges()} edges (track-quality-filtered, min_track_frames="
            f"{self.config.min_track_frames})"
        )
