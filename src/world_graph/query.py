"""
query.py — ChainSight World Graph Query Layer
A thin abstraction over GraphBuilder's NetworkX graphs so the rule
engine can ask questions in domain language instead of writing raw
graph traversals.

Rationale: without this layer, every rule in the rule engine would
need to know NetworkX internals (node/edge attribute names, graph
structure). This class is the only thing the rule engine should need
to import from world_graph.

Pipeline position:
    world_graph/graph_builder.py -> world_graph/query.py (THIS) -> rules
"""

from typing import List, Optional
import networkx as nx

from .graph_builder import GraphBuilder


class WorldQuery:
    """Domain-language queries over a ChainSight world graph."""

    def __init__(self, builder: GraphBuilder):
        self.builder = builder

    # --- point-in-time queries (operate on a single frame graph) ---
    def objects_in_zone(self, frame_idx: int, zone_name: str) -> List[int]:
        """Track IDs whose 'inside_zone' edge points to zone_name in this frame."""
        G = self.builder.build_frame_graph(frame_idx)
        if zone_name not in G:
            return []
        return [
            n for n in G.neighbors(zone_name)
            if G.nodes[n].get("node_type") == "track"
        ]

    def neighbors(self, frame_idx: int, track_id: int) -> List[int]:
        """Track IDs within proximity threshold of track_id in this frame ('near' edges)."""
        G = self.builder.build_frame_graph(frame_idx)
        if track_id not in G:
            return []
        return [
            n for n in G.neighbors(track_id)
            if G.nodes[n].get("node_type") == "track"
        ]

    def people_near_class(self, frame_idx: int, class_name: str, person_class: str = "person") -> List[dict]:
        """
        Convenience query for the common safety-rule shape:
        "which people are near an object of class X in this frame?"
        Returns [{"person_track_id", "other_track_id", "distance_px"}, ...]
        """
        G = self.builder.build_frame_graph(frame_idx)
        results = []
        for n, data in G.nodes(data=True):
            if data.get("node_type") != "track" or data.get("class_name") != class_name:
                continue
            for neighbor in G.neighbors(n):
                nd = G.nodes[neighbor]
                if nd.get("node_type") == "track" and nd.get("class_name") == person_class:
                    edge = G.get_edge_data(n, neighbor)
                    results.append({
                        "person_track_id": neighbor,
                        "other_track_id": n,
                        "distance_px": edge.get("distance_px"),
                    })
        return results

    # --- temporal queries (operate across the whole clip via zone_transitions) ---
    def zone_history(self, track_id: int) -> List[dict]:
        """
        All entered/left events for a track across the clip, in frame order.
        Sourced directly from spatial_events.json's zone_transitions — this
        is the "temporal edge" data (#8): entered_zone / left_zone over time,
        exposed here instead of duplicated as extra graph edges.
        """
        transitions = self.builder.spatial["zone_transitions"]
        history = [t for t in transitions if t["track_id"] == track_id]
        return sorted(history, key=lambda t: t["frame"])

    def zone_transitions_at_frame(self, frame_idx: int, event_type: Optional[str] = None) -> List[dict]:
        """All entered/left transitions occurring exactly at frame_idx (optionally filtered by type)."""
        transitions = self.builder.spatial["zone_transitions"]
        results = [t for t in transitions if t["frame"] == frame_idx]
        if event_type:
            results = [t for t in results if t["event_type"] == event_type]
        return results

    # --- aggregate/summary queries ---
    def summary_graph(self) -> nx.Graph:
        """Escape hatch to the raw aggregate graph, for narration/debugging only."""
        return self.builder.build_summary_graph()
