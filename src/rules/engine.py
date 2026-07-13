"""
engine.py — ChainSight Rule Engine
Consumes GraphBuilder/WorldQuery (world_graph layer) plus zones.json's
zone-type metadata to evaluate five deterministic safety rules:

  R1  Restricted Zone Intrusion  — person enters any "restricted" zone
  R2  Forklift-Pedestrian Near-Miss — person+forklift near, in a
      restricted zone, sustained for N consecutive frames
  R3  PPE Violation — person inside a zone while a no_vest/no_helmet
      detection's bbox overlaps theirs (bbox-overlap association, not
      centroid-nearest-neighbor — see rule_definitions.py)
  R4  Exit Blockage — a non-person object (box, pallet, forklift, etc.)
      occupies an "exit"-type zone continuously for a duration threshold.
      Duration is computed from continuous per-frame presence, not from
      entered/left transition pairs, because a blocking object's track
      may go dead (occlusion, end of clip) while still inside the zone
      — i.e. no "left" event is ever logged, but the exit is still blocked.
  R5  Loitering in Restricted Zone — person remains inside a restricted
      zone longer than a duration threshold

Pipeline position:
    world_graph/{graph_builder,query}.py + configs/zones.json
        -> rules/engine.py (THIS) -> narration / Streamlit demo

Design principle: deterministic, no training. Every fired rule produces
a RuleEvent with trigger/evidence/conclusion (see events.py) rather than
a bare boolean, per the project's explainability requirement.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..spatial.zones import load_zones
from ..world_graph.query import WorldQuery
from .events import RuleEvent
from .rule_definitions import RuleEngineConfig, bbox_overlap_ratio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chainsight.rules.engine")


class RuleEngine:
    def __init__(self, query: WorldQuery, zones_path: str, config: Optional[RuleEngineConfig] = None):
        self.query = query
        self.config = config or RuleEngineConfig()
        self.builder = query.builder  # GraphBuilder, for direct spatial/tracks access

        zones = load_zones(zones_path)
        self.zone_types: Dict[str, str] = {z.name: z.zone_type for z in zones}

        self._frame_bbox_index = self._build_frame_bbox_index()

    # --- setup: per-frame bbox lookup, built from ALL tracks (unfiltered) ---
    def _build_frame_bbox_index(self) -> Dict[int, Dict[int, dict]]:
        """
        frame_idx -> {track_id: {"class_name", "bbox", "conf"}}
        Deliberately built from self.builder.tracks directly (not the
        ghost-filtered self.builder.valid_track_ids) because PPE
        detections (no_vest, no_helmet) are typically short-lived boxes
        that would fail the world graph's min_track_frames filter, but
        their per-frame bbox is still valid evidence for R3 regardless
        of tracker ID continuity.
        """
        index: Dict[int, Dict[int, dict]] = {}
        for track_id_str, track in self.builder.tracks.items():
            track_id = int(track_id_str)
            class_name = track.get("class_name")
            for obs in track.get("history", []):
                index.setdefault(obs["frame"], {})[track_id] = {
                    "class_name": class_name,
                    "bbox": obs["bbox"],
                    "conf": obs.get("conf"),
                }
        return index

    def _is_restricted(self, zone_name: str) -> bool:
        return self.zone_types.get(zone_name) in self.config.restricted_zone_types

    def _is_exit(self, zone_name: str) -> bool:
        return self.zone_types.get(zone_name) in self.config.exit_zone_types

    # --- R1: Restricted Zone Intrusion ---
    def _run_r1(self) -> List[RuleEvent]:
        events = []
        for t in self.builder.spatial["zone_transitions"]:
            if (
                t["event_type"] == "entered"
                and t["class_name"] == "person"
                and self._is_restricted(t["zone_name"])
            ):
                events.append(RuleEvent(
                    rule_id="R1_RESTRICTED_ZONE_INTRUSION",
                    rule_name="Restricted Zone Intrusion",
                    frame=t["frame"],
                    timestamp=t["timestamp"],
                    severity="warning",
                    track_ids=[t["track_id"]],
                    trigger=f"Person {t['track_id']} entered restricted zone '{t['zone_name']}'.",
                    evidence={"zone_name": t["zone_name"], "zone_type": self.zone_types.get(t["zone_name"])},
                    conclusion=(
                        f"Person {t['track_id']} entered restricted zone '{t['zone_name']}' "
                        f"at frame {t['frame']}. Reason: person detected inside a restricted zone."
                    ),
                ))
        return events

    # --- R2: Forklift-Pedestrian Near-Miss (temporal persistence) ---
    def _run_r2(self) -> List[RuleEvent]:
        events = []
        consecutive_count: Dict[Tuple[int, int], int] = {}
        fired: Dict[Tuple[int, int], bool] = {}

        frames = self.builder.spatial["frames"]
        for frame_idx_str in sorted(frames.keys(), key=int):
            frame_idx = int(frame_idx_str)
            frame_events = frames[frame_idx_str]

            active_pairs: Set[Tuple[int, int]] = set()
            for event in frame_events:
                if event["class_name"] != "person":
                    continue
                person_zones = [z for z in event.get("zones_inside", []) if self._is_restricted(z)]
                if not person_zones:
                    continue
                for near in event.get("nearby_tracks", []):
                    if near["class_name"] != "forklift":
                        continue
                    if near["distance_px"] > self.config.near_miss_distance_px:
                        continue
                    active_pairs.add((event["track_id"], near["track_id"]))

            for pair in list(consecutive_count.keys()):
                if pair not in active_pairs:
                    consecutive_count[pair] = 0
                    fired[pair] = False
            for pair in active_pairs:
                consecutive_count[pair] = consecutive_count.get(pair, 0) + 1

                if (
                    consecutive_count[pair] >= self.config.near_miss_min_consecutive_frames
                    and not fired.get(pair, False)
                ):
                    person_id, forklift_id = pair
                    fired[pair] = True
                    events.append(RuleEvent(
                        rule_id="R2_FORKLIFT_PEDESTRIAN_NEAR_MISS",
                        rule_name="Forklift-Pedestrian Near-Miss",
                        frame=frame_idx,
                        timestamp=next(
                            (e["timestamp"] for e in frame_events if e["track_id"] == person_id), 0.0
                        ),
                        severity="critical",
                        track_ids=[person_id, forklift_id],
                        trigger=(
                            f"Person {person_id} sustained proximity to forklift {forklift_id} "
                            f"in a restricted zone for {consecutive_count[pair]}+ consecutive frames."
                        ),
                        evidence={
                            "distance_threshold_px": self.config.near_miss_distance_px,
                            "consecutive_frames": consecutive_count[pair],
                        },
                        conclusion=(
                            f"Person {person_id} and forklift {forklift_id} remained within "
                            f"{self.config.near_miss_distance_px}px of each other inside a "
                            f"restricted zone for at least "
                            f"{self.config.near_miss_min_consecutive_frames} consecutive frames. "
                            f"Reason: sustained near-miss proximity, not a brief pass-through."
                        ),
                    ))
        return events

    # --- R3: PPE Violation (bbox-overlap association) ---
    def _run_r3(self) -> List[RuleEvent]:
        events = []
        last_violations: Dict[int, frozenset] = {}

        frames = self.builder.spatial["frames"]
        for frame_idx_str in sorted(frames.keys(), key=int):
            frame_idx = int(frame_idx_str)
            frame_events = frames[frame_idx_str]
            bbox_this_frame = self._frame_bbox_index.get(frame_idx, {})

            for event in frame_events:
                if event["class_name"] != "person" or not event.get("zones_inside"):
                    continue
                person_id = event["track_id"]
                person_det = bbox_this_frame.get(person_id)
                if not person_det:
                    continue
                person_bbox = person_det["bbox"]

                violations = set()
                for other_id, det in bbox_this_frame.items():
                    if det["class_name"] not in self.config.ppe_violation_classes:
                        continue
                    if bbox_overlap_ratio(det["bbox"], person_bbox) >= self.config.ppe_overlap_threshold:
                        violations.add(det["class_name"])

                prev = last_violations.get(person_id, frozenset())
                if violations and frozenset(violations) != prev:
                    zone_name = event["zones_inside"][0]
                    events.append(RuleEvent(
                        rule_id="R3_PPE_VIOLATION",
                        rule_name="PPE Violation",
                        frame=frame_idx,
                        timestamp=event["timestamp"],
                        severity="warning",
                        track_ids=[person_id],
                        trigger=f"Person {person_id} missing PPE: {sorted(violations)} in zone '{zone_name}'.",
                        evidence={"zone_name": zone_name, "missing_ppe": sorted(violations)},
                        conclusion=(
                            f"Person {person_id} detected inside zone '{zone_name}' without "
                            f"required PPE ({', '.join(sorted(violations))}). Reason: PPE-class "
                            f"detection bbox overlaps person bbox above threshold "
                            f"({self.config.ppe_overlap_threshold})."
                        ),
                    ))
                last_violations[person_id] = frozenset(violations)
        return events

    # --- R4: Exit Blockage (continuous per-frame presence, not entered/left pairs) ---
    def _run_r4(self) -> List[RuleEvent]:
        events = []
        frames = self.builder.spatial["frames"]

        # (track_id, zone_name) -> list of (frame_idx, timestamp) while continuously present
        presence: Dict[Tuple[int, str], List[Tuple[int, float]]] = {}

        for frame_idx_str in sorted(frames.keys(), key=int):
            frame_idx = int(frame_idx_str)
            for event in frames[frame_idx_str]:
                if event["class_name"] not in self.config.exit_blocking_classes:
                    continue
                for zone_name in event.get("zones_inside", []):
                    if not self._is_exit(zone_name):
                        continue
                    key = (event["track_id"], zone_name)
                    presence.setdefault(key, []).append((frame_idx, event["timestamp"]))

        for (track_id, zone_name), obs in presence.items():
            if len(obs) < 2:
                continue
            first_frame, first_ts = obs[0]
            last_frame, last_ts = obs[-1]
            duration = last_ts - first_ts
            if duration >= self.config.min_exit_block_seconds:
                # class_name lives one level up on the track record itself,
                # not inside individual per-frame history entries.
                track_record = self.builder.tracks.get(str(track_id), {})
                class_name = track_record.get("class_name", "object")

                events.append(RuleEvent(
                    rule_id="R4_EXIT_BLOCKAGE",
                    rule_name="Exit Blockage",
                    frame=last_frame,
                    timestamp=last_ts,
                    severity="critical",
                    track_ids=[track_id],
                    trigger=(
                        f"{class_name.capitalize()} {track_id} occupied exit zone '{zone_name}' "
                        f"continuously for {duration:.1f}s."
                    ),
                    evidence={
                        "zone_name": zone_name,
                        "duration_seconds": round(duration, 2),
                        "first_frame": first_frame,
                        "last_frame": last_frame,
                        "class_name": class_name,
                    },
                    conclusion=(
                        f"A {class_name} (track {track_id}) remained inside exit zone "
                        f"'{zone_name}' for {duration:.1f} seconds (threshold: "
                        f"{self.config.min_exit_block_seconds}s), continuously through frame "
                        f"{last_frame}. Reason: sustained object presence in a designated exit "
                        f"path constitutes a blockage hazard."
                    ),
                ))
        return events

    # --- R5: Loitering in Restricted Zone (duration from zone_transitions) ---
    def _run_r5(self) -> List[RuleEvent]:
        events = []
        transitions = self.builder.spatial["zone_transitions"]

        by_key: Dict[Tuple[int, str], List[dict]] = {}
        for t in transitions:
            if t["class_name"] != "person" or not self._is_restricted(t["zone_name"]):
                continue
            key = (t["track_id"], t["zone_name"])
            by_key.setdefault(key, []).append(t)

        for (track_id, zone_name), events_list in by_key.items():
            events_list.sort(key=lambda t: t["frame"])
            entered_ts = None
            for t in events_list:
                if t["event_type"] == "entered":
                    entered_ts = t
                elif t["event_type"] == "left" and entered_ts is not None:
                    duration = t["timestamp"] - entered_ts["timestamp"]
                    if duration >= self.config.loiter_min_seconds:
                        events.append(RuleEvent(
                            rule_id="R5_LOITERING_IN_RESTRICTED_ZONE",
                            rule_name="Loitering in Restricted Zone",
                            frame=t["frame"],
                            timestamp=t["timestamp"],
                            severity="warning",
                            track_ids=[track_id],
                            trigger=(
                                f"Person {track_id} remained in restricted zone '{zone_name}' "
                                f"for {duration:.1f}s."
                            ),
                            evidence={
                                "zone_name": zone_name,
                                "duration_seconds": round(duration, 2),
                                "entered_frame": entered_ts["frame"],
                                "left_frame": t["frame"],
                            },
                            conclusion=(
                                f"Person {track_id} stayed inside restricted zone '{zone_name}' "
                                f"for {duration:.1f} seconds (threshold: "
                                f"{self.config.loiter_min_seconds}s). Reason: prolonged presence "
                                f"in a restricted area, not a brief pass-through."
                            ),
                        ))
                    entered_ts = None
        return events

    # --- orchestration ---
    def run(self) -> List[RuleEvent]:
        all_events = []
        all_events.extend(self._run_r1())
        all_events.extend(self._run_r2())
        all_events.extend(self._run_r3())
        all_events.extend(self._run_r4())
        all_events.extend(self._run_r5())
        all_events.sort(key=lambda e: e.frame)
        return all_events

    def summary(self, events: List[RuleEvent]) -> str:
        counts: Dict[str, int] = {}
        for e in events:
            counts[e.rule_id] = counts.get(e.rule_id, 0) + 1
        lines = [f"Rule engine fired {len(events)} event(s) total:"]
        for rule_id, count in sorted(counts.items()):
            lines.append(f"  {rule_id}: {count}")
        return "\n".join(lines)
