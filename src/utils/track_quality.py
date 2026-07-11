"""
track_quality.py — ChainSight Track Quality Layer
Decides which tracks from tracker.py's tracks.json are "real" objects
versus tracker-ID-churn artifacts (ghost tracks), independent of any
downstream consumer (world_graph, rule engine, narration).

Rationale: ghost filtering was originally embedded inside world_graph's
GraphBuilder, but track quality is a property of the tracker's OUTPUT,
not of graph construction. Keeping it here means any future consumer
of tracks.json (not just the world graph) gets the same, single
definition of "valid track" — one source of truth instead of each
downstream layer re-implementing its own filter.

Pipeline position:
    tracker.py (tracks.json) -> track_quality.py (THIS) -> world_graph / anything else
"""

import logging
from dataclasses import dataclass
from typing import Dict, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chainsight.track_quality")


@dataclass
class TrackQualityConfig:
    min_track_frames: int = 15  # tracks alive fewer frames than this are treated as ghost/churn


def filter_valid_tracks(tracks: Dict[str, dict], config: TrackQualityConfig = None) -> Set[int]:
    """
    Returns the set of track_ids considered "real" (not ghost/churn),
    based on tracker.py's age_frames field.

    A dedicated function (rather than inline filtering in a consumer)
    so tracker ID-churn — see tracker.py's documented ByteTrack-occlusion
    caveat — is defined once and reused anywhere tracks.json is consumed.
    """
    config = config or TrackQualityConfig()
    valid = set()
    for track_id_str, track in tracks.items():
        age = track.get("age_frames", 0)
        if age >= config.min_track_frames:
            valid.add(int(track_id_str))
    dropped = len(tracks) - len(valid)
    if dropped:
        logger.info(
            f"Track quality filter: kept {len(valid)}/{len(tracks)} tracks "
            f"(dropped {dropped} likely ghost/churn tracks, min_track_frames={config.min_track_frames})"
        )
    return valid
