"""
run_rules.py — CLI entry point for the ChainSight rule engine.
Usage:
    python scripts\\run_rules.py --tracks outputs\\tracks.json --spatial outputs\\spatial_events.json --zones configs\\zones.json --out outputs\\rule_events.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.world_graph import GraphBuilder, WorldGraphConfig, WorldQuery
from src.rules import RuleEngine, RuleEngineConfig

logger = logging.getLogger("chainsight.rules.cli")


def main():
    parser = argparse.ArgumentParser(description="ChainSight Rule Engine")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json from tracker.py")
    parser.add_argument("--spatial", required=True, help="Path to spatial_events.json from run_spatial.py")
    parser.add_argument("--zones", required=True, help="Path to zones.json")
    parser.add_argument("--out", default="outputs/rule_events.json", help="Path to save fired rule events")
    parser.add_argument("--min-track-frames", type=int, default=15,
                         help="Ghost/churn filter for world graph track nodes (does not affect PPE/exit-blockage detection)")
    parser.add_argument("--near-miss-distance-px", type=float, default=100.0)
    parser.add_argument("--near-miss-min-frames", type=int, default=10)
    parser.add_argument("--near-miss-gap-tolerance-frames", type=int, default=3,
                         help="Brief gaps up to this many frames don't reset a near-miss streak "
                              "(tolerates single-frame tracker/detector flicker)")
    parser.add_argument("--ppe-overlap-threshold", type=float, default=0.5)
    parser.add_argument("--loiter-min-seconds", type=float, default=10.0)
    parser.add_argument("--min-exit-block-seconds", type=float, default=3.0)
    args = parser.parse_args()

    graph_config = WorldGraphConfig(min_track_frames=args.min_track_frames)
    try:
        builder = GraphBuilder(tracks_path=args.tracks, spatial_events_path=args.spatial, config=graph_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"World graph build failed: {e}")
        raise SystemExit(1)

    query = WorldQuery(builder)

    rule_config = RuleEngineConfig(
        near_miss_distance_px=args.near_miss_distance_px,
        near_miss_min_consecutive_frames=args.near_miss_min_frames,
        near_miss_gap_tolerance_frames=args.near_miss_gap_tolerance_frames,
        ppe_overlap_threshold=args.ppe_overlap_threshold,
        loiter_min_seconds=args.loiter_min_seconds,
        min_exit_block_seconds=args.min_exit_block_seconds,
    )

    try:
        engine = RuleEngine(query=query, zones_path=args.zones, config=rule_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Rule engine setup failed: {e}")
        raise SystemExit(1)

    events = engine.run()
    logger.info(engine.summary(events))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump([e.to_dict() for e in events], f, indent=2)
    logger.info(f"Saved {len(events)} rule event(s) to {args.out}")


if __name__ == "__main__":
    main()
