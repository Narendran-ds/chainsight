"""
run_world_graph.py — CLI entry point for the ChainSight world graph layer.
Usage:
    python scripts\\run_world_graph.py --tracks outputs\\tracks.json --spatial outputs\\spatial_events.json --out outputs\\world_graph_summary.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# allow running from scripts/ without installing the package
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.world_graph import GraphBuilder, WorldGraphConfig

logger = logging.getLogger("chainsight.world_graph.cli")


def main():
    parser = argparse.ArgumentParser(description="ChainSight World Graph Layer (NetworkX)")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json from tracker.py")
    parser.add_argument("--spatial", required=True, help="Path to spatial_events.json from run_spatial.py")
    parser.add_argument("--out", default="outputs/world_graph_summary.json",
                         help="Path to save the aggregate summary graph (node-link JSON)")
    parser.add_argument("--min-track-frames", type=int, default=15,
                         help="Minimum track lifespan (frames) to keep; filters ghost/churned tracks")
    parser.add_argument("--dump-frame-graphs", action="store_true",
                         help="Also dump every per-frame graph to outputs/frame_graphs.json (large file)")
    args = parser.parse_args()

    config = WorldGraphConfig(min_track_frames=args.min_track_frames)

    try:
        builder = GraphBuilder(tracks_path=args.tracks, spatial_events_path=args.spatial, config=config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"World graph build failed: {e}")
        raise SystemExit(1)

    logger.info(builder.summary())

    summary_graph = builder.build_summary_graph()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(builder.frame_graph_to_json(summary_graph), f, indent=2)
    logger.info(f"Saved summary graph to {args.out}")

    if args.dump_frame_graphs:
        frame_graphs_out = Path(args.out).parent / "frame_graphs.json"
        all_frames = {}
        for G in builder.iter_frame_graphs():
            all_frames[str(G.graph["frame"])] = builder.frame_graph_to_json(G)
        with open(frame_graphs_out, "w") as f:
            json.dump(all_frames, f, indent=2)
        logger.info(f"Saved {len(all_frames)} per-frame graphs to {frame_graphs_out}")


if __name__ == "__main__":
    main()
