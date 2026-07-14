"""
run_pipeline.py — CLI entry point for the full ChainSight pipeline
(tracker -> spatial -> world_graph -> rules) in a single command.

Usage:
    python scripts\\run_pipeline.py --model models\\finetuned\\run2_exit_marker\\weights\\best.pt ^
        --video data\\staged_clips\\videos\\forklift_pedestrian_nearmiss.mp4 ^
        --zones configs\\zones_forklift_pedestrian_nearmiss.json --run-name nearmiss

Equivalent to running tracker.py, run_spatial.py, run_world_graph.py, and
run_rules.py in sequence by hand — see CLAUDE.md's "Reasoning pipeline"
section for what that looked like before this script existed.
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.pipeline import ChainSightPipeline, PipelineConfig
from src.vision.tracker import TrackerConfig
from src.spatial import SpatialConfig
from src.world_graph import WorldGraphConfig
from src.rules import RuleEngineConfig
from src.narration import GeminiClientConfig

logger = logging.getLogger("chainsight.pipeline.cli")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="ChainSight full pipeline (tracker -> spatial -> world_graph -> rules)")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 .pt weights")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--zones", required=True, help="Path to zones.json from define_zones.py")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write all stage outputs into")
    parser.add_argument("--run-name", default=None,
                         help="Suffix applied to every output file, e.g. 'nearmiss' -> tracks_nearmiss.json")
    parser.add_argument("--save-annotated", action="store_true", help="Also save an annotated video from the tracker stage")

    # --- tracker ---
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or leave unset for auto")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker-cfg", default="bytetrack.yaml")
    parser.add_argument("--max-history", type=int, default=5000)
    parser.add_argument("--max-missing-frames", type=int, default=30)

    # --- spatial ---
    parser.add_argument("--proximity-threshold", type=float, default=150.0,
                         help="Pixel distance under which two tracks are considered 'nearby'")
    parser.add_argument("--zone-overlap-threshold", type=float, default=0.1,
                         help="Minimum fraction of an object's bbox area that must overlap a zone to count as 'inside'")
    parser.add_argument("--centroid-only-zones", action="store_true",
                         help="Use legacy centroid-in-polygon zone check instead of bbox overlap")

    # --- world graph ---
    parser.add_argument("--min-track-frames", type=int, default=15,
                         help="Ghost/churn filter: minimum track lifespan (frames) to keep")

    # --- rules ---
    parser.add_argument("--near-miss-distance-px", type=float, default=100.0)
    parser.add_argument("--near-miss-min-frames", type=int, default=10)
    parser.add_argument("--near-miss-gap-tolerance-frames", type=int, default=3,
                         help="Brief gaps up to this many frames don't reset a near-miss streak "
                              "(tolerates single-frame tracker/detector flicker)")
    parser.add_argument("--ppe-overlap-threshold", type=float, default=0.5)
    parser.add_argument("--loiter-min-seconds", type=float, default=10.0)
    parser.add_argument("--min-exit-block-seconds", type=float, default=3.0)

    # --- narration (optional) ---
    parser.add_argument("--narrate", action="store_true",
                         help="Also run the Gemini narration stage on the fired rule events "
                              "(requires GEMINI_API_KEY — see .env.example)")
    parser.add_argument("--gemini-model", default="gemini-flash-latest")

    args = parser.parse_args()

    try:
        config = PipelineConfig(
            model_path=args.model,
            video_path=args.video,
            zones_path=args.zones,
            output_dir=args.output_dir,
            run_name=args.run_name,
            save_annotated=args.save_annotated,
            tracker=TrackerConfig(
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                tracker_cfg=args.tracker_cfg,
                device=args.device,
                max_history_len=args.max_history,
                max_missing_frames=args.max_missing_frames,
            ),
            spatial=SpatialConfig(
                proximity_threshold_px=args.proximity_threshold,
                zone_overlap_threshold=args.zone_overlap_threshold,
                use_bbox_for_zones=not args.centroid_only_zones,
            ),
            world_graph=WorldGraphConfig(min_track_frames=args.min_track_frames),
            rules=RuleEngineConfig(
                near_miss_distance_px=args.near_miss_distance_px,
                near_miss_min_consecutive_frames=args.near_miss_min_frames,
                near_miss_gap_tolerance_frames=args.near_miss_gap_tolerance_frames,
                ppe_overlap_threshold=args.ppe_overlap_threshold,
                loiter_min_seconds=args.loiter_min_seconds,
                min_exit_block_seconds=args.min_exit_block_seconds,
            ),
            narrate=args.narrate,
            narration=GeminiClientConfig(model=args.gemini_model),
        )
    except ValueError as e:
        logger.error(f"Invalid pipeline configuration: {e}")
        raise SystemExit(1)

    pipeline = ChainSightPipeline(config)
    try:
        result = pipeline.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Pipeline run failed: {e}")
        raise SystemExit(1)

    summary_lines = (
        "\nPipeline complete:\n"
        f"  manifest            -> {result.manifest_path}\n"
        f"  tracks              -> {result.tracks_path}\n"
        f"  spatial events      -> {result.spatial_events_path}\n"
        f"  world graph summary -> {result.world_graph_summary_path}\n"
        f"  rule events         -> {result.rule_events_path} ({len(result.rule_events)} event(s))"
    )
    if result.narration_path:
        summary_lines += f"\n  narration           -> {result.narration_path}"
    logger.info(summary_lines)


if __name__ == "__main__":
    main()
