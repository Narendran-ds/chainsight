"""
run_spatial.py — CLI entry point for the ChainSight spatial layer.
Usage:
    python scripts\run_spatial.py --tracks outputs\tracks.json --zones configs\zones.json --out outputs\spatial_events.json
"""

import sys
import json
import logging
import argparse
from pathlib import Path

# allow running from scripts/ without installing the package
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.spatial import SpatialAnalyzer, SpatialConfig

logger = logging.getLogger("chainsight.spatial.cli")


def main():
    parser = argparse.ArgumentParser(description="ChainSight Spatial Layer (Shapely zone + proximity analysis)")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json from tracker.py")
    parser.add_argument("--zones", required=True, help="Path to zones.json from define_zones.py")
    parser.add_argument("--out", default="outputs/spatial_events.json", help="Path to save spatial events JSON")
    parser.add_argument("--proximity-threshold", type=float, default=150.0,
                         help="Pixel distance under which two tracks are considered 'nearby'")
    parser.add_argument("--zone-overlap-threshold", type=float, default=0.1,
                         help="Minimum fraction of an object's bbox area that must overlap a zone to count as 'inside'")
    parser.add_argument("--centroid-only-zones", action="store_true",
                         help="Use legacy centroid-in-polygon zone check instead of bbox overlap")
    args = parser.parse_args()

    config = SpatialConfig(
        proximity_threshold_px=args.proximity_threshold,
        zone_overlap_threshold=args.zone_overlap_threshold,
        use_bbox_for_zones=not args.centroid_only_zones,
    )

    try:
        analyzer = SpatialAnalyzer(zones_path=args.zones, config=config)
        events_by_frame, zone_transitions = analyzer.analyze(args.tracks)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Spatial analysis failed: {e}")
        raise SystemExit(1)

    logger.info("\n" + analyzer.summary(events_by_frame, zone_transitions))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(analyzer.to_json_serializable(events_by_frame, zone_transitions), f, indent=2)
    logger.info(f"Saved spatial events to {args.out}")


if __name__ == "__main__":
    main()