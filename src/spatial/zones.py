from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import logging

from shapely.geometry import Point, Polygon, box

logger = logging.getLogger("chainsight.spatial")


@dataclass
class SpatialConfig:
    # Calibrated at video_io.REFERENCE_FRAME_WIDTH (1920px). analyzer.py scales
    # this by the clip's actual frame_width (from tracks.json's "_meta", set by
    # tracker.py) before use, so it stays meaningful on 4K footage instead of
    # silently requiring 1000+px real distances to ever register as "near".
    proximity_threshold_px: float = 150.0
    zone_overlap_threshold: float = 0.1
    use_bbox_for_zones: bool = True


@dataclass
class Zone:
    name: str
    zone_type: str
    polygon: Polygon

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        try:
            poly = Polygon(d["polygon"])
            if not poly.is_valid or poly.area == 0:
                raise ValueError(f"Zone '{d.get('name', '?')}' polygon is invalid or has zero area")
            return cls(name=d["name"], zone_type=d["type"], polygon=poly)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Malformed zone definition {d}: {e}") from e


def load_zones(zones_path: str) -> List[Zone]:
    path = Path(zones_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Zones file not found: {zones_path}. Run define_zones.py first to create it."
        )
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"zones.json is not valid JSON: {e}") from e

    raw_zones = data.get("zones", [])
    zones = []
    for z in raw_zones:
        try:
            zones.append(Zone.from_dict(z))
        except ValueError as e:
            logger.error(f"Skipping invalid zone: {e}")
    if not zones:
        logger.warning("No valid zones loaded — zone containment will always be empty.")
    return zones