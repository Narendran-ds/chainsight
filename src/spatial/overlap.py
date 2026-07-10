from typing import List
from shapely.geometry import Point, box

from .zones import Zone, SpatialConfig


def zones_containing_bbox(bbox: List[float], zones: List[Zone], config: SpatialConfig) -> List[str]:
    if not zones:
        return []

    if not config.use_bbox_for_zones:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        pt = Point(cx, cy)
        return [z.name for z in zones if z.polygon.contains(pt)]

    obj_box = box(bbox[0], bbox[1], bbox[2], bbox[3])
    obj_area = obj_box.area
    if obj_area == 0:
        return []

    inside = []
    for z in zones:
        if not obj_box.intersects(z.polygon):
            continue
        overlap_area = obj_box.intersection(z.polygon).area
        if (overlap_area / obj_area) >= config.zone_overlap_threshold:
            inside.append(z.name)
    return inside