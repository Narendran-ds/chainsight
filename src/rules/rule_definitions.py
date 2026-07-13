"""
rule_definitions.py — ChainSight Rule Engine, shared config + geometry helpers
Pure, stateless helpers used by engine.py's rule implementations.
No graph/track state lives here — this module only knows geometry and
thresholds, so it can be unit-tested independently of tracker/spatial output.
"""

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class RuleEngineConfig:
    # --- Scenario 1: Restricted Zone Intrusion ---
    # Fires for any zone whose type is in this set (not hardcoded to
    # "forklift_lane" by name) so adding another restricted zone later
    # automatically gets this rule for free. Deliberately does NOT
    # include "exit" — people are supposed to use exits, so a person
    # in an exit-type zone must never be flagged as an intrusion.
    restricted_zone_types: Set[str] = field(default_factory=lambda: {"restricted"})

    # --- Scenario 2: Forklift-Pedestrian Near-Miss ---
    # NOTE: this must be <= spatial.py's SpatialConfig.proximity_threshold_px
    # (default 150px) — a "near" edge only exists in spatial_events.json at
    # all within that threshold, so setting this higher than that value
    # would silently never fire.
    near_miss_distance_px: float = 100.0
    near_miss_min_consecutive_frames: int = 10

    # --- Scenario 3: PPE Violation ---
    # class_name -> what it means when detected (the "missing PPE" classes)
    ppe_violation_classes: Set[str] = field(default_factory=lambda: {"no_vest", "no_helmet"})
    # Overlap ratio = intersection_area / ppe_bbox_area (asymmetric, NOT
    # symmetric IoU). A no_vest/no_helmet box is a small sub-region of the
    # person's box, so symmetric IoU is always small even for a correct
    # match — using the smaller (PPE) box as the denominator gives a much
    # more stable "is this PPE box basically inside this person box?" signal.
    ppe_overlap_threshold: float = 0.5

    # --- Scenario 4: Exit Blockage ---
    # Separate zone-type set from restricted_zone_types (see note above).
    exit_zone_types: Set[str] = field(default_factory=lambda: {"exit"})
    # Non-person, non-exit_zone_marker classes that can physically obstruct
    # an exit. Deliberately excludes "person" (a person standing near an
    # exit isn't a blockage) and "exit_zone_marker" (the sign itself).
    exit_blocking_classes: Set[str] = field(default_factory=lambda: {
        "box", "damaged_box", "open_box", "package",
        "pallet", "pallet_jack", "forklift", "small_load_carrier",
    })
    min_exit_block_seconds: float = 3.0

    # --- Scenario 5: Loitering in Restricted Zone ---
    loiter_min_seconds: float = 10.0


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """Standard symmetric IoU. [x1, y1, x2, y2] format. Provided for
    completeness / debugging — the PPE rule uses bbox_overlap_ratio instead
    (see RuleEngineConfig.ppe_overlap_threshold docstring for why)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def bbox_overlap_ratio(inner_box: List[float], outer_box: List[float]) -> float:
    """intersection_area / area(inner_box) — "how much of inner_box is
    covered by outer_box". Used to associate a small PPE-violation
    detection (e.g. no_vest) with the person bbox it most likely belongs
    to, since a straightforward centroid-nearest-neighbor match can
    misattribute PPE when two people stand close together."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter_area / inner_area if inner_area > 0 else 0.0
