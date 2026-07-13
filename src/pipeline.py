"""
pipeline.py — ChainSight single-entry orchestrator
Chains the four independently-runnable stages — tracker -> spatial ->
world_graph -> rules — into one call, instead of invoking four CLI scripts
by hand and keeping their config invariants in sync manually (e.g. R2's
near_miss_distance_px must stay <= spatial's proximity_threshold_px, or
R2 can structurally never fire — see rule_definitions.py).

Pipeline position:
    src/vision/tracker.py + src/spatial/*.py + src/world_graph/*.py + src/rules/*.py
        -> pipeline.py (THIS) -> scripts/run_pipeline.py (CLI) -> narration / demo

Design principle: thin orchestration only, no new business logic. Each
stage's existing class and *Config dataclass (ChainSightTracker/TrackerConfig,
SpatialAnalyzer/SpatialConfig, GraphBuilder/WorldGraphConfig,
RuleEngine/RuleEngineConfig) is reused as-is. Stage hand-off is still via
JSON files on disk (not in-memory objects) because GraphBuilder's
constructor is where tracks.json/spatial_events.json schema validation
happens (REQUIRED_TRACK_KEYS / REQUIRED_SPATIAL_EVENT_KEYS) — routing
through the same file boundary the CLI scripts use means upstream schema
drift is caught in the same place, not bypassed for a false sense of
speed.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .vision.tracker import ChainSightTracker, TrackerConfig
from .spatial import SpatialAnalyzer, SpatialConfig
from .world_graph import GraphBuilder, WorldGraphConfig, WorldQuery
from .rules import RuleEngine, RuleEngineConfig, RuleEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chainsight.pipeline")


@dataclass
class PipelineConfig:
    model_path: str
    video_path: str
    zones_path: str
    output_dir: str = "outputs"
    # Suffix applied to every output file, e.g. run_name="nearmiss" ->
    # outputs/tracks_nearmiss.json, matching this project's existing
    # per-clip naming convention (tracks_blocked_exit.json, etc.).
    run_name: Optional[str] = None
    save_annotated: bool = False

    tracker: Optional[TrackerConfig] = None
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    world_graph: WorldGraphConfig = field(default_factory=WorldGraphConfig)
    rules: RuleEngineConfig = field(default_factory=RuleEngineConfig)

    def __post_init__(self):
        if self.tracker is None:
            self.tracker = TrackerConfig(model_path=self.model_path)
        if self.rules.near_miss_distance_px > self.spatial.proximity_threshold_px:
            raise ValueError(
                f"rules.near_miss_distance_px ({self.rules.near_miss_distance_px}) must be "
                f"<= spatial.proximity_threshold_px ({self.spatial.proximity_threshold_px}) "
                f"— a 'near' proximity pair only exists in spatial_events.json within that "
                f"threshold at all, so R2 could never fire otherwise (see rule_definitions.py)."
            )


@dataclass
class PipelineResult:
    tracks_path: str
    spatial_events_path: str
    world_graph_summary_path: str
    rule_events_path: str
    rule_events: List[RuleEvent]


class ChainSightPipeline:
    """Runs tracker -> spatial -> world_graph -> rules end to end for one clip."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _out_path(self, stem: str, ext: str = "json") -> Path:
        suffix = f"_{self.config.run_name}" if self.config.run_name else ""
        return Path(self.config.output_dir) / f"{stem}{suffix}.{ext}"

    @staticmethod
    def _write_json(path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def run(self) -> PipelineResult:
        cfg = self.config
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # --- Stage 1: tracking ---
        tracker = ChainSightTracker(cfg.tracker)
        annotated_path = str(self._out_path("annotated", "mp4")) if cfg.save_annotated else None
        tracker.run(cfg.video_path, save_annotated=annotated_path)
        logger.info("\n" + tracker.summary())

        tracks_path = self._out_path("tracks")
        self._write_json(tracks_path, tracker.to_json_serializable())

        # --- Stage 2: spatial ---
        analyzer = SpatialAnalyzer(zones_path=cfg.zones_path, config=cfg.spatial)
        events_by_frame, zone_transitions = analyzer.analyze(str(tracks_path))
        logger.info("\n" + analyzer.summary(events_by_frame, zone_transitions))

        spatial_path = self._out_path("spatial_events")
        self._write_json(spatial_path, analyzer.to_json_serializable(events_by_frame, zone_transitions))

        # --- Stage 3: world graph ---
        builder = GraphBuilder(
            tracks_path=str(tracks_path), spatial_events_path=str(spatial_path), config=cfg.world_graph
        )
        logger.info(builder.summary())

        world_graph_path = self._out_path("world_graph_summary")
        summary_graph = builder.build_summary_graph()
        self._write_json(world_graph_path, builder.frame_graph_to_json(summary_graph))

        # --- Stage 4: rules ---
        query = WorldQuery(builder)
        engine = RuleEngine(query=query, zones_path=cfg.zones_path, config=cfg.rules)
        rule_events = engine.run()
        logger.info(engine.summary(rule_events))

        rule_events_path = self._out_path("rule_events")
        self._write_json(rule_events_path, [e.to_dict() for e in rule_events])

        return PipelineResult(
            tracks_path=str(tracks_path),
            spatial_events_path=str(spatial_path),
            world_graph_summary_path=str(world_graph_path),
            rule_events_path=str(rule_events_path),
            rule_events=rule_events,
        )
