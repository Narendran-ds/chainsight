"""
tracker.py — ChainSight Vision Layer, Stage 2
Wraps YOLOv8 detections with ByteTrack for persistent object IDs across frames.

Pipeline position:
    YOLOv8 (detection) -> tracker.py (THIS) -> spatial layer (Shapely) -> world graph (NetworkX)

Design principle (per ChainSight scope): tracker.py is deterministic, no-training-required.
Only the YOLOv8 detector is a trained model.

Improvements folded in (priority order):
  3  Timestamp (frame/fps) alongside frame index
  7  Auto device detection (cuda if available, else cpu)
  2  Velocity (vx, vy, speed) computed in Track.update()
  4  Object lifecycle (alive / lost / reappeared)
  5  Occlusion tracking (missing_frames, recovered)
  6  Precomputed Track properties (latest_bbox, age, duration, path_length, avg_speed)
  12 Richer JSON export (duration, first_seen, last_seen, avg_confidence, etc.)
  13 Logging instead of print()
  1  Memory control (history as flat arrays via deque, bounded but generous)
  8  Error handling around model.track()
  14 TrackerConfig dataclass instead of loose constructor args
"""

import json
import logging
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Logging setup (13)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chainsight.tracker")


# --------------------------------------------------------------------------- #
# Config (14)
# --------------------------------------------------------------------------- #
@dataclass
class TrackerConfig:
    model_path: str
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    tracker_cfg: str = "bytetrack.yaml"
    device: Optional[str] = None            # (7) auto-detected if None
    max_history_len: int = 5000              # (1) bounded per-track history
    max_missing_frames: int = 30             # (5) frames before marking "lost"

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device auto-detected: {self.device}")


# --------------------------------------------------------------------------- #
# Track (2, 4, 5, 6)
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    """Single tracked object's state across the video."""
    track_id: int
    class_id: int
    class_name: str
    fps: float
    max_history_len: int = 5000
    max_missing_frames: int = 30

    history: deque = field(default_factory=lambda: deque(maxlen=5000))  # (1)
    status: str = "alive"            # (4) alive | lost | reappeared
    missing_frames: int = 0          # (5)
    times_lost: int = 0              # (5) how many times it disappeared/reappeared
    _first_frame: Optional[int] = None
    _last_frame: Optional[int] = None
    _last_centroid: Optional[np.ndarray] = None
    _last_timestamp: Optional[float] = None
    _total_path_length: float = 0.0  # (6)
    _speed_sum: float = 0.0          # (6) for average speed
    _speed_count: int = 0

    def __post_init__(self):
        # deque maxlen must be set at construction; re-create with correct bound
        self.history = deque(maxlen=self.max_history_len)

    def update(self, frame_idx: int, bbox: np.ndarray, conf: float):
        """Record a new observation and update derived fields (velocity, lifecycle)."""
        x1, y1, x2, y2 = bbox
        centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
        timestamp = frame_idx / self.fps if self.fps > 0 else float(frame_idx)  # (3)

        # --- velocity (2) ---
        vx, vy, speed = 0.0, 0.0, 0.0
        if self._last_centroid is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt > 0:
                vx = (centroid[0] - self._last_centroid[0]) / dt
                vy = (centroid[1] - self._last_centroid[1]) / dt
                speed = float(np.hypot(vx, vy))
                dist = float(np.linalg.norm(centroid - self._last_centroid))
                self._total_path_length += dist
                self._speed_sum += speed
                self._speed_count += 1

        # --- lifecycle / occlusion recovery (4, 5) ---
        if self.status == "lost":
            self.status = "reappeared"
            self.times_lost += 1
            logger.info(f"Track {self.track_id} ({self.class_name}) reappeared "
                        f"after {self.missing_frames} missing frame(s)")
        elif self.status != "alive":
            self.status = "alive"
        self.missing_frames = 0

        entry = {
            "frame": frame_idx,
            "timestamp": round(timestamp, 4),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "centroid": [float(centroid[0]), float(centroid[1])],
            "conf": float(conf),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "speed": round(speed, 4),
        }
        self.history.append(entry)

        if self._first_frame is None:
            self._first_frame = frame_idx
        self._last_frame = frame_idx
        self._last_centroid = centroid
        self._last_timestamp = timestamp

    def mark_missing(self, current_frame: int):
        """Call once per frame the track is NOT detected, to track occlusion (5)."""
        if self._last_frame is None:
            return
        gap = current_frame - self._last_frame
        self.missing_frames = gap
        if self.status == "alive" and gap > 0:
            self.status = "lost" if gap <= self.max_missing_frames else "dead"
        elif self.status == "lost" and gap > self.max_missing_frames:
            self.status = "dead"

    # --- precomputed properties (6) ---
    @property
    def latest_bbox(self) -> Optional[List[float]]:
        return self.history[-1]["bbox"] if self.history else None

    @property
    def latest_confidence(self) -> Optional[float]:
        return self.history[-1]["conf"] if self.history else None

    @property
    def first_seen_frame(self) -> Optional[int]:
        return self._first_frame

    @property
    def last_seen_frame(self) -> Optional[int]:
        return self._last_frame

    @property
    def age_frames(self) -> int:
        if self._first_frame is None or self._last_frame is None:
            return 0
        return self._last_frame - self._first_frame + 1

    @property
    def duration_seconds(self) -> float:
        if not self.history:
            return 0.0
        return round(self.history[-1]["timestamp"] - self.history[0]["timestamp"], 4)

    @property
    def path_length(self) -> float:
        return round(self._total_path_length, 4)

    @property
    def average_speed(self) -> float:
        if self._speed_count == 0:
            return 0.0
        return round(self._speed_sum / self._speed_count, 4)

    @property
    def average_confidence(self) -> float:
        if not self.history:
            return 0.0
        return round(sum(h["conf"] for h in self.history) / len(self.history), 4)


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
class ChainSightTracker:
    """
    Runs YOLOv8 + built-in ByteTrack (via ultralytics .track()) and produces
    a per-object track history ready for the spatial layer.
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        try:
            self.model = YOLO(config.model_path)
        except Exception as e:  # (8)
            logger.error(f"Failed to load YOLO model from '{config.model_path}': {e}")
            raise

        self.tracks: Dict[int, Track] = {}
        self.class_names = self.model.names  # {id: name}

    def run(self, video_path: str, save_annotated: Optional[str] = None) -> Dict[int, Track]:
        """
        Process a video, populate self.tracks, and optionally save an annotated video.
        Returns dict of track_id -> Track.
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        fps = self._get_fps(video_path)
        writer = None
        if save_annotated:
            writer = self._make_writer(video_path, save_annotated, fps)

        try:
            results_gen = self.model.track(
                source=video_path,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                tracker=self.config.tracker_cfg,
                persist=True,
                stream=True,
                device=self.config.device,
                verbose=False,
            )
        except Exception as e:  # (8)
            logger.error(f"model.track() failed to initialize: {e}")
            if writer:
                writer.release()
            raise

        frame_idx = 0
        try:
            for result in results_gen:
                seen_ids = set()

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

                    for bbox, conf, cls_id, tid in zip(boxes, confs, cls_ids, track_ids):
                        tid = int(tid)
                        seen_ids.add(tid)
                        if tid not in self.tracks:
                            self.tracks[tid] = Track(
                                track_id=tid,
                                class_id=int(cls_id),
                                class_name=self.class_names[int(cls_id)],
                                fps=fps,
                                max_history_len=self.config.max_history_len,
                                max_missing_frames=self.config.max_missing_frames,
                            )
                            logger.info(f"New track {tid} ({self.class_names[int(cls_id)]}) "
                                        f"at frame {frame_idx}")
                        self.tracks[tid].update(frame_idx, bbox, conf)

                # any existing track not seen this frame -> mark missing (5)
                for tid, track in self.tracks.items():
                    if tid not in seen_ids and track.status in ("alive", "lost", "reappeared"):
                        track.mark_missing(frame_idx)

                if writer:
                    frame_out = result.plot() if (result.boxes is not None) else result.orig_img
                    writer.write(frame_out)

                frame_idx += 1

        except Exception as e:  # (8)
            logger.error(f"Error while processing frame {frame_idx}: {e}")
            raise
        finally:
            if writer:
                writer.release()

        logger.info(f"Finished processing {frame_idx} frames, {len(self.tracks)} tracks total.")
        return self.tracks

    # --- helpers ---
    def _get_fps(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps

    def _make_writer(self, video_path: str, out_path: str, fps: float):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # --- export (12) ---
    def to_json_serializable(self) -> dict:
        """Export tracks for spatial.py / world_graph.py consumption."""
        out = {}
        for tid, t in self.tracks.items():
            out[str(tid)] = {
                "class_id": t.class_id,
                "class_name": t.class_name,
                "status": t.status,
                "first_seen_frame": t.first_seen_frame,
                "last_seen_frame": t.last_seen_frame,
                "age_frames": t.age_frames,
                "duration_seconds": t.duration_seconds,
                "path_length": t.path_length,
                "average_speed": t.average_speed,
                "average_confidence": t.average_confidence,
                "times_lost": t.times_lost,
                "history": list(t.history),
            }
        return out

    def summary(self) -> str:
        lines = [f"Tracked {len(self.tracks)} objects:"]
        for tid, t in sorted(self.tracks.items()):
            lines.append(
                f"  ID {tid:>3} | {t.class_name:<20} | status={t.status:<10} | "
                f"frames {t.first_seen_frame}-{t.last_seen_frame} "
                f"({len(t.history)} obs) | avg_speed={t.average_speed} | "
                f"avg_conf={t.average_confidence}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChainSight Tracker (ByteTrack over YOLOv8)")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 .pt weights")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="outputs/tracks.json", help="Path to save track JSON")
    parser.add_argument("--annotated", default=None, help="Optional path to save annotated video")
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or leave unset for auto")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker-cfg", default="bytetrack.yaml")
    parser.add_argument("--max-history", type=int, default=5000)
    parser.add_argument("--max-missing-frames", type=int, default=30)
    args = parser.parse_args()

    config = TrackerConfig(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tracker_cfg=args.tracker_cfg,
        device=args.device,
        max_history_len=args.max_history,
        max_missing_frames=args.max_missing_frames,
    )

    tracker = ChainSightTracker(config)

    try:
        tracker.run(args.video, save_annotated=args.annotated)
    except Exception as e:
        logger.error(f"Tracking run failed: {e}")
        raise SystemExit(1)

    logger.info("\n" + tracker.summary())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(tracker.to_json_serializable(), f, indent=2)
    logger.info(f"Saved track data to {args.out}")