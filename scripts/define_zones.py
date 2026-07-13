"""
define_zones.py — ChainSight Spatial Layer, Setup Tool
Interactively draw zone polygons on a reference video frame by clicking points.

Usage:
    python define_zones.py --video path\to\clip.mp4 --frame 0 --out configs\zones.json

Controls:
    Left click   -> add a point to the current zone polygon
    'n'          -> finish current zone, start a new one (will prompt for name/type in terminal)
    's'          -> save all zones to the output JSON and exit
    'u'          -> undo last point
    'q'          -> quit without saving

Output format (zones.json):
{
  "zones": [
    {
      "name": "exit_zone_north",
      "type": "exit",
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    },
    {
      "name": "restricted_forklift_lane",
      "type": "restricted",
      "polygon": [[x1,y1], ...]
    }
  ],
  "frame_width": 768,
  "frame_height": 432
}
"""

import json
import argparse
from pathlib import Path

import cv2


class ZoneDrawer:
    def __init__(self, frame):
        self.frame = frame
        self.display = frame.copy()
        self.zones = []
        self.current_points = []

    def redraw(self):
        self.display = self.frame.copy()
        # draw completed zones
        for zone in self.zones:
            pts = zone["polygon"]
            for i in range(len(pts)):
                cv2.line(self.display, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            cv2.putText(self.display, zone["name"], (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        # draw current in-progress polygon
        for i, pt in enumerate(self.current_points):
            cv2.circle(self.display, tuple(pt), 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(self.display, tuple(self.current_points[i - 1]), tuple(pt), (0, 0, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self.redraw()

    def finish_zone(self):
        if len(self.current_points) < 3:
            print("Need at least 3 points to make a zone. Ignoring.")
            return
        name = input("Zone name (e.g. exit_zone_north): ").strip() or f"zone_{len(self.zones)}"
        zone_type = input("Zone type (exit / restricted / staging / other): ").strip() or "other"
        self.zones.append({
            "name": name,
            "type": zone_type,
            "polygon": self.current_points.copy(),
        })
        print(f"Saved zone '{name}' with {len(self.current_points)} points.")
        self.current_points = []
        self.redraw()

    def undo_point(self):
        if self.current_points:
            self.current_points.pop()
            self.redraw()


def read_frame_sequentially(cap: cv2.VideoCapture, target_frame: int):
    """
    Reads frames one at a time up to target_frame, instead of using
    cap.set(cv2.CAP_PROP_POS_FRAMES, ...) to seek directly.

    Rationale: CAP_PROP_POS_FRAMES seeking is unreliable on many H.264
    encodes (especially compressed stock-footage downloads with sparse
    keyframes) — OpenCV can silently fail the seek and just return an
    early frame regardless of the requested index. Sequential reading
    is slower for large target_frame values but always correct
    regardless of codec/keyframe layout.
    """
    frame = None
    for i in range(target_frame + 1):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"Video ended at frame {i} while seeking to frame {target_frame} "
                f"(video may be shorter than expected, or corrupted)."
            )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Draw ChainSight zone polygons on a reference frame")
    parser.add_argument("--video", required=True, help="Path to a video to grab a reference frame from")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use as reference")
    parser.add_argument("--out", default="configs/zones.json", help="Output path for zones.json")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} total frames, {fps:.2f} fps "
          f"(~{total_frames / fps:.1f}s) — reading sequentially to frame {args.frame}...")

    frame = read_frame_sequentially(cap, args.frame)
    h, w = frame.shape[:2]
    cap.release()

    drawer = ZoneDrawer(frame)
    window_name = "ChainSight Zone Editor - click points, 'n' new zone, 's' save, 'u' undo, 'q' quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, drawer.mouse_callback)

    print("\n--- ChainSight Zone Editor ---")
    print("Left click: add point | 'n': finish zone | 'u': undo point | 's': save & exit | 'q': quit\n")

    while True:
        cv2.imshow(window_name, drawer.display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n'):
            drawer.finish_zone()
        elif key == ord('u'):
            drawer.undo_point()
        elif key == ord('s'):
            if drawer.current_points:
                print("Finishing in-progress zone before saving...")
                drawer.finish_zone()
            break
        elif key == ord('q'):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    output = {
        "zones": drawer.zones,
        "frame_width": w,
        "frame_height": h,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(drawer.zones)} zone(s) to {args.out}")


if __name__ == "__main__":
    main()
