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


def main():
    parser = argparse.ArgumentParser(description="Draw ChainSight zone polygons on a reference frame")
    parser.add_argument("--video", required=True, help="Path to a video to grab a reference frame from")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use as reference")
    parser.add_argument("--out", default="configs/zones.json", help="Output path for zones.json")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {args.frame} from video")
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