"""
extract_frame_grid.py — one-off debugging helper (not part of the pipeline)
Dumps one frame per second (or per N frames) from a video to a folder as
JPEGs, so you can eyeball many moments at once instead of guessing frame
numbers one at a time via define_zones.py.

Usage:
    python extract_frame_grid.py --video data\\staged_clips\\videos\\forklift_pedestrian_nearmiss.mp4 --every 25 --out data\\staged_clips\\extracted_frames\\nearmiss_preview
"""

import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--every", type=int, default=25, help="Save every Nth frame (default 25 = ~1/sec at 25fps)")
    parser.add_argument("--out", required=True, help="Output folder for extracted JPEGs")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.2f} fps (~{total_frames / fps:.1f}s)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.every == 0:
            # downscale for fast preview if frame is large (4K etc.)
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960 / w
                frame = cv2.resize(frame, (960, int(h * scale)))
            out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} preview frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
