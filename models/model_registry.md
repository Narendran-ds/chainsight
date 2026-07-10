| Run Name | Date | Base Model | Epochs | ImgSz | Batch | Train Time | mAP50 | mAP50-95 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| smoketest | 2026-07-07 | yolov8s.pt | 3 | 640 | 16 | 3.3min | 0.4106 | 0.2546 | Sanity check only |
| run1_baseline | 2026-07-07 | yolov8s.pt | 100 | 640 | 16 | 180.4min | 0.7447 | 0.4844 | 16-class, early-stop ep63, best ckpt ep48 |
| run2_exit_marker | 2026-07-09 | yolov8s.pt | 100 | 640 | 16 | 202.7min | 0.7678 | 0.5140 | 17-class, early-stop ep68, best ckpt ep53. See docs/scope_and_limitations.md |