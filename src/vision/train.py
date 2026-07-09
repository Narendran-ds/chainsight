"""
ChainSight — YOLOv8 Fine-Tuning Script
=========================================
Fine-tunes a pretrained YOLOv8 checkpoint on the merged ChainSight dataset.
Automatically logs the run to Weights & Biases if wandb is installed and
you've run `wandb login` beforehand.

Usage:
    python src/vision/train.py --run-name run1_baseline
    python src/vision/train.py --run-name run2_50epoch --epochs 50 --imgsz 640
    python src/vision/train.py --run-name run3_final --epochs 100 --model yolov8s.pt --batch 16
    python src/vision/train.py --run-name check --dry-run          # sanity check only, no training

Before running:
    1. wandb login          (one-time, paste your API key when prompted)
    2. yolo settings wandb=True
    3. Confirm data.yaml `path:` is an ABSOLUTE path — Ultralytics resolves
       relative paths against your current working directory, not the yaml's
       location, which silently breaks things if you run from a different folder.

Resuming an interrupted run:
    Only pass --resume if a run folder with the SAME --run-name already exists
    under models/finetuned/ with a saved last.pt checkpoint.
"""

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "data" / "processed" / "chainsight_dataset" / "data.yaml"
PRETRAINED_DIR = REPO_ROOT / "models" / "pretrained"
FINETUNED_DIR = REPO_ROOT / "models" / "finetuned"
REGISTRY_PATH = REPO_ROOT / "models" / "model_registry.md"

LOW_SUPPORT_CLASSES = ("damaged_box", "open_box", "no_vest", "exit_zone_marker")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[SEED] Random seed set to {seed} (random, numpy, torch, cuda)")


# ------------------------------------------------------------------------------
# Environment logging
# ------------------------------------------------------------------------------
def get_environment_info():
    info = {
        "python_version": str(sys.version.split()[0]),
        "torch_version": str(torch.__version__),  # torch.__version__ is a TorchVersion
        # (str subclass) in newer torch releases; PyYAML's safe dumper does an
        # exact-type lookup and doesn't recognize subclasses, so cast explicitly.
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": str(platform.platform()),
    }
    try:
        import ultralytics
        info["ultralytics_version"] = str(ultralytics.__version__)
    except Exception:
        info["ultralytics_version"] = "unknown"

    if torch.cuda.is_available():
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        info["cuda_version"] = str(torch.version.cuda)
        info["gpu_memory_total_mb"] = int(round(torch.cuda.get_device_properties(0).total_memory / 1024**2))
    else:
        info["gpu_name"] = "N/A (CPU)"
        info["cuda_version"] = "N/A"
        info["gpu_memory_total_mb"] = "N/A"

    return info


def print_environment_info(info):
    print("[ENV] Environment snapshot:")
    for k, v in info.items():
        print(f"       {k}: {v}")


# ------------------------------------------------------------------------------
# Device validation
# ------------------------------------------------------------------------------
def validate_device(device_arg: str):
    if device_arg.lower() == "cpu":
        print("[DEVICE] Training on CPU as requested. This will be SLOW for this dataset size.")
        return
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError(
            f"--device {device_arg} was requested but no CUDA devices are visible to torch. "
            f"Check `torch.cuda.is_available()` and your torch install (needs a +cuXXX build, not +cpu)."
        )
    requested_ids = [int(x) for x in device_arg.split(",")]
    for rid in requested_ids:
        if rid >= available:
            raise RuntimeError(
                f"--device {device_arg} requests GPU index {rid}, but only {available} "
                f"CUDA device(s) are visible (indices 0..{available - 1})."
            )
    print(f"[DEVICE] Validated device(s) {device_arg} against {available} visible CUDA device(s).")


# ------------------------------------------------------------------------------
# Dataset sanity check (also powers --dry-run)
# ------------------------------------------------------------------------------
def check_dataset_stats(data_yaml_path: Path):
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = data_yaml_path.parent / root

    print(f"[DATA] Resolved dataset root: {root}")
    nc = cfg.get("nc")
    names = cfg.get("names", {})
    print(f"[DATA] Classes declared: {nc}")

    splits = {}
    for split_key in ("train", "val", "test"):
        rel = cfg.get(split_key)
        if not rel:
            continue
        img_dir = root / rel
        if not img_dir.exists():
            print(f"[DATA] ⚠ {split_key}: MISSING directory {img_dir}")
            splits[split_key] = {"images": 0, "labels": 0, "missing_labels": 0, "empty_images": 0}
            continue

        images = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        label_dir = Path(str(img_dir).replace("images", "labels"))
        missing_labels = 0
        empty_images = 0
        for img_path in images:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                missing_labels += 1
            elif label_path.stat().st_size == 0:
                empty_images += 1

        splits[split_key] = {
            "images": len(images),
            "labels": len(images) - missing_labels,
            "missing_labels": missing_labels,
            "empty_images": empty_images,
        }
        print(f"[DATA] {split_key:5s}: {len(images):5d} images | "
              f"{missing_labels} missing labels | {empty_images} empty label files")

    if splits.get("train", {}).get("images", 0) == 0:
        raise FileNotFoundError(
            "[DATA] train split has 0 images — check data.yaml `path:` is absolute and correct "
            "before proceeding. See module docstring."
        )
    if splits.get("val", {}).get("images", 0) == 0:
        raise FileNotFoundError(
            "[DATA] val split has 0 images — check data.yaml `path:` is absolute and correct "
            "before proceeding. See module docstring."
        )

    return {"root": str(root), "nc": nc, "names": names, "splits": splits}


# ------------------------------------------------------------------------------
# Config saving
# ------------------------------------------------------------------------------
def save_run_config(run_dir: Path, args, env_info, dataset_stats):
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_name": args.run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "device": args.device,
            "workers": args.workers,
            "cache": args.cache,
            "seed": args.seed,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "optimizer": args.optimizer,
            "amp": args.amp,
            "resume": args.resume,
        },
        "environment": env_info,
        "dataset": dataset_stats,
    }
    config_path = run_dir / "config.yaml"
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"[CONFIG] Saved run config to {config_path}")
    except yaml.representer.RepresenterError as e:
        # Config saving should never be able to block/crash an actual training run —
        # fall back to JSON (str() on everything) rather than losing GPU time over this.
        fallback_path = run_dir / "config.json"
        with open(fallback_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"[CONFIG] YAML dump failed ({e}); saved as JSON fallback instead: {fallback_path}")


# ------------------------------------------------------------------------------
# Registry logging
# ------------------------------------------------------------------------------
def append_to_registry(run_name, args, results, duration_sec):
    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    header_needed = not REGISTRY_PATH.exists()

    with open(REGISTRY_PATH, "a") as f:
        if header_needed:
            f.write("# ChainSight — Model Registry\n\n")
            f.write("| Run Name | Date | Base Model | Epochs | Img Size | Batch | Duration | mAP50 | mAP50-95 | Notes |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        try:
            map50 = f"{results.box.map50:.4f}"
            map5095 = f"{results.box.map:.4f}"
        except Exception:
            map50 = "N/A"
            map5095 = "N/A"

        date_str = datetime.now().strftime("%Y-%m-%d")
        duration_str = f"{duration_sec / 60:.1f}min"

        f.write(
            f"| {run_name} | {date_str} | {args.model} | {args.epochs} | {args.imgsz} | "
            f"{args.batch} | {duration_str} | {map50} | {map5095} | |\n"
        )

    print(f"\n[LOGGED] Run summary appended to {REGISTRY_PATH}")


def check_resume_validity(run_name, resume_requested):
    run_dir = FINETUNED_DIR / run_name
    if resume_requested and not run_dir.exists():
        raise FileNotFoundError(
            f"--resume was passed but no existing run folder found at {run_dir}. "
            f"Remove --resume to start a new run."
        )
    if resume_requested:
        last_ckpt = run_dir / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"--resume was passed but no checkpoint found at {last_ckpt}.")
        print(f"[RESUME] Found checkpoint at {last_ckpt} — continuing run '{run_name}'")


def check_wandb_ready():
    try:
        import wandb  # noqa: F401
        print("[W&B] wandb package detected. If logs don't appear on your dashboard, "
              "run `wandb login` and `yolo settings wandb=True` before training.")
    except ImportError:
        print("[W&B] wandb not installed — training will proceed WITHOUT experiment tracking.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction (lr0 * lrf)")
    parser.add_argument("--optimizer", type=str, default="auto",
                         choices=["auto", "SGD", "Adam", "AdamW", "RMSProp"])
    parser.add_argument("--amp", type=lambda x: x.lower() != "false", default=True,
                         help="Mixed precision training. Pass --amp false to disable "
                              "(e.g. on older GPUs with poor fp16 support).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate dataset, device, and config only — exits before training starts.")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}.")

    print("=" * 70)
    env_info = get_environment_info()
    print_environment_info(env_info)
    print("=" * 70)

    validate_device(args.device)
    set_seed(args.seed)
    dataset_stats = check_dataset_stats(DATA_YAML)

    print("=" * 70)
    for split, stats in dataset_stats["splits"].items():
        for cname in LOW_SUPPORT_CLASSES:
            pass  # per-class instance counts require label parsing; AP50 check post-training still covers this

    if args.dry_run:
        print("\n[DRY RUN] All checks passed. Exiting without training (--dry-run was set).")
        return

    check_resume_validity(args.run_name, args.resume)
    check_wandb_ready()

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training run: {args.run_name}")
    run_dir = FINETUNED_DIR / args.run_name
    save_run_config(run_dir, args, env_info, dataset_stats)

    start_time = time.time()
    interrupted = False
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            lr0=args.lr0,
            lrf=args.lrf,
            optimizer=args.optimizer,
            amp=args.amp,
            seed=args.seed,
            project=str(FINETUNED_DIR),
            name=args.run_name,
            resume=args.resume,
            exist_ok=True,
        )
    except KeyboardInterrupt:
        interrupted = True
        duration = time.time() - start_time
        print(f"\n[INTERRUPTED] Training stopped by user after {duration / 60:.1f} min.")
        print(f"[INTERRUPTED] Partial checkpoints (if any) should be under {run_dir / 'weights'}")
        print(f"[INTERRUPTED] To resume: python src/vision/train.py --run-name {args.run_name} --resume ...")
        return
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[ERROR] Training failed after {duration / 60:.1f} min: {e}")
        raise
    finally:
        if not interrupted:
            pass  # normal completion falls through below

    duration = time.time() - start_time
    print(f"\nTraining complete in {duration / 60:.1f} min. Best weights: {run_dir / 'weights' / 'best.pt'}")

    # Reload best.pt explicitly rather than trusting the in-memory model object,
    # so the registry always reflects the actual best checkpoint, not just
    # whatever state training happened to end on.
    best_ckpt = run_dir / "weights" / "best.pt"
    if best_ckpt.exists():
        print(f"\nReloading {best_ckpt} for final validation...")
        best_model = YOLO(str(best_ckpt))
    else:
        print(f"\n[WARN] best.pt not found at {best_ckpt}, falling back to in-memory model for validation.")
        best_model = model

    val_results = best_model.val(data=str(DATA_YAML))
    append_to_registry(args.run_name, args, val_results, duration)

    print("\n---- Per-class results (best.pt) ----")
    try:
        names = best_model.names
        # IMPORTANT: val_results.box.ap50 only contains rows for classes that had
        # >=1 ground-truth instance in this val set. Classes with 0 instances are
        # silently dropped from the array, which SHIFTS every subsequent index if
        # you naively enumerate() it. Use ap_class_index to get the real class id
        # for each row instead of trusting array position.
        ap_class_ids = val_results.box.ap_class_index  # actual class id per row
        ap50_values = val_results.box.ap50

        reported_ids = set(int(c) for c in ap_class_ids)
        for row_idx, class_id in enumerate(ap_class_ids):
            class_id = int(class_id)
            class_name = names.get(class_id, f"class_{class_id}")
            ap = ap50_values[row_idx]
            flag = "  <-- LOW SUPPORT (flagged during data prep)" if class_name in LOW_SUPPORT_CLASSES else ""
            print(f"  {class_id:2d} {class_name:20s} AP50={ap:.4f}{flag}")

        # Explicitly call out classes with ZERO val instances — these aren't in
        # ap_class_index at all, and silently omitting them from the printout
        # (as before) hid a real coverage gap rather than surfacing it.
        all_ids = set(names.keys())
        zero_instance_ids = sorted(all_ids - reported_ids)
        if zero_instance_ids:
            print("\n  Classes with ZERO instances in this val set (no AP50 computed):")
            for class_id in zero_instance_ids:
                class_name = names.get(class_id, f"class_{class_id}")
                flag = "  <-- LOW SUPPORT (flagged during data prep)" if class_name in LOW_SUPPORT_CLASSES else ""
                print(f"  {class_id:2d} {class_name:20s} (no instances){flag}")
    except Exception as e:
        print(f"  (Per-class breakdown unavailable: {e})")


if __name__ == "__main__":
    main()