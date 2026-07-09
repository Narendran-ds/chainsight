"""
ChainSight — Dataset Preparation Script
=========================================
Filters the Arvist dataset (removes traffic-only images), remaps all 3
source datasets' class IDs to the ChainSight master class list, and merges
everything into data/processed/chainsight_dataset/ in standard YOLOv8 format.

Usage:
    python scripts/prepare_data.py --report        # dry-run, just print stats
    python scripts/prepare_data.py                  # actually perform the merge

Requires:
    data/class_mapping.yaml
    data/raw/arvist/{train,valid,test}/{images,labels}
    data/raw/forklift/{train,valid,test}/{images,labels}
    data/raw/pallet/{train,valid,test}/{images,labels}
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed" / "chainsight_dataset"
MAPPING_PATH = REPO_ROOT / "data" / "class_mapping.yaml"
DATASET_KEYS = ["arvist", "forklift", "pallet", "exit_dataset_1", "exit_dataset_2"]
# YOLOv8 export splits are usually named train/valid/test — normalize to train/val/test
SPLIT_NAME_MAP = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def load_mapping():
    with open(MAPPING_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    master_classes = cfg["master_classes"]
    name_to_id = {v: k for k, v in master_classes.items()}
    return cfg, name_to_id


def load_source_class_list(dataset_key, cfg):
    original_classes = cfg["source_datasets"][dataset_key]["original_classes"]
    return {i: name for i, name in enumerate(original_classes)}


def build_remap_table(dataset_key, cfg, name_to_id):
    """
    Returns dict: original_class_id (int) -> master_class_id (int) or None
    None means this class is excluded and any box with it must be dropped.
    """
    source_classes = load_source_class_list(dataset_key, cfg)
    mapping = cfg["source_datasets"][dataset_key]["mapping"]
    excluded = set(cfg["source_datasets"][dataset_key].get("excluded_classes", []))

    remap = {}
    for orig_id, orig_name in source_classes.items():
        if orig_name in excluded:
            remap[orig_id] = None
        elif orig_name in mapping:
            master_name = mapping[orig_name]
            remap[orig_id] = name_to_id[master_name]
        else:
            raise ValueError(
                f"Class '{orig_name}' (id={orig_id}) in dataset '{dataset_key}' "
                f"has no mapping and is not marked excluded. Fix class_mapping.yaml."
            )
    return remap


def check_split_completeness(dataset_key, dataset_root):
    """
    Sanity-check each split's images/ vs labels/ folder before processing.
    Catches partial/broken downloads (missing splits, image count far
    exceeding label count, etc.) loudly and immediately, instead of letting
    every image silently fall into dropped_no_labels with no signal that
    something is wrong with the SOURCE DATA rather than normal filtering.
    """
    problems = []
    for src_split in ["train", "valid", "test"]:
        images_dir = dataset_root / src_split / "images"
        labels_dir = dataset_root / src_split / "labels"

        if not images_dir.exists():
            problems.append(f"  '{src_split}': images/ folder MISSING entirely at {images_dir}")
            continue

        n_images = sum(1 for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

        if not labels_dir.exists():
            problems.append(
                f"  '{src_split}': {n_images} images found but labels/ folder MISSING entirely "
                f"at {labels_dir} — every image in this split will be silently dropped."
            )
            continue

        n_labels = sum(1 for _ in labels_dir.glob("*.txt"))

        if n_images == 0:
            problems.append(f"  '{src_split}': 0 images found at {images_dir} (empty or missing split)")
        elif n_labels == 0:
            problems.append(
                f"  '{src_split}': {n_images} images but 0 label files at {labels_dir} — "
                f"this split has NO usable annotations."
            )
        elif n_labels < n_images * 0.5:
            problems.append(
                f"  '{src_split}': {n_images} images but only {n_labels} label files "
                f"({n_labels / n_images * 100:.0f}% coverage) — likely a partial/broken download, "
                f"not normal missing-annotation noise."
            )

    if problems:
        print(f"\n[DATA WARNING] '{dataset_key}' has split-completeness issues:")
        for p in problems:
            print(p)
        print(
            f"  This usually means an interrupted or partial download from the source, "
            f"not a class-mapping bug. Verify against the dataset's official split counts "
            f"(e.g. the Roboflow Universe page) before trusting merge output for '{dataset_key}'.\n"
        )
    return problems


def process_label_file(label_path, remap_table):
    """
    Reads a YOLO label .txt file, remaps class IDs, drops excluded boxes.
    Returns (new_lines, kept_any_box, had_only_excluded).
    """
    if not label_path.exists():
        return [], False, False

    new_lines = []
    had_any_line = False
    kept_any = False

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            had_any_line = True
            parts = line.split()
            orig_class_id = int(parts[0])
            new_class_id = remap_table.get(orig_class_id)
            if new_class_id is None:
                continue  # excluded class — drop this box
            kept_any = True
            new_lines.append(" ".join([str(new_class_id)] + parts[1:]))

    had_only_excluded = had_any_line and not kept_any
    return new_lines, kept_any, had_only_excluded


def process_dataset(dataset_key, cfg, name_to_id, report_only, counters, completeness_issues):
    remap_table = build_remap_table(dataset_key, cfg, name_to_id)
    dataset_root = RAW_DIR / dataset_key

    problems = check_split_completeness(dataset_key, dataset_root)
    if problems:
        completeness_issues[dataset_key] = problems

    for src_split in ["train", "valid", "test"]:
        images_dir = dataset_root / src_split / "images"
        labels_dir = dataset_root / src_split / "labels"

        if not images_dir.exists():
            print(f"[WARN] {images_dir} does not exist — skipping split '{src_split}' for '{dataset_key}'")
            continue

        dst_split = SPLIT_NAME_MAP[src_split]
        out_images_dir = OUT_DIR / dst_split / "images"
        out_labels_dir = OUT_DIR / dst_split / "labels"

        if not report_only:
            out_images_dir.mkdir(parents=True, exist_ok=True)
            out_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(images_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = labels_dir / (image_path.stem + ".txt")
            new_lines, kept_any, had_only_excluded = process_label_file(label_path, remap_table)

            counters["total_images_seen"][dataset_key] += 1

            if had_only_excluded:
                counters["dropped_traffic_only"][dataset_key] += 1
                continue

            if not kept_any:
                counters["dropped_no_labels"][dataset_key] += 1
                continue

            counters["kept_images"][dataset_key] += 1
            for line in new_lines:
                class_id = int(line.split()[0])
                counters["class_instance_counts"][class_id] += 1

            if not report_only:
                new_stem = f"{dataset_key}_{image_path.stem}"
                shutil.copy(image_path, out_images_dir / f"{new_stem}{image_path.suffix}")
                with open(out_labels_dir / f"{new_stem}.txt", "w") as f:
                    f.write("\n".join(new_lines) + ("\n" if new_lines else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Dry run — print stats only, do not write files")
    parser.add_argument("--ignore-completeness-warnings", action="store_true",
                         help="Proceed with merge even if split-completeness check flags issues. "
                              "Off by default — you should look at the warning before merging.")
    args = parser.parse_args()

    cfg, name_to_id = load_mapping()
    id_to_name = {v: k for k, v in name_to_id.items()}

    counters = {
        "total_images_seen": Counter(),
        "kept_images": Counter(),
        "dropped_traffic_only": Counter(),
        "dropped_no_labels": Counter(),
        "class_instance_counts": Counter(),
    }
    completeness_issues = {}

    for dataset_key in DATASET_KEYS:
        print(f"Processing dataset: {dataset_key} ...")
        process_dataset(dataset_key, cfg, name_to_id, args.report, counters, completeness_issues)

    if completeness_issues and not args.ignore_completeness_warnings and not args.report:
        print("\n" + "=" * 70)
        print("[STOPPED] Split-completeness warnings were found (see above) and this is "
              "NOT a --report dry run. Refusing to write a merge built on incomplete source data.")
        print("Fix the source data (e.g. re-download the affected dataset) and re-run, "
              "or pass --ignore-completeness-warnings if you are certain this is expected.")
        print("=" * 70)
        return

    print("\n========== SUMMARY ==========")
    for dataset_key in DATASET_KEYS:
        print(
            f"{dataset_key:10s} | seen: {counters['total_images_seen'][dataset_key]:5d} | "
            f"kept: {counters['kept_images'][dataset_key]:5d} | "
            f"dropped_traffic_only: {counters['dropped_traffic_only'][dataset_key]:5d} | "
            f"dropped_no_labels: {counters['dropped_no_labels'][dataset_key]:5d}"
        )

    print("\n---- Per-class instance counts (post-merge) ----")
    for class_id in sorted(id_to_name):
        count = counters["class_instance_counts"].get(class_id, 0)
        flag = "  <-- LOW SUPPORT" if count < 50 else ""
        print(f"{class_id:2d} {id_to_name[class_id]:20s}: {count:6d}{flag}")

    if args.report:
        print("\n[DRY RUN] No files were written. Re-run without --report to perform the merge.")
    else:
        print(f"\nMerge complete. Output written to: {OUT_DIR}")


if __name__ == "__main__":
    main()