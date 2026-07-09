"""
ChainSight — Full Train/Val/Test Leakage-Safe Split Script
=========================================================================
Replaces split_val.py. That script only ever recombined+re-split train/val,
leaving test/ untouched under the assumption Roboflow's original test split
was already leakage-safe. It wasn't: verify_no_leakage.py found 350+49
Arvist source images duplicated across train/test and val/test, because
Roboflow's own original split was done at the FILE level, not the grouped
source-image level.

This script fixes it properly by recombining ALL THREE splits back into one
pool, then re-splitting GROUPS (not files) into train/val/test in one pass.
Every augmented '.rf.<hash>' variant of a given source image is guaranteed
to land in exactly one split.

Usage:
    python scripts/split_three_way.py --val-ratio 0.15 --test-ratio 0.08
    python scripts/split_three_way.py --val-ratio 0.15 --test-ratio 0.08 --seed 42
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "processed" / "chainsight_dataset"

VALID_EXTS = {".jpg", ".jpeg", ".png"}
RF_SUFFIX_PATTERN = re.compile(r"\.rf\..*$")

SPLITS = ["train", "val", "test"]


def base_key(filename_stem: str) -> str:
    return RF_SUFFIX_PATTERN.sub("", filename_stem)


def split_dirs(split):
    return (
        DATASET_ROOT / split / "images",
        DATASET_ROOT / split / "labels",
    )


def recombine_all_into_train():
    """Move every image+label from val/ and test/ back into train/,
    so we start from one unified pool before re-splitting."""
    train_images_dir, train_labels_dir = split_dirs("train")
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    total_moved = 0
    for split in ["val", "test"]:
        images_dir, labels_dir = split_dirs(split)
        if not images_dir.exists():
            continue
        moved = 0
        for image_path in list(images_dir.glob("*")):
            if image_path.suffix.lower() not in VALID_EXTS:
                continue
            shutil.move(str(image_path), str(train_images_dir / image_path.name))
            label_path = labels_dir / (image_path.stem + ".txt")
            if label_path.exists():
                shutil.move(str(label_path), str(train_labels_dir / label_path.name))
            moved += 1
        if moved:
            print(f"[RECOMBINE] Moved {moved} images (+ labels) from {split}/ back into train/.")
        total_moved += moved
    return total_moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.15,
                         help="Fraction of GROUPS assigned to val (default 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.08,
                         help="Fraction of GROUPS assigned to test (default 0.08)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_ratio + args.test_ratio >= 1.0:
        print("[ERROR] val-ratio + test-ratio must be < 1.0")
        return

    random.seed(args.seed)

    recombine_all_into_train()

    train_images_dir, train_labels_dir = split_dirs("train")
    val_images_dir, val_labels_dir = split_dirs("val")
    test_images_dir, test_labels_dir = split_dirs("test")
    for d in [val_images_dir, val_labels_dir, test_images_dir, test_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_images = sorted(train_images_dir.glob("*"))
    all_images = [p for p in all_images if p.suffix.lower() in VALID_EXTS]

    if not all_images:
        print(f"[ERROR] No images found in {train_images_dir}. Did you run prepare_data.py first?")
        return

    groups = defaultdict(list)
    for image_path in all_images:
        key = base_key(image_path.stem)
        groups[key].append(image_path)

    group_keys = sorted(groups.keys())
    n_groups_total = len(group_keys)
    n_files_total = len(all_images)
    multi_member_groups = sum(1 for k in group_keys if len(groups[k]) > 1)

    print(f"[GROUPING] {n_files_total} files grouped into {n_groups_total} unique source images "
          f"({multi_member_groups} groups have multiple augmented variants).")

    random.shuffle(group_keys)
    n_val_groups = int(n_groups_total * args.val_ratio)
    n_test_groups = int(n_groups_total * args.test_ratio)

    val_keys = set(group_keys[:n_val_groups])
    test_keys = set(group_keys[n_val_groups:n_val_groups + n_test_groups])
    # everything else stays in train

    def move_groups(keys, dst_images_dir, dst_labels_dir):
        file_count = 0
        missing_labels = 0
        for key in keys:
            for image_path in groups[key]:
                label_path = train_labels_dir / (image_path.stem + ".txt")
                shutil.move(str(image_path), str(dst_images_dir / image_path.name))
                if label_path.exists():
                    shutil.move(str(label_path), str(dst_labels_dir / label_path.name))
                else:
                    missing_labels += 1
                file_count += 1
        return file_count, missing_labels

    val_file_count, val_missing = move_groups(val_keys, val_images_dir, val_labels_dir)
    test_file_count, test_missing = move_groups(test_keys, test_images_dir, test_labels_dir)

    remaining_files = n_files_total - val_file_count - test_file_count

    print("========== SPLIT SUMMARY (three-way, leakage-safe) ==========")
    print(f"Total source-image groups: {n_groups_total}")
    print(f"Groups -> val:  {len(val_keys)} ({len(val_keys)/n_groups_total*100:.1f}%)")
    print(f"Groups -> test: {len(test_keys)} ({len(test_keys)/n_groups_total*100:.1f}%)")
    print(f"Groups -> train: {n_groups_total - len(val_keys) - len(test_keys)}")
    print()
    print(f"Files -> val:   {val_file_count}")
    print(f"Files -> test:  {test_file_count}")
    print(f"Files -> train: {remaining_files}")
    if val_missing or test_missing:
        print(f"[WARN] {val_missing} val / {test_missing} test images moved with no matching label found.")
    print("\n[VERIFY] Run scripts/verify_no_leakage.py now to confirm 0 overlap across all 3 splits.")


if __name__ == "__main__":
    main()