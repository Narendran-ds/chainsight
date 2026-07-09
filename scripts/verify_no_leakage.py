"""
ChainSight — Train/Val Leakage Verification Script
=========================================================================
Confirms zero source-image overlap between train/ and val/ after running
split_val.py. Uses the same base_key() grouping logic (strips Roboflow's
'.rf.<hash>' suffix) to detect if any augmented variant of the same source
image ended up split across both sets.

Usage:
    python scripts/verify_no_leakage.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "processed" / "chainsight_dataset"

VALID_EXTS = {".jpg", ".jpeg", ".png"}
RF_SUFFIX_PATTERN = re.compile(r"\.rf\..*$")


def base_key(filename_stem: str) -> str:
    return RF_SUFFIX_PATTERN.sub("", filename_stem)


def collect_keys(images_dir: Path):
    keys = set()
    for p in images_dir.glob("*"):
        if p.suffix.lower() in VALID_EXTS:
            keys.add(base_key(p.stem))
    return keys


def main():
    train_images_dir = DATASET_ROOT / "train" / "images"
    val_images_dir = DATASET_ROOT / "val" / "images"
    test_images_dir = DATASET_ROOT / "test" / "images"

    train_keys = collect_keys(train_images_dir)
    val_keys = collect_keys(val_images_dir)
    test_keys = collect_keys(test_images_dir) if test_images_dir.exists() else set()

    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys

    print("========== LEAKAGE CHECK ==========")
    print(f"train unique source images: {len(train_keys)}")
    print(f"val   unique source images: {len(val_keys)}")
    print(f"test  unique source images: {len(test_keys)}")
    print()

    ok = True
    if train_val_overlap:
        ok = False
        print(f"[FAIL] {len(train_val_overlap)} source images appear in BOTH train and val:")
        for k in sorted(train_val_overlap)[:10]:
            print(f"    {k}")
        if len(train_val_overlap) > 10:
            print(f"    ... and {len(train_val_overlap) - 10} more")
    else:
        print("[OK] No overlap between train and val.")

    if train_test_overlap:
        ok = False
        print(f"[FAIL] {len(train_test_overlap)} source images appear in BOTH train and test:")
        for k in sorted(train_test_overlap)[:10]:
            print(f"    {k}")
    else:
        print("[OK] No overlap between train and test.")

    if val_test_overlap:
        ok = False
        print(f"[FAIL] {len(val_test_overlap)} source images appear in BOTH val and test:")
        for k in sorted(val_test_overlap)[:10]:
            print(f"    {k}")
    else:
        print("[OK] No overlap between val and test.")

    print()
    if ok:
        print("[PASS] Split is leakage-safe. Safe to proceed with training.")
    else:
        print("[STOPPED] Leakage detected — do NOT trust validation metrics until fixed.")


if __name__ == "__main__":
    main()