"""
clean_dataset.py — Automated dataset cleaner based on data_audit_report.txt findings.

Actions performed:
  1. Remove cross-split duplicates (data leakage) — keeps the train copy,
     removes the val/test copy. If both are in val/test, removes the test copy.
  2. Remove same-split duplicates — keeps the first file, removes the rest.
  3. Remove low-resolution images (below 224×224).
  4. Remove suspicious mono-colour images (>90% single colour).

Run:
    python clean_dataset.py              # dry run — shows what would be deleted
    python clean_dataset.py --execute    # actually deletes the files
"""

import os
import sys
import hashlib
from collections import defaultdict
from PIL import Image
import numpy as np

DATASET_DIR  = 'soil_dataset'
SPLITS       = ['train', 'validation', 'test']
MIN_SIZE     = (224, 224)
MONO_THRESH  = 0.90
EXTENSIONS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DRY_RUN      = '--execute' not in sys.argv


def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def is_suspicious(path):
    try:
        with Image.open(path) as img:
            arr = np.array(img.convert('RGB'), dtype=np.float32)
            dominant = arr.mean(axis=(0, 1))
            diff = np.abs(arr - dominant).max(axis=2)
            ratio = (diff <= 15).sum() / (arr.shape[0] * arr.shape[1])
            return ratio >= MONO_THRESH
    except Exception:
        return False


def is_low_res(path):
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w < MIN_SIZE[0] or h < MIN_SIZE[1]
    except Exception:
        return True  # unreadable = treat as bad


def collect_all_images():
    all_files = []
    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(DATASET_DIR, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in EXTENSIONS:
                    all_files.append((split, cls, os.path.join(cls_dir, fname)))
    return all_files


def remove_file(path, reason, to_delete):
    to_delete.append((path, reason))


def main():
    print("=" * 65)
    print(f"Dataset Cleaner — {'DRY RUN (pass --execute to apply)' if DRY_RUN else 'LIVE EXECUTION'}")
    print("=" * 65)

    all_files = collect_all_images()
    print(f"\nTotal images found: {len(all_files)}")

    to_delete = []   # list of (path, reason)
    seen_hashes = defaultdict(list)  # hash → [(split, cls, path)]

    # ── Pass 1: hash all files ────────────────────────────────────────────────
    print("\nHashing all images...")
    for i, (split, cls, path) in enumerate(all_files):
        h = file_hash(path)
        seen_hashes[h].append((split, cls, path))
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(all_files)}...", end='\r')
    print(f"  {len(all_files)}/{len(all_files)} done.          ")

    # ── Pass 2: resolve duplicates ────────────────────────────────────────────
    print("\nResolving duplicates...")
    dup_cross = 0
    dup_same  = 0

    for h, entries in seen_hashes.items():
        if len(entries) < 2:
            continue

        splits_present = [e[0] for e in entries]

        if len(set(splits_present)) > 1:
            # Cross-split duplicate — keep train copy, remove others
            # Priority: train > validation > test
            priority = {'train': 0, 'validation': 1, 'test': 2}
            entries_sorted = sorted(entries, key=lambda e: priority.get(e[0], 3))
            keep = entries_sorted[0]
            for entry in entries_sorted[1:]:
                remove_file(entry[2],
                    f"cross-split duplicate of [{keep[0]}/{keep[1]}] {os.path.basename(keep[2])}",
                    to_delete)
                dup_cross += 1
        else:
            # Same-split duplicate — keep first, remove rest
            for entry in entries[1:]:
                remove_file(entry[2],
                    f"same-split duplicate of {os.path.basename(entries[0][2])}",
                    to_delete)
                dup_same += 1

    print(f"  Cross-split duplicates to remove : {dup_cross}")
    print(f"  Same-split duplicates to remove  : {dup_same}")

    # ── Pass 3: low resolution ────────────────────────────────────────────────
    print("\nChecking resolutions...")
    low_res_count = 0
    already_flagged = {p for p, _ in to_delete}

    for split, cls, path in all_files:
        if path in already_flagged:
            continue
        if is_low_res(path):
            remove_file(path, f"below {MIN_SIZE[0]}×{MIN_SIZE[1]}", to_delete)
            already_flagged.add(path)
            low_res_count += 1

    print(f"  Low-resolution images to remove  : {low_res_count}")

    # ── Pass 4: suspicious mono-colour ───────────────────────────────────────
    print("\nChecking for mono-colour images...")
    mono_count = 0

    for split, cls, path in all_files:
        if path in already_flagged:
            continue
        if is_suspicious(path):
            remove_file(path, f"mono-colour (>{MONO_THRESH*100:.0f}% single colour)", to_delete)
            already_flagged.add(path)
            mono_count += 1

    print(f"  Mono-colour images to remove     : {mono_count}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nTotal files to remove: {len(to_delete)}")
    print(f"Remaining after clean: {len(all_files) - len(to_delete)}")

    # ── Execute or dry-run ────────────────────────────────────────────────────
    if DRY_RUN:
        print("\n[DRY RUN] Files that would be deleted:")
        for path, reason in to_delete[:40]:
            print(f"  {reason}")
            print(f"    {path}")
        if len(to_delete) > 40:
            print(f"  ... and {len(to_delete)-40} more")
        print("\nRun with --execute to apply changes.")
    else:
        print("\nDeleting files...")
        deleted = 0
        errors  = 0
        for path, reason in to_delete:
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                print(f"  ERROR removing {path}: {e}")
                errors += 1
        print(f"\nDeleted : {deleted}")
        print(f"Errors  : {errors}")
        print("\nDataset cleaned. Re-run verify_data.py to confirm, then retrain.")


if __name__ == '__main__':
    main()
