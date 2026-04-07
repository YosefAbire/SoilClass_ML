"""
verify_data.py — Data Audit & Leakage Detection Utility

Checks:
  1. Duplicate image hashes across train / val / test (data leakage)
  2. Images below 224×224 resolution (too small for MobileNetV2)
  3. Suspicious images (>90% single colour — likely non-soil / blank)

Usage:
    python verify_data.py                        # audits soil_dataset/
    python verify_data.py --dir my_dataset/      # custom directory
    python verify_data.py --move-bad             # move bad images to _rejected/
"""

import os
import sys
import hashlib
import argparse
from collections import defaultdict
from PIL import Image
import numpy as np

DATASET_DIR  = 'soil_dataset'
SPLITS       = ['train', 'validation', 'test']
MIN_SIZE     = (224, 224)
MONO_THRESH  = 0.90   # flag if >90% pixels are within ±15 of the dominant colour
EXTENSIONS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    """MD5 hash of raw file bytes — fast duplicate detection."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def is_suspicious(img: Image.Image, threshold: float = MONO_THRESH) -> tuple[bool, str]:
    """
    Returns (True, reason) if the image is likely not a soil photo.
    Checks whether >threshold fraction of pixels are within ±15 intensity
    of the dominant colour bucket (per channel average).
    """
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w, _ = arr.shape
    total_pixels = h * w

    # Dominant colour = mean of each channel
    dominant = arr.mean(axis=(0, 1))  # shape (3,)

    # Distance of each pixel from dominant colour (L-inf per pixel)
    diff = np.abs(arr - dominant).max(axis=2)  # shape (h, w)
    close_pixels = (diff <= 15).sum()
    ratio = close_pixels / total_pixels

    if ratio >= threshold:
        return True, f"mono-colour ({ratio*100:.1f}% pixels near dominant RGB {dominant.astype(int).tolist()})"
    return False, ""


def collect_images(dataset_dir: str) -> dict:
    """
    Returns {split: {class: [filepath, ...]}} for all images in the dataset.
    """
    result = {}
    for split in SPLITS:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        result[split] = {}
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if os.path.splitext(f)[1].lower() in EXTENSIONS
            ]
            result[split][cls] = files
    return result


# ── Audit functions ───────────────────────────────────────────────────────────

def audit_duplicates(images: dict) -> dict:
    """
    Finds images with identical MD5 hashes.
    Returns {hash: [filepath, ...]} for hashes that appear more than once.
    """
    print("\n[1/3] Checking for duplicate images (data leakage)...")
    hash_map = defaultdict(list)

    all_files = [
        (split, cls, fp)
        for split, classes in images.items()
        for cls, files in classes.items()
        for fp in files
    ]

    for i, (split, cls, fp) in enumerate(all_files):
        h = file_hash(fp)
        hash_map[h].append((split, cls, fp))
        if (i + 1) % 200 == 0:
            print(f"  Hashed {i+1}/{len(all_files)} images...", end='\r')

    duplicates = {h: entries for h, entries in hash_map.items() if len(entries) > 1}
    print(f"  Hashed {len(all_files)} images total.                    ")

    leakage_pairs = []
    same_split_dups = []
    for h, entries in duplicates.items():
        splits_seen = {e[0] for e in entries}
        if len(splits_seen) > 1:
            leakage_pairs.append(entries)
        else:
            same_split_dups.append(entries)

    print(f"  Cross-split duplicates (leakage): {len(leakage_pairs)}")
    print(f"  Same-split duplicates:            {len(same_split_dups)}")

    return {'leakage': leakage_pairs, 'same_split': same_split_dups}


def audit_resolution(images: dict) -> list:
    """
    Returns list of (filepath, actual_size) for images below MIN_SIZE.
    """
    print(f"\n[2/3] Checking image resolutions (minimum {MIN_SIZE[0]}×{MIN_SIZE[1]})...")
    low_res = []
    total = sum(len(f) for cls in images.values() for f in cls.values())
    checked = 0

    for split, classes in images.items():
        for cls, files in classes.items():
            for fp in files:
                try:
                    with Image.open(fp) as img:
                        w, h = img.size
                        if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
                            low_res.append((fp, (w, h)))
                except Exception as e:
                    low_res.append((fp, f"UNREADABLE: {e}"))
                checked += 1
                if checked % 200 == 0:
                    print(f"  Checked {checked}/{total}...", end='\r')

    print(f"  Checked {total} images.                    ")
    print(f"  Below {MIN_SIZE[0]}×{MIN_SIZE[1]}: {len(low_res)}")
    return low_res


def audit_suspicious(images: dict) -> list:
    """
    Returns list of (filepath, reason) for images flagged as non-soil.
    """
    print(f"\n[3/3] Checking for suspicious images (>{MONO_THRESH*100:.0f}% single colour)...")
    suspicious = []
    total = sum(len(f) for cls in images.values() for f in cls.values())
    checked = 0

    for split, classes in images.items():
        for cls, files in classes.items():
            for fp in files:
                try:
                    with Image.open(fp) as img:
                        flagged, reason = is_suspicious(img)
                        if flagged:
                            suspicious.append((fp, reason))
                except Exception as e:
                    suspicious.append((fp, f"UNREADABLE: {e}"))
                checked += 1
                if checked % 200 == 0:
                    print(f"  Checked {checked}/{total}...", end='\r')

    print(f"  Checked {total} images.                    ")
    print(f"  Suspicious images found: {len(suspicious)}")
    return suspicious


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(images: dict, duplicates: dict, low_res: list,
                 suspicious: list, out_path: str = 'data_audit_report.txt'):

    lines = []
    lines.append("=" * 70)
    lines.append("DATA AUDIT REPORT — SoilClass ML")
    lines.append("=" * 70)

    # Dataset summary
    lines.append("\n--- DATASET SUMMARY ---")
    for split, classes in images.items():
        lines.append(f"\n  {split.upper()}")
        total = 0
        for cls, files in classes.items():
            lines.append(f"    {cls:14}: {len(files):5d} images")
            total += len(files)
        lines.append(f"    {'TOTAL':14}: {total:5d} images")

    # Duplicates
    lines.append("\n--- DUPLICATE IMAGES ---")
    if duplicates['leakage']:
        lines.append(f"\n  ⚠️  CROSS-SPLIT DUPLICATES (DATA LEAKAGE): {len(duplicates['leakage'])} groups")
        for group in duplicates['leakage'][:20]:
            lines.append(f"    Hash group:")
            for split, cls, fp in group:
                lines.append(f"      [{split}/{cls}] {os.path.basename(fp)}")
        if len(duplicates['leakage']) > 20:
            lines.append(f"    ... and {len(duplicates['leakage'])-20} more groups")
    else:
        lines.append("  ✅ No cross-split duplicates found.")

    if duplicates['same_split']:
        lines.append(f"\n  ℹ️  Same-split duplicates: {len(duplicates['same_split'])} groups")
        for group in duplicates['same_split'][:10]:
            for split, cls, fp in group:
                lines.append(f"      [{split}/{cls}] {os.path.basename(fp)}")
    else:
        lines.append("  ✅ No same-split duplicates found.")

    # Low resolution
    lines.append(f"\n--- LOW RESOLUTION IMAGES (< {MIN_SIZE[0]}×{MIN_SIZE[1]}) ---")
    if low_res:
        lines.append(f"  ⚠️  {len(low_res)} images below minimum resolution:")
        for item in low_res[:30]:
            fp, size = item
            lines.append(f"    {size}  {fp}")
        if len(low_res) > 30:
            lines.append(f"    ... and {len(low_res)-30} more")
    else:
        lines.append("  ✅ All images meet minimum resolution.")

    # Suspicious
    lines.append(f"\n--- SUSPICIOUS IMAGES (>{MONO_THRESH*100:.0f}% SINGLE COLOUR) ---")
    if suspicious:
        lines.append(f"  ⚠️  {len(suspicious)} suspicious images:")
        for fp, reason in suspicious[:30]:
            lines.append(f"    {reason}")
            lines.append(f"      {fp}")
        if len(suspicious) > 30:
            lines.append(f"    ... and {len(suspicious)-30} more")
    else:
        lines.append("  ✅ No suspicious images found.")

    # Overall verdict
    total_issues = (len(duplicates['leakage']) + len(duplicates['same_split'])
                    + len(low_res) + len(suspicious))
    lines.append("\n--- VERDICT ---")
    if total_issues == 0:
        lines.append("  ✅ Dataset is clean. No issues found.")
    else:
        lines.append(f"  ⚠️  {total_issues} total issues found. Review and clean before retraining.")

    lines.append("\n" + "=" * 70)
    report = "\n".join(lines)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*70}")
    print(report)
    print(f"\nReport saved to: {out_path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Audit soil dataset for quality issues.')
    parser.add_argument('--dir',      default=DATASET_DIR, help='Dataset root directory')
    parser.add_argument('--move-bad', action='store_true',  help='Move flagged images to _rejected/')
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Dataset directory '{args.dir}' not found.")
        sys.exit(1)

    print(f"Auditing dataset: {args.dir}")
    images     = collect_images(args.dir)
    duplicates = audit_duplicates(images)
    low_res    = audit_resolution(images)
    suspicious = audit_suspicious(images)
    write_report(images, duplicates, low_res, suspicious)

    if args.move_bad:
        rejected_dir = os.path.join(args.dir, '_rejected')
        os.makedirs(rejected_dir, exist_ok=True)
        moved = 0
        bad_files = (
            [fp for group in duplicates['leakage'] for _, _, fp in group[1:]]  # keep first copy
            + [fp for fp, _ in low_res if isinstance(_, tuple)]
            + [fp for fp, _ in suspicious]
        )
        for fp in set(bad_files):
            dst = os.path.join(rejected_dir, os.path.basename(fp))
            os.rename(fp, dst)
            moved += 1
        print(f"\nMoved {moved} flagged images to {rejected_dir}/")


if __name__ == '__main__':
    main()
