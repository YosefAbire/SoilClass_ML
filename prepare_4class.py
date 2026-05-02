"""
prepare_4class.py — Builds a perfectly equalised 4-class dataset.

Strategy:
  - Exclude arid entirely
  - Cap every class to the MINIMUM count across that split
    so all classes have exactly the same number of images.

  Train    : min(alluvial=481, black=754, red=772, yellow=968) = 481 each
  Validation: min(alluvial=102, black=155, red=164, yellow=204) = 102 each
  Test     : min(alluvial=101, black=161, red=164, yellow=205) = 101 each

  Total after equalisation:
    Train      : 4 × 481 = 1924
    Validation : 4 × 102 =  408
    Test       : 4 × 101 =  404

Output: soil_dataset_4class/  (original soil_dataset/ is untouched)

Usage:
    python prepare_4class.py
    python prepare_4class.py --force   # delete existing and rebuild
"""

import os
import shutil
import random
import argparse

SRC_DIR = 'soil_dataset'
DST_DIR = 'soil_dataset_4class'
CLASSES = ['alluvial', 'black', 'red', 'yellow']   # arid excluded
SEED    = 42
random.seed(SEED)


def get_min_count(split):
    """Returns the minimum image count across all 4 classes for a split."""
    counts = []
    for cls in CLASSES:
        cls_dir = os.path.join(SRC_DIR, split, cls)
        if not os.path.exists(cls_dir):
            continue
        n = len([f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        counts.append(n)
    return min(counts)


def copy_split(split, cap):
    print(f"\n  {split.upper()} — capping at {cap} per class:")
    for cls in CLASSES:
        src = os.path.join(SRC_DIR, split, cls)
        dst = os.path.join(DST_DIR, split, cls)
        os.makedirs(dst, exist_ok=True)

        images = [f for f in os.listdir(src)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Random sample down to cap
        selected = random.sample(images, cap)

        for fname in selected:
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))

        print(f"    {cls:12}: {len(images):4d} → {cap} copied")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Delete existing output and rebuild')
    args = parser.parse_args()

    if os.path.exists(DST_DIR):
        if args.force:
            shutil.rmtree(DST_DIR)
            print(f"Deleted existing '{DST_DIR}'")
        else:
            print(f"'{DST_DIR}' already exists.")
            print("Run with --force to rebuild: python prepare_4class.py --force")
            return

    print(f"Building equalised 4-class dataset → {DST_DIR}/")
    print(f"Classes  : {CLASSES}  (arid excluded)")

    # Compute caps per split
    caps = {split: get_min_count(split) for split in ['train', 'validation', 'test']}
    print(f"\nCaps per split:")
    for split, cap in caps.items():
        print(f"  {split:12}: {cap} per class  ({cap * len(CLASSES)} total)")

    for split, cap in caps.items():
        copy_split(split, cap)

    # Final summary
    print("\n--- Final counts ---")
    grand_total = 0
    for split in ['train', 'validation', 'test']:
        split_total = 0
        for cls in CLASSES:
            n = len(os.listdir(os.path.join(DST_DIR, split, cls)))
            split_total += n
        grand_total += split_total
        print(f"  {split:12}: {split_total} images  ({split_total // len(CLASSES)} per class)")
    print(f"  {'TOTAL':12}: {grand_total} images")

    print(f"\nDone. Dataset saved to '{DST_DIR}/'")
    print("\nNext steps:")
    print("  python preprocess_dataset.py --input soil_dataset_4class --output preprocessed_4class")
    print("  python train.py")


if __name__ == '__main__':
    main()
