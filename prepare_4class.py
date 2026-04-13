"""
prepare_4class.py — Removes Arid class and applies Cap & Boost strategy.

Strategy:
  - Exclude arid entirely from all splits
  - Training set: target 650 images per class
      > 650 (black=754, red=772, yellow=968) → random down-sample to 650
      < 650 (alluvial=481)                   → copy as-is (augmentation
                                               happens live in data_loader)
  - Val / Test: keep as-is (just exclude arid)

Output: soil_dataset_4class/  (original soil_dataset/ is untouched)

Usage:
    python prepare_4class.py
"""

import os
import shutil
import random

SRC_DIR    = 'soil_dataset'
DST_DIR    = 'soil_dataset_4class'
CLASSES    = ['alluvial', 'black', 'red', 'yellow']   # arid excluded
TRAIN_CAP  = 650
SEED       = 42
random.seed(SEED)


def copy_split(split, cap=None):
    for cls in CLASSES:
        src = os.path.join(SRC_DIR, split, cls)
        dst = os.path.join(DST_DIR, split, cls)
        os.makedirs(dst, exist_ok=True)

        images = [f for f in os.listdir(src)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if cap and len(images) > cap:
            # Down-sample: random subset
            images = random.sample(images, cap)
            print(f"  {split}/{cls}: {len(os.listdir(src))} → {cap} (down-sampled)")
        else:
            print(f"  {split}/{cls}: {len(images)} (kept as-is)")

        for fname in images:
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))


def main():
    if os.path.exists(DST_DIR):
        print(f"'{DST_DIR}' already exists — delete it to re-run.")
        return

    print(f"Building 4-class dataset → {DST_DIR}/")
    print(f"Classes : {CLASSES}")
    print(f"Train cap: {TRAIN_CAP} per class\n")

    print("TRAIN (cap at 650):")
    copy_split('train', cap=TRAIN_CAP)

    print("\nVALIDATION (no cap):")
    copy_split('validation', cap=None)

    print("\nTEST (no cap):")
    copy_split('test', cap=None)

    # Summary
    print("\n--- Final counts ---")
    for split in ['train', 'validation', 'test']:
        total = 0
        for cls in CLASSES:
            n = len(os.listdir(os.path.join(DST_DIR, split, cls)))
            total += n
            print(f"  {split}/{cls}: {n}")
        print(f"  {split} TOTAL: {total}\n")

    print(f"Done. Dataset saved to '{DST_DIR}/'")
    print("Next: python preprocess_dataset.py  (point it at soil_dataset_4class)")
    print("Then: python train.py")


if __name__ == '__main__':
    main()
