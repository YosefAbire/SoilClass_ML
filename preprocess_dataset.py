"""
preprocess_dataset.py — Image preprocessing pipeline.

Changes from original:
  - Replaced global Histogram Equalization (destroyed colour) with CLAHE
    (Contrast Limited Adaptive HE) applied only to the L channel in LAB
    space — enhances local contrast while preserving colour information.
  - Gaussian blur applied BEFORE contrast enhancement (correct order:
    denoise first, then enhance).
  - Falls back to PIL if OpenCV is not installed.
"""

import os

def preprocess_image(input_path, output_path):
    """
    Pipeline:
      1. Gaussian blur (radius=0.8) — light noise removal
      2. CLAHE on L channel in LAB space — local contrast enhancement
         without destroying colour (critical for soil type distinction)
      3. Save to output path
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("cv2 could not read image")

        # Step 1 — light Gaussian blur (denoise before enhancing)
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)

        # Step 2 — CLAHE on L channel only (preserves colour)
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        lab   = cv2.merge([l, a, b])
        img   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True

    except ImportError:
        # Fallback: PIL-only path (no CLAHE, just blur)
        from PIL import Image, ImageFilter
        try:
            img = Image.open(input_path).convert('RGB')
            img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    input_dir  = 'soil_dataset'
    output_dir = 'preprocessed_soil_dataset'

    if not os.path.exists(input_dir):
        print(f"Error: '{input_dir}' not found.")
        return

    try:
        import cv2
        print("Using OpenCV CLAHE preprocessing pipeline.")
    except ImportError:
        print("OpenCV not found — using PIL fallback (no CLAHE).")
        print("Install OpenCV for best results: pip install opencv-python")

    print(f"Preprocessing '{input_dir}' → '{output_dir}'...")

    processed = 0
    errors    = 0

    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(input_dir, split)
        if not os.path.exists(split_dir):
            continue
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(output_dir, split, cls, fname)
                if preprocess_image(src, dst):
                    processed += 1
                    if processed % 100 == 0:
                        print(f"  Processed {processed} images...", end='\r')
                else:
                    errors += 1

    print(f"\nDone. Processed: {processed}  Errors: {errors}")
    print(f"Output saved to '{output_dir}'. Run python train.py next.")


if __name__ == '__main__':
    main()
