"""
preprocess_dataset.py — Image preprocessing pipeline.

Pipeline (per image):
  1. CLAHE on L channel only (LAB colour space) — enhances local contrast
     while preserving colour information (critical for soil type distinction).
     Global histogram equalization was removed — it destroyed colour.
  2. Gaussian blur removed — it softened texture edges which are the primary
     discriminating feature between soil types. ImageNet normalisation in the
     data loader handles input distribution alignment instead.
"""

import os


def preprocess_image(input_path, output_path):
    """
    Applies CLAHE on the L channel of LAB colour space.
    No blur — texture preservation is more important than noise removal
    for soil classification.
    """
    try:
        import cv2

        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("cv2 could not read image")

        # Convert BGR → LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on L channel only — preserves A and B (colour) channels
        # clipLimit=2.0 prevents over-amplification of noise
        # tileGridSize=(8,8) applies local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back to BGR
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True

    except ImportError:
        # PIL fallback — no CLAHE, just copy (better than destroying colour)
        from PIL import Image
        try:
            img = Image.open(input_path).convert('RGB')
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='soil_dataset_4class',
                        help='Source dataset directory (default: soil_dataset_4class)')
    parser.add_argument('--output', default='preprocessed_4class',
                        help='Output directory (default: preprocessed_4class)')
    args = parser.parse_args()

    input_dir  = args.input
    output_dir = args.output

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
