"""
main_gradcam.py — Grad-CAM analysis for the soil classification model.

Generates heatmaps showing WHICH regions of each image the model focused on
when making its prediction. Use this to understand why the 38.27% accuracy
model is confusing arid/red/yellow.

Usage:
    python main_gradcam.py                  # auto-samples 2 images per class
    python main_gradcam.py --image path.jpg # analyse a single image

Output: gradcam_outputs/ folder with side-by-side original + heatmap images.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

from losses import FocalLoss
from gradcam import find_last_conv_layer, get_gradcam_heatmap, overlay_heatmap
from utils import preprocess_image, deprocess_image

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH  = 'soil_classifier_final.keras'   # latest trained model
DATASET_DIR = 'soil_dataset/test'             # source of test images

# Alphabetical order — must match how tf.keras reads the dataset folder
CLASS_NAMES = ['alluvial', 'arid', 'black', 'red', 'yellow']

OUTPUT_DIR  = 'gradcam_outputs'
FINAL_DIR   = 'final_explanations'   # "After" balanced model outputs
SAMPLES_PER_CLASS = 2   # images to analyse per class in auto mode


# ── Core analysis function ────────────────────────────────────────────────────
def explain_prediction(image_path, model, layer_name, class_names, out_dir=None):
    if out_dir is None:
        out_dir = OUTPUT_DIR
    """
    Runs Grad-CAM on one image and saves a side-by-side figure:
      left  — original image
      right — Grad-CAM heatmap overlay

    Also prints prediction probabilities to stdout.
    """
    processed = preprocess_image(image_path)
    original  = deprocess_image(image_path)

    # Predict
    preds    = model.predict(processed, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    top3     = preds[0].argsort()[-3:][::-1]

    # True label from folder name
    true_label = os.path.basename(os.path.dirname(image_path))
    correct    = true_label == class_names[pred_idx]

    print(f"\n{'✓' if correct else '✗'} {os.path.basename(image_path)}")
    print(f"  True     : {true_label}")
    print(f"  Predicted: {class_names[pred_idx]} ({preds[0][pred_idx]:.1%})")
    print(f"  Top 3    : " + " | ".join(
        f"{class_names[i]} {preds[0][i]:.1%}" for i in top3))

    # Grad-CAM
    heatmap = get_gradcam_heatmap(processed, model, layer_name, pred_index=pred_idx)
    overlay = overlay_heatmap(original, heatmap)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title(f'Original\nTrue: {true_label}', fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    border_color = 'green' if correct else 'red'
    for spine in axes[1].spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
    axes[1].set_title(
        f'Grad-CAM\nPred: {class_names[pred_idx]} ({preds[0][pred_idx]:.1%})',
        fontsize=10, color=border_color, fontweight='bold'
    )
    axes[1].axis('off')

    # Probability bar chart
    colors = ['#2ecc71' if i == pred_idx else
              '#e74c3c' if class_names[i] == true_label else '#bdc3c7'
              for i in range(len(class_names))]
    axes[2].barh([c.capitalize() for c in class_names], preds[0], color=colors)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel('Probability')
    axes[2].set_title('Class Probabilities', fontsize=10)
    for i, p in enumerate(preds[0]):
        axes[2].text(p + 0.01, i, f'{p:.1%}', va='center', fontsize=8)

    plt.suptitle(
        f"{'CORRECT' if correct else 'WRONG'} — {os.path.basename(image_path)}",
        fontsize=11, fontweight='bold',
        color='green' if correct else 'red'
    )
    plt.tight_layout()

    fname = f"{'OK' if correct else 'WRONG'}_{true_label}_pred{class_names[pred_idx]}_{os.path.basename(image_path)}"
    save_path = os.path.join(out_dir, fname + '.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

    return correct, true_label, class_names[pred_idx]


# ── Summary plot ──────────────────────────────────────────────────────────────
def print_summary(results):
    print("\n" + "=" * 55)
    print("GRAD-CAM ANALYSIS SUMMARY")
    print("=" * 55)
    total   = len(results)
    correct = sum(1 for r in results if r[0])
    print(f"Analysed : {total} images")
    print(f"Correct  : {correct} ({correct/total:.1%})")
    print(f"Wrong    : {total-correct} ({(total-correct)/total:.1%})")

    # Confusion breakdown
    from collections import Counter
    wrong = [(r[1], r[2]) for r in results if not r[0]]
    if wrong:
        print("\nMisclassification patterns:")
        for (true, pred), count in Counter(wrong).most_common():
            print(f"  {true:12} → {pred:12} : {count}×")
    print("=" * 55)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image to analyse')
    parser.add_argument('--final', action='store_true',
                        help='Save to final_explanations/ instead of gradcam_outputs/')
    args = parser.parse_args()

    out_dir = FINAL_DIR if args.final else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'FocalLoss': FocalLoss}
    )
    print(f"Model loaded. Input shape: {model.input_shape}")

    # Find last conv layer automatically
    layer_name = find_last_conv_layer(model)
    print(f"Grad-CAM layer: {layer_name}")

    results = []

    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"Error: {args.image} not found.")
            return
        r = explain_prediction(args.image, model, layer_name, CLASS_NAMES, out_dir)
        results.append(r)
    else:
        print(f"\nAuto mode: {SAMPLES_PER_CLASS} images per class from {DATASET_DIR}")
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(DATASET_DIR, cls)
            if not os.path.exists(cls_dir):
                print(f"  Skipping {cls} — folder not found")
                continue
            images = [f for f in os.listdir(cls_dir)
                      if os.path.splitext(f)[1].lower() in {'.jpg','.jpeg','.png'}]
            indices = np.linspace(0, len(images)-1, SAMPLES_PER_CLASS, dtype=int)
            for idx in indices:
                path = os.path.join(cls_dir, images[idx])
                r = explain_prediction(path, model, layer_name, CLASS_NAMES, out_dir)
                results.append(r)

    print_summary(results)
    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == '__main__':
    main()
