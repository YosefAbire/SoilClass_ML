"""
train.py — Two-phase transfer learning training pipeline.

Phase 1: Frozen MobileNetV2 base, train classification head only.
         High LR (1e-3) is safe because only the small head is updated.

Phase 2: Unfreeze top 40 MobileNetV2 layers, fine-tune at low LR (1e-5).
         BatchNorm layers in the base remain frozen to preserve ImageNet
         running statistics (critical for small-batch fine-tuning).

Loss: FocalLoss(gamma=2.0, alpha=1.0)
  - gamma=2.0 down-weights easy examples, focuses on hard/minority classes.
  - alpha=1.0 correct for multi-class (0.25 is the binary detection default).
  - class_weight NOT used — combining with Focal Loss double-counts imbalance
    correction and causes gradient instability.

Monitoring: val_accuracy (not val_loss) — Focal Loss values are small
  (~0.001–0.05) so absolute changes don't reflect accuracy improvements well.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_loader import get_data_loaders
from model import build_model, unfreeze_top_layers, print_layer_status
from losses import FocalLoss


def compute_class_weights(class_names, data_dir, max_weight=2.0):
    """
    Log-smoothed class weights (kept for reference / optional use).
    NOT used in model.fit() — Focal Loss handles imbalance.

    Formula: w_i = log(N_total) / log(N_i), normalised, capped at max_weight.
    """
    counts = np.array([
        len(os.listdir(os.path.join(data_dir, 'train', cls)))
        for cls in class_names
    ], dtype=np.float32)
    total   = counts.sum()
    weights = np.log(total) / np.log(counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, None, max_weight)
    print("\nClass weights (reference only — not used in training):")
    for cls, n, w in zip(class_names, counts, weights):
        print(f"  {cls:14}: {int(n):4d} samples  weight={w:.4f}")
    return {i: float(w) for i, w in enumerate(weights)}


def train_system(data_dir='preprocessed_soil_dataset', epochs=20, batch_size=32):
    """
    Full two-phase training pipeline.

    Args:
        data_dir   : Root of preprocessed dataset (train/val/test subdirs).
        epochs     : Max epochs for Phase 1.
        batch_size : Samples per batch (32 recommended for MobileNetV2).

    Returns:
        (history_p1, history_p2) — Keras History objects for both phases.
    """

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading data...")
    train_ds, val_ds, test_ds, class_names = get_data_loaders(
        data_dir, batch_size=batch_size
    )

    class_indices = {name: i for i, name in enumerate(class_names)}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print(f"\nClasses: {class_names}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model, base_model = build_model(num_classes=len(class_names))

    # Verify input shape matches MobileNetV2 requirement
    assert model.input_shape[1:] == (224, 224, 3), \
        f"Input shape mismatch: {model.input_shape} — must be (None, 224, 224, 3)"

    # Count parameters
    total_params     = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n  Total parameters      : {total_params:,}")
    print(f"  Trainable (Phase 1)   : {trainable_params:,}  (head only)")
    print(f"  Frozen (base)         : {total_params - trainable_params:,}")

    # ── 3. Loss ───────────────────────────────────────────────────────────────
    # alpha=1.0 for multi-class; gamma=2.0 focuses on hard examples
    focal_loss = FocalLoss(gamma=2.0, alpha=1.0)

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — Training classification head (base frozen)")
    print("=" * 60)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=focal_loss,
        metrics=['accuracy']
    )

    # Print model summary once
    model.summary(line_length=80)

    callbacks_p1 = [
        # val_accuracy is more meaningful than val_loss with Focal Loss
        EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, verbose=1, min_delta=1e-3
        ),
        ModelCheckpoint(
            'soil_classifier_initial.keras', monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=3,
            min_lr=1e-6, verbose=1, mode='max'
        ),
    ]

    history_p1 = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks_p1
        # NOTE: class_weight intentionally omitted.
        # Focal Loss already handles class imbalance via (1-p_t)^gamma.
        # Using both causes double-counting → gradient instability.
    )

    best_p1_acc = max(history_p1.history['val_accuracy'])
    print(f"\nPhase 1 best val_accuracy: {best_p1_acc:.4f}")

    # ── Phase 2: fine-tune top 40 layers ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Fine-tuning top 40 MobileNetV2 layers")
    print("=" * 60)

    # Unfreeze top 40 layers, keep all BN layers frozen
    unfreeze_top_layers(base_model, n_layers=40)
    print_layer_status(model)

    # Low LR essential for fine-tuning — prevents catastrophic forgetting
    # of ImageNet features learned in the base model.
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=focal_loss,
        metrics=['accuracy']
    )

    fine_tune_epochs = 20
    total_epochs     = epochs + fine_tune_epochs

    callbacks_p2 = [
        EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1, min_delta=1e-3
        ),
        ModelCheckpoint(
            'soil_classifier_final.keras', monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=4,
            min_lr=1e-8, verbose=1, mode='max'
        ),
    ]

    history_p2 = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history_p1.epoch[-1],
        validation_data=val_ds,
        callbacks=callbacks_p2
    )

    best_p2_acc = max(history_p2.history['val_accuracy'])
    print(f"\nPhase 2 best val_accuracy: {best_p2_acc:.4f}")
    print(f"Improvement over Phase 1 : {best_p2_acc - best_p1_acc:+.4f}")

    model.save('soil_classifier_final.keras')
    print("\nModel saved → soil_classifier_final.keras")
    return history_p1, history_p2


def plot_history(h1, h2):
    """Plots combined Phase 1 + Phase 2 accuracy and loss curves."""
    acc      = h1.history['accuracy']     + h2.history['accuracy']
    val_acc  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss     = h1.history['loss']         + h2.history['loss']
    val_loss = h1.history['val_loss']     + h2.history['val_loss']
    p1_end   = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, title in [
        (ax1, acc,  val_acc,  'Accuracy'),
        (ax2, loss, val_loss, 'Focal Loss'),
    ]:
        ax.plot(train_vals, label='Train')
        ax.plot(val_vals,   label='Validation')
        ax.axvline(p1_end - 1, color='gray', linestyle='--',
                   linewidth=1, label='Phase 1 → 2')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=150)
    plt.show()
    print("Training plots saved → training_plots.png")


if __name__ == "__main__":
    if os.path.exists('preprocessed_soil_dataset'):
        h1, h2 = train_system()
        plot_history(h1, h2)
    else:
        if os.path.exists('soil_dataset'):
            print("Error: 'preprocessed_soil_dataset' not found.")
            print("Run: python preprocess_dataset.py")
        else:
            print("Error: 'soil_dataset' not found.")
            print("Organise as: soil_dataset/train, /validation, /test")
