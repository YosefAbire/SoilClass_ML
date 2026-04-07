import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_loader import get_data_loaders
from model import build_model
from losses import FocalLoss


# ── Class Weights ─────────────────────────────────────────────────────────────
def compute_class_weights(class_names, data_dir, max_weight=2.0):
    """
    Computes logarithmic-smoothed class weights to counter imbalance
    without over-penalising minority classes.

    Formula:
        w_i = log(N_total) / log(N_i)

    Weights are normalised so mean == 1.0, then capped at max_weight.
    """
    counts = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, 'train', cls)
        counts.append(len(os.listdir(cls_dir)))
    counts  = np.array(counts, dtype=np.float32)
    total   = counts.sum()

    weights = np.log(total) / np.log(counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, a_min=None, a_max=max_weight)

    print("\nClass weights (log-smoothed, capped at {:.1f}):".format(max_weight))
    for i, (cls, w) in enumerate(zip(class_names, weights)):
        print(f"  {cls:12}: {counts[i]:4.0f} samples  →  weight {w:.4f}")

    return {i: float(w) for i, w in enumerate(weights)}


# ── Training ──────────────────────────────────────────────────────────────────
def train_system(data_dir='preprocessed_soil_dataset', epochs=20, batch_size=32):
    """
    Two-phase training with Focal Loss and log-smoothed class weights.

    Phase 1 — frozen MobileNetV2 base, train classification head only.
    Phase 2 — unfreeze top 30 layers, fine-tune at a very low LR.
    """

    # 1. Load data
    print("Loading data...")
    train_ds, val_ds, test_ds, class_names = get_data_loaders(data_dir, batch_size=batch_size)

    class_indices = {name: i for i, name in enumerate(class_names)}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)

    # 2. Class weights — removed (Focal Loss handles imbalance; double-counting
    #    caused gradient instability). Kept compute_class_weights() for reference.
    # class_weight = compute_class_weights(class_names, raw_dir)

    # 3. Build model
    print("\nBuilding model...")
    model, base_model = build_model(num_classes=len(class_names))

    # 4. Focal loss — alpha=1.0 for multi-class (0.25 was binary-detection default)
    focal_loss = FocalLoss(gamma=2.0, alpha=1.0)

    # ── Phase 1: train head, base frozen ─────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=['accuracy']
    )

    callbacks_p1 = [
        # Monitor val_accuracy — more meaningful than val_loss with Focal Loss
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1, min_delta=1e-3),
        ModelCheckpoint('soil_classifier_initial.keras', monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3,
                          min_lr=1e-6, verbose=1, mode='max'),
    ]

    print("\nPhase 1: Training classification head (base frozen)...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        # NOTE: class_weight removed — Focal Loss already handles imbalance.
        # Using both caused double-counting and gradient instability.
        callbacks=callbacks_p1
    )

    # ── Phase 2: fine-tune top 30 MobileNetV2 layers ─────────────────────────
    print("\nPhase 2: Fine-tuning top 30 MobileNetV2 layers...")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Phase 2 LR raised back to 5e-5 now that alpha=1.0 (no more 0.25 scaling)
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss=focal_loss,
        metrics=['accuracy']
    )

    fine_tune_epochs = 15
    total_epochs     = epochs + fine_tune_epochs

    callbacks_p2 = [
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1, min_delta=1e-3),
        ModelCheckpoint('soil_classifier_final.keras', monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4,
                          min_lr=1e-8, verbose=1, mode='max'),
    ]

    history_fine = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        # No class_weight — Focal Loss handles imbalance
        callbacks=callbacks_p2
    )

    model.save('soil_classifier_final.keras')
    print("\nModel saved as soil_classifier_final.keras")
    return history, history_fine


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_history(history, history_fine):
    """Plots and saves training/validation accuracy and loss curves."""
    acc     = history.history['accuracy']     + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss    = history.history['loss']         + history_fine.history['loss']
    val_loss= history.history['val_loss']     + history_fine.history['val_loss']

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc,     label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss,     label='Train Loss (Focal)')
    plt.plot(val_loss, label='Val Loss (Focal)')
    plt.title('Focal Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()


if __name__ == "__main__":
    if os.path.exists('preprocessed_soil_dataset'):
        h1, h2 = train_system()
        plot_history(h1, h2)
    else:
        if os.path.exists('soil_dataset'):
            print("Error: 'preprocessed_soil_dataset' not found.")
            print("Run 'python preprocess_dataset.py' first.")
        else:
            print("Error: 'soil_dataset' not found.")
            print("Organise data as: soil_dataset/train, /validation, /test")
