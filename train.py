"""
train.py — Soil classification training pipeline.

Two selectable training modes via TRAIN_CONFIG['use_focal_loss']:

  OPTION A (default) — CrossEntropy + class_weight
    - Standard categorical cross-entropy loss
    - Inverse-frequency class weights passed to model.fit()
    - Stable, well-understood training signal
    - Recommended when dataset is moderately imbalanced

  OPTION B — Focal Loss only (no class_weight)
    - FocalLoss(gamma=2.0, alpha=1.0)
    - class_weight intentionally removed — Focal Loss already handles
      imbalance via (1-p_t)^gamma; combining both double-counts and
      causes gradient instability
    - alpha=1.0 correct for multi-class (0.25 is binary detection default)
    - Recommended when minority classes are severely underrepresented

Switch between modes by changing: TRAIN_CONFIG['use_focal_loss'] = True/False
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ── CPU optimisation — must be set BEFORE TF initialises any ops ─────────────
# Calling these inside a function is too late; TF initialises on first import.
_cpu_count = os.cpu_count() or 4
tf.config.threading.set_inter_op_parallelism_threads(_cpu_count)
tf.config.threading.set_intra_op_parallelism_threads(_cpu_count)
print(f"CPU threads: {_cpu_count} inter-op + {_cpu_count} intra-op")

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_loader import get_data_loaders
from model import build_model, unfreeze_top_layers, print_layer_status
from losses import FocalLoss


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change settings here
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_CONFIG = {
    # Set True for Option B (Focal Loss), False for Option A (CrossEntropy)
    'use_focal_loss': False,

    # Focal Loss parameters (Option B only)
    'focal_gamma': 2.0,
    'focal_alpha': 1.0,   # must be 1.0 for multi-class, NOT 0.25

    # Phase 1 — frozen base, train head only
    'phase1_lr':     1e-3,
    'phase1_epochs': 20,

    # Phase 2 — fine-tune top N layers
    'phase2_unfreeze_layers': 40,
    'phase2_lr_crossentropy': 1e-5,
    'phase2_lr_focal':        3e-5,
    'phase2_epochs':          20,

    # Callbacks — patience=10 gives minority classes time to converge
    'early_stop_patience':    10,
    'reduce_lr_patience':     3,
    'reduce_lr_factor':       0.5,
    'min_delta':              1e-3,

    # Gradient clipping — prevents exploding gradients during fine-tuning
    'gradient_clip_norm':     1.0,

    'batch_size': 32,
}


# ══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(class_names, data_dir):
    """
    Manually tuned class weights based on Grad-CAM analysis results.

    Findings: model has Arid bias — it over-predicts Arid even for Yellow
    and Alluvial samples. Grad-CAM shows the model focuses on colour/tone
    rather than texture for these classes.

    Fix:
      - Arid   reduced to 1.2  (was ~1.75) — reduce its dominance
      - Yellow raised to 1.8   (was ~1.08) — force more attention
      - Alluvial raised to 1.8 (was ~0.72) — force more attention
      - Black / Red kept near 1.0 (balanced classes)

    Returns:
        dict {class_index: weight}
    """
    # Manual overrides from Grad-CAM diagnosis
    MANUAL_WEIGHTS = {
        'alluvial': 1.8,
        'arid':     1.2,
        'black':    0.9,
        'red':      0.9,
        'yellow':   1.8,
    }

    weights = {}
    print("\nClass weights (manually tuned from Grad-CAM analysis):")
    for i, cls in enumerate(class_names):
        w = MANUAL_WEIGHTS.get(cls, 1.0)
        weights[i] = w
        bar = '█' * int(w * 10)
        print(f"  [{i}] {cls:14}: w={w:.1f}  {bar}")

    return weights


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def get_loss_function(config):
    """
    Returns (loss_fn, use_class_weight) based on config.

    Option A: CrossEntropy + class_weight
    Option B: FocalLoss    + no class_weight
    """
    if config['use_focal_loss']:
        loss_fn = FocalLoss(
            gamma=config['focal_gamma'],
            alpha=config['focal_alpha']
        )
        print(f"\nLoss: FocalLoss(gamma={config['focal_gamma']}, "
              f"alpha={config['focal_alpha']})")
        print("      class_weight: DISABLED (Focal Loss handles imbalance)")
        return loss_fn, False   # False = don't use class_weight
    else:
        loss_fn = 'categorical_crossentropy'
        print("\nLoss: Categorical Cross-Entropy")
        print("      class_weight: ENABLED (inverse-frequency)")
        return loss_fn, True    # True = use class_weight


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
def make_callbacks(checkpoint_path, config):
    """
    Builds the standard callback set for one training phase.

    All callbacks monitor val_accuracy (not val_loss) because:
    - Focal Loss values are very small (~0.001–0.05) — absolute changes
      don't reflect accuracy improvements reliably.
    - val_accuracy is the metric we actually care about.
    """
    return [
        EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=config['early_stop_patience'],
            restore_best_weights=True,
            verbose=1,
            min_delta=config['min_delta']
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            mode='max',
            factor=config['reduce_lr_factor'],
            patience=config['reduce_lr_patience'],
            min_lr=1e-8,
            verbose=1
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_model(data_dir='preprocessed_soil_dataset', config=None):
    """
    Full two-phase training pipeline.

    Args:
        data_dir : Root of preprocessed dataset.
        config   : Training configuration dict (defaults to TRAIN_CONFIG).

    Returns:
        (history_p1, history_p2)
    """
    if config is None:
        config = TRAIN_CONFIG

    mode = "OPTION B — Focal Loss" if config['use_focal_loss'] \
           else "OPTION A — CrossEntropy + class_weight"

    print("=" * 65)
    print(f"Training mode: {mode}")
    print("=" * 65)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_ds, val_ds, _, class_names = get_data_loaders(
        data_dir, batch_size=config['batch_size']
    )

    class_indices = {name: i for i, name in enumerate(class_names)}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)

    # ── steps_per_epoch — calculated from actual dataset size ─────────────────
    # Counts total virtual training samples (after oversampling) and divides
    # by batch_size so Keras knows exactly how many steps = 1 epoch.
    # Without this, Keras may under- or over-count steps with sample_from_datasets.
    train_dir = os.path.join(data_dir, 'train')
    raw_counts = {
        cls: len(os.listdir(os.path.join(train_dir, cls)))
        for cls in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, cls))
    }
    from data_loader import MINORITY_CLASSES
    max_count = max(raw_counts.values())
    total_virtual = sum(
        int(np.ceil(max_count / n)) * n if cls in MINORITY_CLASSES else n
        for cls, n in raw_counts.items()
    )
    steps_per_epoch = max(1, total_virtual // config['batch_size'])
    print(f"\n  Virtual training samples : {total_virtual:,}")
    print(f"  Steps per epoch          : {steps_per_epoch}")

    # ── 2. Class weights ──────────────────────────────────────────────────────
    raw_dir = data_dir if os.path.exists(os.path.join(data_dir, 'train')) \
              else 'soil_dataset'
    class_weight_dict = compute_class_weights(class_names, raw_dir)

    # ── 3. Loss function ──────────────────────────────────────────────────────
    loss_fn, use_class_weight = get_loss_function(config)
    fit_class_weight = class_weight_dict if use_class_weight else None

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model, base_model = build_model(num_classes=len(class_names))

    assert model.input_shape[1:] == (224, 224, 3), \
        f"Input shape must be (None,224,224,3), got {model.input_shape}"

    total_params     = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\n  Total parameters    : {total_params:,}")
    print(f"  Trainable (Phase 1) : {trainable_params:,}  (head only)")
    print(f"  Frozen (base)       : {total_params - trainable_params:,}")
    print(f"  Input shape         : {model.input_shape}")

    # ── Phase 1: train head, base frozen ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 1 — Head training (MobileNetV2 base frozen)")
    print(f"  LR={config['phase1_lr']}  |  max_epochs={config['phase1_epochs']}")
    print("=" * 65)

    model.compile(
        optimizer=Adam(
            learning_rate=config['phase1_lr'],
            clipnorm=config['gradient_clip_norm']
        ),
        loss=loss_fn,
        metrics=['accuracy']
    )

    history_p1 = model.fit(
        train_ds,
        epochs=config['phase1_epochs'],
        steps_per_epoch=steps_per_epoch,   # explicit step count for CPU pipeline
        validation_data=val_ds,
        class_weight=fit_class_weight,
        callbacks=make_callbacks('soil_classifier_initial.keras', config),
        verbose=1
    )

    best_p1 = max(history_p1.history['val_accuracy'])
    print(f"\nPhase 1 best val_accuracy: {best_p1:.4f}")

    # ── Phase 2: fine-tune top layers ─────────────────────────────────────────
    n_unfreeze = config['phase2_unfreeze_layers']
    p2_lr = config['phase2_lr_focal'] if config['use_focal_loss'] \
            else config['phase2_lr_crossentropy']

    print("\n" + "=" * 65)
    print(f"PHASE 2 — Fine-tuning top {n_unfreeze} MobileNetV2 layers")
    print(f"  LR={p2_lr}  |  max_epochs={config['phase2_epochs']}")
    print("=" * 65)

    unfreeze_top_layers(base_model, n_layers=n_unfreeze)
    print_layer_status(model)

    model.compile(
        optimizer=Adam(
            learning_rate=p2_lr,
            clipnorm=config['gradient_clip_norm']
        ),
        loss=loss_fn,
        metrics=['accuracy']
    )

    total_epochs = config['phase1_epochs'] + config['phase2_epochs']

    history_p2 = model.fit(
        train_ds,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,   # explicit step count for CPU pipeline
        initial_epoch=history_p1.epoch[-1],
        validation_data=val_ds,
        class_weight=fit_class_weight,
        callbacks=make_callbacks('soil_classifier_final.keras', config),
        verbose=1
    )

    best_p2 = max(history_p2.history['val_accuracy'])
    print(f"\nPhase 2 best val_accuracy : {best_p2:.4f}")
    print(f"Improvement over Phase 1  : {best_p2 - best_p1:+.4f}")

    model.save('soil_classifier_final.keras')
    print("\nSaved → soil_classifier_final.keras")
    return history_p1, history_p2


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_history(h1, h2):
    """Plots Phase 1 + Phase 2 accuracy and loss with phase boundary marker."""
    acc      = h1.history['accuracy']     + h2.history['accuracy']
    val_acc  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss     = h1.history['loss']         + h2.history['loss']
    val_loss = h1.history['val_loss']     + h2.history['val_loss']
    p1_end   = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tr, va, title in [
        (ax1, acc,  val_acc,  'Accuracy'),
        (ax2, loss, val_loss, 'Loss'),
    ]:
        ax.plot(tr, label='Train')
        ax.plot(va, label='Validation')
        ax.axvline(p1_end - 1, color='gray', linestyle='--',
                   linewidth=1, label='Phase 1 → 2')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=150)
    plt.show()
    print("Saved → training_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE PHASE 2 FINE-TUNING  (python train.py --finetune)
# ══════════════════════════════════════════════════════════════════════════════
def finetune_phase2(data_dir='preprocessed_soil_dataset', epochs=10, config=None):
    """
    Loads the existing best model and runs fine-tuning with Grad-CAM
    corrected class weights:
      arid=1.2  (reduced — was causing prediction bias)
      yellow=1.8, alluvial=1.8  (increased — under-predicted classes)
    """
    if config is None:
        config = TRAIN_CONFIG

    print("=" * 65)
    print("PHASE 2 FINE-TUNING — Grad-CAM corrected weights")
    print(f"  Epochs : {epochs}  |  LR : {config['phase2_lr_crossentropy']}")
    print("=" * 65)

    train_ds, val_ds, _, class_names = get_data_loaders(
        data_dir, batch_size=config['batch_size']
    )

    train_dir = os.path.join(data_dir, 'train')
    raw_counts = {
        cls: len(os.listdir(os.path.join(train_dir, cls)))
        for cls in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, cls))
    }
    from data_loader import MINORITY_CLASSES
    max_count = max(raw_counts.values())
    total_virtual = sum(
        int(np.ceil(max_count / n)) * n if cls in MINORITY_CLASSES else n
        for cls, n in raw_counts.items()
    )
    steps_per_epoch = max(1, total_virtual // config['batch_size'])
    print(f"\n  Steps per epoch : {steps_per_epoch}")

    raw_dir = data_dir if os.path.exists(os.path.join(data_dir, 'train')) else 'soil_dataset'
    class_weight_dict = compute_class_weights(class_names, raw_dir)
    fit_class_weight  = None if config['use_focal_loss'] else class_weight_dict

    from losses import FocalLoss
    model = tf.keras.models.load_model(
        'soil_classifier_final.keras',
        custom_objects={'FocalLoss': FocalLoss}
    )
    print(f"\n  Loaded : soil_classifier_final.keras  {model.input_shape}")

    from model import unfreeze_top_layers, print_layer_status
    base = next(l for l in model.layers if 'mobilenetv2' in l.name.lower())
    unfreeze_top_layers(base, n_layers=config['phase2_unfreeze_layers'])
    print_layer_status(model)

    loss_fn = FocalLoss(gamma=2.0, alpha=1.0) if config['use_focal_loss'] \
              else 'categorical_crossentropy'

    model.compile(
        optimizer=Adam(
            learning_rate=config['phase2_lr_crossentropy'],
            clipnorm=config['gradient_clip_norm']
        ),
        loss=loss_fn,
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        class_weight=fit_class_weight,
        callbacks=make_callbacks('soil_classifier_final.keras', config),
        verbose=1
    )

    best = max(history.history['val_accuracy'])
    print(f"\nBest val_accuracy : {best:.4f}")
    print("Saved → soil_classifier_final.keras")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if '--finetune' in sys.argv:
        if os.path.exists('preprocessed_soil_dataset'):
            finetune_phase2(epochs=10)
        else:
            print("Error: 'preprocessed_soil_dataset' not found.")
    elif os.path.exists('preprocessed_soil_dataset'):
        h1, h2 = train_model()
        plot_history(h1, h2)
    else:
        if os.path.exists('soil_dataset'):
            print("Error: 'preprocessed_soil_dataset' not found.")
            print("Run: python preprocess_dataset.py")
        else:
            print("Error: 'soil_dataset' not found.")
