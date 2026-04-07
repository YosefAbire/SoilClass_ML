"""
model.py — MobileNetV2 transfer learning model for soil classification.

Architecture decisions:
  - Simplified head: GAP → BN → Dense(128) → Dropout(0.3) → Dense(5)
    Removing the Dense(256) layer reduces overfitting risk on a small dataset.
    BatchNorm is placed immediately after GAP (before Dense) so it normalises
    the raw feature distribution from the backbone — more effective than
    placing it between Dense layers.

  - Dropout(0.3) is conservative — aggressive dropout (>0.4) on a small
    dataset causes underfitting, not regularisation.

  - L2 regularisation on Dense(128) adds a second regularisation axis
    independent of dropout, helping minority class generalisation.

  - unfreeze_top_layers() is a separate function so train.py controls
    exactly when and how many layers are unfrozen, keeping concerns separated.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def build_model(num_classes=5, input_shape=(224, 224, 3)):
    """
    Builds the MobileNetV2-based classification model.

    Phase 1 usage: base is frozen, only the head is trained.
    Phase 2 usage: call unfreeze_top_layers() then recompile.

    Args:
        num_classes  : Number of output classes (5 soil types).
        input_shape  : Must be (224, 224, 3) — MobileNetV2 requirement.

    Returns:
        model      : Full compiled-ready Keras model.
        base_model : MobileNetV2 reference (for unfreezing in Phase 2).
    """
    # ── Base: MobileNetV2 pretrained on ImageNet ──────────────────────────────
    # include_top=False removes the original 1000-class classifier.
    # weights='imagenet' loads pretrained feature extraction weights.
    # Input MUST be 224×224×3 and ImageNet-normalised for these weights to work.
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze entire base for Phase 1 — only train the head
    base_model.trainable = False

    # ── Classification head ───────────────────────────────────────────────────
    x = base_model.output

    # GlobalAveragePooling: (7, 7, 1280) → (1280,)
    # More parameter-efficient than Flatten; reduces spatial overfitting.
    x = GlobalAveragePooling2D(name="gap")(x)

    # BatchNorm immediately after GAP normalises the backbone feature
    # distribution before it enters the Dense layers. This is more effective
    # than placing BN between Dense layers because the GAP output can have
    # very different scales across the 1280 channels.
    x = BatchNormalization(name="head_bn")(x)

    # Single Dense layer — sufficient capacity for 5-class problem.
    # L2 regularisation (1e-4) adds weight decay independent of dropout.
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name="head_dense"
    )(x)

    # Dropout(0.3) — conservative; aggressive dropout causes underfitting
    # on datasets with <1000 samples per class.
    x = Dropout(0.3, name="head_dropout")(x)

    # Output: 5-class softmax probability distribution
    predictions = Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def unfreeze_top_layers(base_model, n_layers=40):
    """
    Unfreezes the top N layers of MobileNetV2 for Phase 2 fine-tuning.

    Strategy:
    - Unfreeze the last 40 layers (increased from 30 — captures more
      high-level texture/colour features relevant to soil classification).
    - Keep all BatchNorm layers in the base FROZEN even when their
      surrounding conv layers are unfrozen. This is critical: unfreezing
      BN layers with a small batch size corrupts the running statistics
      accumulated during ImageNet training, destabilising fine-tuning.

    Args:
        base_model : The MobileNetV2 base returned by build_model().
        n_layers   : Number of layers from the end to unfreeze.

    Returns:
        trainable_count  : Number of trainable parameters after unfreezing.
        frozen_count     : Number of frozen parameters.
    """
    base_model.trainable = True

    # Freeze all layers first, then selectively unfreeze the top N
    freeze_until = len(base_model.layers) - n_layers
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # Keep ALL BatchNorm layers in the base frozen regardless of position.
    # Reason: BN running mean/variance were calibrated on ImageNet.
    # Fine-tuning with small batches (32) would corrupt these statistics.
    bn_frozen = 0
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            bn_frozen += 1

    # Diagnostic output
    trainable     = sum(tf.size(w).numpy() for w in base_model.trainable_weights)
    non_trainable = sum(tf.size(w).numpy() for w in base_model.non_trainable_weights)
    total_layers  = len(base_model.layers)
    unfrozen      = sum(1 for l in base_model.layers if l.trainable)

    print(f"\n  Base model layers     : {total_layers}")
    print(f"  Unfrozen layers       : {unfrozen}  (top {n_layers}, BN kept frozen)")
    print(f"  BatchNorm kept frozen : {bn_frozen}")
    print(f"  Trainable params      : {trainable:,}")
    print(f"  Frozen params         : {non_trainable:,}")

    return trainable, non_trainable


def print_layer_status(model, max_lines=20):
    """
    Prints a summary of which layers are frozen vs trainable.
    Shows the first and last max_lines layers to keep output readable.
    """
    print("\n  Layer freeze status (sample):")
    print(f"  {'Layer name':<45} {'Trainable'}")
    print(f"  {'-'*55}")
    all_layers = model.layers
    show = (list(range(min(5, len(all_layers)))) +
            list(range(max(5, len(all_layers)-15), len(all_layers))))
    prev = -1
    for i in show:
        if i != prev + 1:
            print(f"  {'...':<45}")
        layer = all_layers[i]
        status = "✓ trainable" if layer.trainable else "✗ frozen"
        print(f"  {layer.name:<45} {status}")
        prev = i
