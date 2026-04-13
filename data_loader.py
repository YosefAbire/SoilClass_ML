"""
data_loader.py — Preprocessing & augmentation pipeline for MobileNetV2.

Key design decisions:
  1. ImageNet normalisation applied to ALL splits (train/val/test) so the
     input distribution matches MobileNetV2's pre-training exactly.
  2. Augmentation operates on [0,1] float images BEFORE normalisation,
     then normalisation is applied as the final step.
  3. Minority classes (arid, alluvial, yellow) use texture-focused
     augmentation: narrow colour jitter, Gaussian noise, random sharpening.
  4. Mixup blending creates genuinely new synthetic samples for minority
     classes rather than just repeating the same images.
  5. Shuffle-before-repeat ensures each oversampled pass sees a different
     image order, preventing the model memorising repetition patterns.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# ── Constants ─────────────────────────────────────────────────────────────────
MINORITY_CLASSES = {'alluvial'}   # arid removed; alluvial is the only class below 650

# MobileNetV2 was pre-trained with these ImageNet statistics.
# Applying them aligns our input distribution with the base model's
# learned feature space — critical for effective transfer learning.
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ── Normalisation ─────────────────────────────────────────────────────────────
def imagenet_normalize(image):
    """
    Scale pixel values from [0, 255] → [0, 1] → ImageNet normalised.

    Formula: x_norm = (x / 255 - mean) / std
    Output range: approximately [-2.1, 2.6] per channel.
    """
    image = tf.cast(image, tf.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


# ── Mixup ─────────────────────────────────────────────────────────────────────
def mixup_batch(images, labels, alpha=0.3):
    """
    Mixup augmentation: creates synthetic samples by blending image pairs.

    x_mix = λ·x_i + (1-λ)·x_j
    y_mix = λ·y_i + (1-λ)·y_j

    λ ~ Uniform(alpha, 1-alpha)  →  keeps blends close to one original,
    not a 50/50 blend, so the dominant class label is still meaningful.

    Applied after ImageNet normalisation so blending is in the same
    feature space the model will operate in.
    """
    batch_size = tf.shape(images)[0]
    lam        = tf.random.uniform([batch_size], minval=alpha, maxval=1.0 - alpha)
    indices    = tf.random.shuffle(tf.range(batch_size))
    images_j   = tf.gather(images, indices)
    labels_j   = tf.gather(labels, indices)
    lam_img    = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_lbl    = tf.reshape(lam, [batch_size, 1])
    return (lam_img * images + (1.0 - lam_img) * images_j,
            lam_lbl * labels + (1.0 - lam_lbl) * labels_j)


# ── Augmentation policies ─────────────────────────────────────────────────────
def build_standard_aug():
    """
    Standard augmentation for majority classes (black, red).
    Moderate spatial transforms + narrow colour jitter.
    Narrow brightness/contrast (0.1) prevents the model from relying
    on lighting conditions rather than soil texture.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.20),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.10, 0.10),
        # Narrow range — model should not rely on lighting
        layers.RandomBrightness(0.10),
        layers.RandomContrast(0.10),
    ], name="standard_aug")


def build_texture_aug():
    """
    Texture-focused augmentation for minority classes (arid, alluvial, yellow).

    Rationale:
    - Arid, Red, Yellow share similar warm colour tones → narrow colour
      jitter (0.12) so the model cannot cheat by memorising hue.
    - GaussianNoise (stddev=0.025) forces the model to learn robust texture
      features (grain size, surface roughness) rather than exact pixel values.
    - RandomSharpen emphasises soil grain and texture edges — the primary
      discriminating feature between visually similar classes.
    - Per-channel scaling [0.88, 1.12] simulates camera white-balance
      variation across different field photography conditions.
    - Wider spatial transforms (rotation ±30°, zoom ±25%) create more
      geometric variety from the limited pool of real minority images.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.30),
        layers.RandomZoom(0.25),
        layers.RandomTranslation(0.15, 0.15),

        # Narrow colour jitter — prevent colour-reliance
        layers.RandomBrightness(0.12),
        layers.RandomContrast(0.12),

        # Per-channel white-balance simulation
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x * tf.random.uniform([1, 1, 3], 0.88, 1.12), 0.0, 1.0
            ), name="channel_scale"
        ),

        # Gaussian noise — forces texture-based feature learning
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.025),
                0.0, 1.0
            ), name="gaussian_noise"
        ),

        # Random sharpening — emphasises grain size and texture edges
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x + tf.random.uniform([], 0.0, 0.20) * (
                    x - tf.nn.avg_pool2d(
                        tf.expand_dims(x, 0), ksize=3, strides=1, padding='SAME'
                    )[0]
                ), 0.0, 1.0
            ), name="random_sharpen"
        ),
    ], name="texture_aug")


# ── Main loader ───────────────────────────────────────────────────────────────
def get_data_loaders(data_dir, target_size=(224, 224), batch_size=32,
                     use_mixup=True, mixup_alpha=0.3):
    """
    Builds tf.data pipelines for train / validation / test.

    Pipeline order (training):
      raw uint8 image
        → cast to float32, divide by 255  →  [0, 1]
        → augmentation (spatial + texture)  →  [0, 1]
        → batch
        → optional Mixup blending
        → ImageNet normalisation  →  ~[-2.1, 2.6]

    Pipeline order (val / test):
      raw uint8 image
        → cast to float32, divide by 255  →  [0, 1]
        → ImageNet normalisation  →  ~[-2.1, 2.6]
    """

    # ── Load datasets ─────────────────────────────────────────────────────────
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        label_mode='categorical',
        image_size=target_size,
        batch_size=None,          # unbatched — needed for per-class filtering
        shuffle=True,
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/validation",
        label_mode='categorical',
        image_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        label_mode='categorical',
        image_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds_raw.class_names
    n_classes   = len(class_names)
    train_dir   = f"{data_dir}/train"

    counts = np.array([
        len(os.listdir(os.path.join(train_dir, cls)))
        for cls in class_names
    ], dtype=np.float32)
    max_count = counts.max()

    print("\nDataset class counts:")
    for cls, n in zip(class_names, counts):
        tag = " ← minority (oversample + texture aug)" if cls in MINORITY_CLASSES else ""
        print(f"  {cls:14}: {int(n):4d} images{tag}")

    standard_aug = build_standard_aug()
    texture_aug  = build_texture_aug()
    AUTOTUNE     = tf.data.AUTOTUNE

    # ── Per-class oversampling + augmentation ─────────────────────────────────
    class_datasets = []
    for i, cls in enumerate(class_names):
        is_minority = cls in MINORITY_CLASSES
        aug_fn      = texture_aug if is_minority else standard_aug

        # Filter to this class only
        cls_ds = train_ds_raw.filter(
            lambda x, y, idx=i: tf.equal(tf.argmax(y), idx)
        )

        # Step 1: cast to float32 and scale to [0, 1]
        # Augmentation must operate on [0,1] before ImageNet normalisation
        cls_ds = cls_ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=AUTOTUNE
        )

        if is_minority:
            # Repeat so minority class reaches ~max_count virtual samples.
            # Shuffle BEFORE repeat → each pass sees a different image order.
            # Augmentation AFTER repeat → every copy gets fresh random transforms.
            repeat_factor = int(np.ceil(max_count / counts[i]))
            print(f"  {cls:14}: repeat×{repeat_factor} "
                  f"→ ~{int(counts[i]*repeat_factor)} virtual samples")
            cls_ds = (cls_ds
                      .shuffle(int(counts[i]), reshuffle_each_iteration=True)
                      .repeat(repeat_factor)
                      .map(lambda x, y: (aug_fn(x, training=True), y),
                           num_parallel_calls=AUTOTUNE))
        else:
            cls_ds = (cls_ds
                      .shuffle(int(counts[i]), reshuffle_each_iteration=True)
                      .map(lambda x, y: (aug_fn(x, training=True), y),
                           num_parallel_calls=AUTOTUNE))

        class_datasets.append(cls_ds)

    # Interleave all class streams with equal probability
    train_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0 / n_classes] * n_classes,
        seed=42,
        stop_on_empty_dataset=False
    )

    # Step 2: batch
    train_ds = train_ds.batch(batch_size)

    # Step 3: optional Mixup (operates on batched [0,1] images)
    if use_mixup:
        train_ds = train_ds.map(
            lambda x, y: mixup_batch(x, y, alpha=mixup_alpha),
            num_parallel_calls=AUTOTUNE
        )

    # Step 4: repeat — makes the dataset an infinite stream so model.fit
    # can pull exactly steps_per_epoch batches every epoch without hitting
    # end-of-dataset and skipping even-numbered epochs.
    train_ds = train_ds.repeat()

    # Step 5: ImageNet normalisation — MUST be last, after all augmentation
    train_ds = train_ds.map(
        lambda x, y: ((x - IMAGENET_MEAN) / IMAGENET_STD, y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    # ── Eval pipelines: normalise only, no augmentation ───────────────────────
    def preprocess_eval(image, label):
        """Scale to [0,1] then apply ImageNet normalisation."""
        image = tf.cast(image, tf.float32) / 255.0
        return (image - IMAGENET_MEAN) / IMAGENET_STD, label

    val_ds = (val_ds
              .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
              .cache()
              .prefetch(AUTOTUNE))

    test_ds = (test_ds
               .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
               .cache()
               .prefetch(AUTOTUNE))

    return train_ds, val_ds, test_ds, class_names
