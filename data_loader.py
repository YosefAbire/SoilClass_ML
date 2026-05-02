"""
data_loader.py — Augmentation pipeline optimised for 4-class balanced soil dataset.

Key changes for 70-80% accuracy push:
  - Strong colour jitter (brightness ±30%, contrast ±30%, saturation via
    channel scaling ±25%) forces the model to learn TEXTURE not colour.
    Alluvial/Red confusion is colour-based — this directly addresses it.
  - Gaussian noise (stddev=0.03) prevents pixel-level memorisation.
  - Random sharpening emphasises grain size differences between classes.
  - All classes get the same strong augmentation — dataset is balanced.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

MINORITY_CLASSES = set()   # all 4 classes equalised at 480 — no oversampling

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def build_strong_aug():
    """
    Strong augmentation for all 4 classes.

    Design rationale for 70-80% accuracy:
    - Alluvial is confused with Red/Yellow because they share warm tones.
      Wide colour jitter (±30%) forces the model to ignore colour and
      learn grain size, surface texture, and structural patterns instead.
    - Gaussian noise (stddev=0.03) prevents memorising exact pixel values
      from the 480 training images.
    - Random sharpening emphasises texture edges — the primary discriminating
      feature between alluvial (fine silt), black (clay), red (coarse grain).
    - Wider spatial transforms (rotation ±25°, zoom ±20%) create more
      geometric variety from the limited 480-image pool.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.20),
        layers.RandomTranslation(0.12, 0.12),

        # Wide colour jitter — force texture-based learning, not colour
        layers.RandomBrightness(0.30),
        layers.RandomContrast(0.30),

        # Per-channel saturation simulation (colour jitter)
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x * tf.random.uniform([1, 1, 3], 0.75, 1.25), 0.0, 1.0
            ), name="channel_jitter"
        ),

        # Gaussian noise — prevents pixel memorisation
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.03),
                0.0, 1.0
            ), name="gaussian_noise"
        ),

        # Random sharpening — emphasises grain/texture edges
        # Uses avg_pool on the image directly (already rank 4: B,H,W,C)
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x + tf.random.uniform([], 0.0, 0.25) * (
                    x - tf.nn.avg_pool2d(x, ksize=3, strides=1, padding='SAME')
                ), 0.0, 1.0
            ), name="random_sharpen"
        ),
    ], name="strong_aug")


def get_data_loaders(data_dir, target_size=(224, 224), batch_size=32, **kwargs):
    """
    Builds tf.data pipelines for train / validation / test.

    Training pipeline:
      uint8 → [0,1] → strong augmentation → ImageNet normalise → repeat → prefetch

    Val/Test pipeline:
      uint8 → [0,1] → ImageNet normalise → cache → prefetch
    """
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        label_mode='categorical',
        image_size=target_size,
        batch_size=batch_size,
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
    train_dir   = f"{data_dir}/train"
    counts = {cls: len(os.listdir(os.path.join(train_dir, cls)))
              for cls in class_names}

    print("\nDataset class counts:")
    for cls, n in counts.items():
        print(f"  {cls:14}: {n:4d} images")

    aug = build_strong_aug()

    def preprocess_train(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = aug(image, training=True)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image, label

    def preprocess_eval(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return (image - IMAGENET_MEAN) / IMAGENET_STD, label

    train_ds = (train_ds_raw
                .map(preprocess_train, num_parallel_calls=AUTOTUNE)
                .repeat()
                .prefetch(AUTOTUNE))

    val_ds = (val_ds
              .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
              .cache()
              .prefetch(AUTOTUNE))

    test_ds = (test_ds
               .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
               .cache()
               .prefetch(AUTOTUNE))

    return train_ds, val_ds, test_ds, class_names
