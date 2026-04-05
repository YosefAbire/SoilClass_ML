import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

def get_data_loaders(data_dir, target_size=(224, 224), batch_size=32):
    """
    Creates training, validation, and test datasets using the tf.data.Dataset API.
    Minority classes (e.g. arid) are oversampled in the training set to reduce bias.
    """

    # ── Load raw datasets ────────────────────────────────────────────────────
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        label_mode='categorical',
        image_size=target_size,
        batch_size=None,          # unbatched — needed for per-class oversampling
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

    # ── Count samples per class ───────────────────────────────────────────────
    train_dir = f"{data_dir}/train"
    counts = np.array([
        len(os.listdir(os.path.join(train_dir, cls)))
        for cls in class_names
    ], dtype=np.float32)
    max_count = counts.max()

    # ── Build one dataset per class, then resample ───────────────────────────
    class_datasets = []
    for i, cls in enumerate(class_names):
        # filter to this class
        cls_ds = train_ds_raw.filter(
            lambda x, y, idx=i: tf.equal(tf.argmax(y), idx)
        )
        # repeat minority classes so every class reaches ~max_count
        repeat_factor = int(np.ceil(max_count / counts[i]))
        if repeat_factor > 1:
            cls_ds = cls_ds.repeat(repeat_factor)
        class_datasets.append(cls_ds)

    # sample_from_datasets interleaves all class streams uniformly
    train_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0 / n_classes] * n_classes,
        seed=42
    )

    # ── Augmentation (training only) ─────────────────────────────────────────
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.25),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    def preprocess_train(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = data_augmentation(image, training=True)
        return image, label

    def preprocess_eval(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (train_ds
                .map(preprocess_train, num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
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
