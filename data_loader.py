import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# ── Minority classes that need stronger augmentation ─────────────────────────
MINORITY_CLASSES = {'arid', 'yellow'}


def get_data_loaders(data_dir, target_size=(224, 224), batch_size=32):
    """
    Builds tf.data pipelines for train / validation / test.

    Minority class handling (Arid, Yellow):
    - Each minority image is repeated up to repeat_factor times BUT each
      repetition goes through a DIFFERENT random augmentation path, so the
      model never sees the exact same pixel values twice in an epoch.
    - Minority classes (arid, yellow) get a stronger augmentation policy
      (extra hue/saturation jitter, sharpness, coarser rotation/zoom) to
      maximise variety from the limited pool of real images.
    - Majority classes get a standard augmentation policy.
    """

    # ── Load raw datasets (unbatched for per-sample processing) ──────────────
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        label_mode='categorical',
        image_size=target_size,
        batch_size=None,
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

    print("\nDataset class counts:")
    for cls, n in zip(class_names, counts):
        tag = " ← minority" if cls in MINORITY_CLASSES else ""
        print(f"  {cls:12}: {int(n):4d} images{tag}")

    # ── Augmentation policies ─────────────────────────────────────────────────

    # Standard policy — majority classes
    standard_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.25),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="standard_aug")

    # Strong policy — minority classes (Arid, Yellow)
    # Extra layers create more pixel-level variety from the same source image.
    strong_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.35),          # wider rotation range
        layers.RandomZoom(0.35),              # wider zoom range
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomBrightness(0.35),        # stronger brightness shift
        layers.RandomContrast(0.35),          # stronger contrast shift
        # Simulate colour/hue variation via channel-wise scaling
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x * tf.random.uniform([1, 1, 3], 0.75, 1.25), 0.0, 1.0
            ),
            name="channel_scale"
        ),
        # Random sharpening via a random blend with a sharpened version
        layers.Lambda(
            lambda x: tf.clip_by_value(
                x + tf.random.uniform([], 0.0, 0.15) * (
                    x - tf.nn.avg_pool2d(
                        tf.expand_dims(x, 0), ksize=3, strides=1, padding='SAME'
                    )[0]
                ),
                0.0, 1.0
            ),
            name="random_sharpen"
        ),
    ], name="strong_aug")

    # ── Build per-class datasets with appropriate augmentation ───────────────
    class_datasets = []
    for i, cls in enumerate(class_names):
        is_minority = cls in MINORITY_CLASSES
        aug_fn      = strong_aug if is_minority else standard_aug

        # Filter to this class
        cls_ds = train_ds_raw.filter(
            lambda x, y, idx=i: tf.equal(tf.argmax(y), idx)
        )

        # Normalise first (augmentation operates on [0,1] floats)
        cls_ds = cls_ds.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if is_minority:
            # repeat_factor: how many times we need to loop the class dataset
            repeat_factor = int(np.ceil(max_count / counts[i]))
            print(f"  {cls:12}: repeat_factor={repeat_factor} "
                  f"(~{int(counts[i] * repeat_factor)} virtual samples, strong aug)")

            # Key fix: shuffle BEFORE repeat so each pass sees a different order,
            # then apply augmentation AFTER repeat so every copy gets a fresh
            # random transformation — not the same one baked in before repeating.
            cls_ds = (cls_ds
                      .shuffle(buffer_size=int(counts[i]), reshuffle_each_iteration=True)
                      .repeat(repeat_factor)
                      .map(lambda x, y: (aug_fn(x, training=True), y),
                           num_parallel_calls=tf.data.AUTOTUNE))
        else:
            cls_ds = (cls_ds
                      .shuffle(buffer_size=int(counts[i]), reshuffle_each_iteration=True)
                      .map(lambda x, y: (aug_fn(x, training=True), y),
                           num_parallel_calls=tf.data.AUTOTUNE))

        class_datasets.append(cls_ds)

    # Interleave all class streams with uniform sampling weights
    train_ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=[1.0 / n_classes] * n_classes,
        seed=42,
        stop_on_empty_dataset=False   # keep sampling from non-exhausted classes
    )

    # ── Eval preprocessing (normalise only, no augmentation) ─────────────────
    def preprocess_eval(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (train_ds
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
