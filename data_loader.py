import tensorflow as tf
from tensorflow.keras import layers

def get_data_loaders(data_dir, target_size=(224, 224), batch_size=32):
    """
    Creates training, validation, and test datasets using the tf.data.Dataset API.
    
    Args:
        data_dir (str): Path to the dataset directory.
        target_size (tuple): Dimensions to resize images.
        batch_size (int): Number of images per batch.
        
    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    
    # Load datasets from directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        label_mode='categorical',
        image_size=target_size,
        batch_size=batch_size,
        shuffle=True
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
    
    class_names = train_ds.class_names
    
    # Data Augmentation Layer (to be applied to training data)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ])
    
    # Preprocessing function: Rescale pixel values
    def preprocess(image, label, augment=False):
        image = tf.cast(image, tf.float32) / 255.0
        if augment:
            image = data_augmentation(image, training=True)
        return image, label

    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.map(lambda x, y: preprocess(x, y, augment=True), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    val_ds = val_ds.map(lambda x, y: preprocess(x, y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    test_ds = test_ds.map(lambda x, y: preprocess(x, y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names
