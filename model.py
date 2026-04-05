import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def build_model(num_classes=5, input_shape=(224, 224, 3)):
    """
    MobileNetV2-based transfer learning model with BatchNorm for better
    generalisation on imbalanced classes.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)   # wider than before (128 → 256)
    x = BatchNormalization()(x)             # stabilises training on minority classes
    x = Dropout(0.4)(x)                    # slightly lower dropout (0.5 → 0.4)
    x = Dense(128, activation='relu')(x)   # extra layer for more capacity
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model
