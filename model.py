import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(num_classes=5, input_shape=(224, 224, 3)):
    """
    Builds the classification model using MobileNetV2 as a base.
    
    Args:
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of input images.
        
    Returns:
        model: Compiled Keras model.
    """
    
    # Load pre-trained MobileNetV2 base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model
