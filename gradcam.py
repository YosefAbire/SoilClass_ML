import tensorflow as tf
import numpy as np
import cv2
from losses import FocalLoss   # needed so load_model can deserialise the saved model

def find_last_conv_layer(model):
    """Automatically finds the last 4D convolutional layer."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("Could not find a 4D convolutional layer.")

def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global Average Pooling of the gradients (weight of each feature map)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by its 'importance'
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to RGB [0, 255]
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Superimpose
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    return superimposed_img