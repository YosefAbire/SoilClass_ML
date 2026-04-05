import os
import sys
import numpy as np
import json

# Suppress TF logging to keep stdout clean for JSON parsing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict_image(image_path, model_path='soil_classifier_final.keras', class_indices_path='class_indices.json'):
    """
    Predicts the class of a single image and returns probabilities for all classes.
    
    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the trained .keras model.
        class_indices_path (str): Path to the JSON file with class mapping.
        
    Returns:
        dict: {
            'predicted_class': str,
            'confidence': float,
            'all_probabilities': {class_name: float}
        }
    """
    
    # 1. Load Model
    model = tf.keras.models.load_model(model_path)
    
    # 2. Load Class Indices
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Invert mapping: {index: class_name}
    labels = {v: k for k, v in class_indices.items()}
    
    # 3. Preprocess Image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    
    # 4. Predict
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index])
    predicted_class = labels[predicted_index]
    
    # 5. Map all probabilities
    all_probs = {labels[i]: float(prob) for i, prob in enumerate(predictions)}
    
    result = {
        'class': predicted_class,
        'confidence': confidence,
        'probabilities': all_probs
    }
    
    if '--json' not in sys.argv:
        print(f"\nPrediction Results:")
        print(f"Top Class: {predicted_class} ({confidence:.4f})")
        print("-" * 30)
        # Print sorted by probability
        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"{cls:10}: {prob:.4f}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        img_path = sys.argv[1]
        results = predict_image(img_path)
        if '--json' in sys.argv:
            print(json.dumps(results))
    else:
        print("Usage: python predict.py <path_to_image.jpg> [--json]")
