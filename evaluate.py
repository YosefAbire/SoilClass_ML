import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_loaders

def evaluate_model(model_path='soil_classifier_final.keras', data_dir='soil_dataset'):
    """
    Evaluates the model on the test set and generates reports.
    """
    
    # 1. Load Model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Load Test Data
    _, _, test_ds, class_labels = get_data_loaders(data_dir)
    
    # 3. Predictions
    print("Generating predictions...")
    predictions = model.predict(test_ds)
    y_pred = np.argmax(predictions, axis=1)
    
    # Extract true labels from the dataset
    y_true = []
    for _, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)
    
    # 4. Accuracy
    loss, accuracy = model.evaluate(test_ds)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    
    # 5. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
