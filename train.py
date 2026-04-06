import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_loader import get_data_loaders
from model import build_model

def compute_class_weights(class_names, data_dir):
    """
    Computes inverse-frequency class weights to counter imbalance.
    Returns a dict {index: weight} for use in model.fit().
    """
    counts = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, 'train', cls)
        counts.append(len(os.listdir(cls_dir)))
    counts = np.array(counts, dtype=np.float32)
    total  = counts.sum()
    n_cls  = len(counts)
    # sklearn-style: weight = total / (n_classes * count)
    weights = total / (n_cls * counts)
    print("\nClass weights (to counter imbalance):")
    for i, (cls, w) in enumerate(zip(class_names, weights)):
        print(f"  {cls:12}: {counts[i]:4.0f} samples  →  weight {w:.4f}")
    return {i: float(w) for i, w in enumerate(weights)}


def train_system(data_dir='preprocessed_soil_dataset', epochs=20, batch_size=32):
    """
    Main training script with class-weight balancing.
    """

    # 1. Load Data
    print("Loading data...")
    train_ds, val_ds, test_ds, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    
    # Save class indices for prediction later
    class_indices = {name: i for i, name in enumerate(class_names)}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)

    # 2. Compute class weights from the raw (non-preprocessed) dataset counts
    raw_dir = data_dir if os.path.exists(os.path.join(data_dir, 'train')) else 'soil_dataset'
    class_weight = compute_class_weights(class_names, raw_dir)

    # 3. Build Model
    print("\nBuilding model...")
    model, base_model = build_model(num_classes=len(class_names))

    # 4. Compile — Phase 1
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint('soil_classifier_initial.keras', monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                          min_lr=1e-6, verbose=1),
    ]

    # 6. Phase 1 — frozen base
    print("\nPhase 1: Training classification head (base frozen)...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=class_weight,   # ← key fix
        callbacks=callbacks
    )

    # 7. Phase 2 — fine-tune top layers
    print("\nPhase 2: Fine-tuning top 30 MobileNetV2 layers...")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30   # was 20, now 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.00005),   # lower than before (1e-4 → 5e-5)
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_epochs = 15
    total_epochs = epochs + fine_tune_epochs

    callbacks_ft = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint('soil_classifier_final.keras', monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='va

def plot_history(history, history_fine):
    """
    Plots training and validation metrics.
    """
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.savefig('training_plots.png')
    plt.show()

if __name__ == "__main__":
    # Ensure dataset exists or provide instructions
    if os.path.exists('preprocessed_soil_dataset'):
        h1, h2 = train_system()
        plot_history(h1, h2)
    else:
        if os.path.exists('soil_dataset'):
            print("Error: 'preprocessed_soil_dataset' not found.")
            print("Please run 'python preprocess_dataset.py' first to generate it.")
        else:
            print("Error: 'soil_dataset' directory not found.")
            print("Please organize your data as: soil_dataset/train, soil_dataset/validation, soil_dataset/test")
