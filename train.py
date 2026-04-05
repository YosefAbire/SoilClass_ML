import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import get_data_loaders
from model import build_model

def train_system(data_dir='preprocessed_soil_dataset', epochs=15, batch_size=32):
    """
    Main training script.
    """
    
    # 1. Load Data
    print("Loading data...")
    train_ds, val_ds, test_ds, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    
    # Save class indices for prediction later
    class_indices = {name: i for i, name in enumerate(class_names)}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    
    # 2. Build Model
    print("Building model...")
    model, base_model = build_model(num_classes=len(class_names))
    
    # 3. Compile Model (Initial Training)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 4. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('soil_classifier_initial.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    # 5. Initial Training (Frozen Base)
    print("Starting initial training...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # 6. Fine-tuning (Unfreeze top layers)
    print("Starting fine-tuning...")
    base_model.trainable = True
    
    # Freeze all layers except the last 20
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Recompile with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Fine-tune
    fine_tune_epochs = 10
    total_epochs = epochs + fine_tune_epochs
    
    history_fine = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # 7. Save Final Model
    model.save('soil_classifier_final.keras')
    print("Model saved as soil_classifier_final.keras")
    
    return history, history_fine

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
