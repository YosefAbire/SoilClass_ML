# Functions & Mathematical Foundations

> Complete reference of every function in the project and the mathematical formulas behind the model.

---

## Table of Contents

1. [File: model.py](#modelpy)
2. [File: data_loader.py](#data_loaderpy)
3. [File: train.py](#trainpy)
4. [File: predict.py](#predictpy)
5. [File: evaluate.py](#evaluatepy)
6. [File: preprocess_dataset.py](#preprocess_datasetpy)
7. [File: split_data.py](#split_datapy)
8. [Mathematical Formulas](#mathematical-formulas)

---

## model.py

### `build_model(num_classes=5, input_shape=(224, 224, 3))`

Constructs the full classification model using MobileNetV2 as a frozen feature extractor with a custom trainable head.

| Parameter | Value | Description |
|---|---|---|
| `num_classes` | 5 | Number of output soil classes |
| `input_shape` | (224, 224, 3) | Height × Width × RGB channels |

**What it does:**
1. Loads MobileNetV2 pre-trained on ImageNet with `include_top=False` (removes the original classifier)
2. Freezes all base layers (`trainable = False`)
3. Attaches a custom head: `GlobalAveragePooling2D → Dense(256, ReLU) → BatchNorm → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.3) → Dense(5, Softmax)`
4. Returns `(model, base_model)` — base_model is returned separately so Phase 2 can unfreeze it

**Returns:** `(model, base_model)`

---

## data_loader.py

### `get_data_loaders(data_dir, target_size=(224, 224), batch_size=32)`

Builds `tf.data` pipelines for train, validation, and test sets with oversampling for minority classes.

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | — | Root directory containing `train/`, `validation/`, `test/` |
| `target_size` | (224, 224) | Resize all images to this shape |
| `batch_size` | 32 | Number of images per batch |

**What it does:**
1. Loads train set unbatched (`batch_size=None`) to allow per-class filtering
2. Counts images per class, computes `repeat_factor = ceil(max_count / class_count)` for each minority class
3. Repeats minority class datasets so all classes reach approximately the same count
4. Uses `tf.data.Dataset.sample_from_datasets` with uniform weights `[1/n_classes] * n_classes` to interleave classes evenly
5. Applies augmentation to training only (flip, rotation, zoom, translation, brightness, contrast)
6. Normalises all pixels: `image / 255.0`
7. Batches, caches, and prefetches for performance

**Returns:** `(train_ds, val_ds, test_ds, class_names)`

#### Inner function: `preprocess_train(image, label)`
Normalises pixel values and applies the augmentation pipeline to a single training sample.

#### Inner function: `preprocess_eval(image, label)`
Normalises pixel values only — no augmentation for validation/test.

---

## train.py

### `compute_class_weights(class_names, data_dir)`

Computes inverse-frequency class weights to penalise misclassification of minority classes more heavily in the loss.

| Parameter | Description |
|---|---|
| `class_names` | List of class name strings |
| `data_dir` | Root dataset directory (uses `train/` subfolder) |

**Formula used:**

```
weight_i = N_total / (N_classes × N_i)
```

Where:
- `N_total` = total training samples across all classes
- `N_classes` = number of classes (5)
- `N_i` = number of training samples for class i

**Returns:** `dict {class_index: weight}` — passed directly to `model.fit(class_weight=...)`

---

### `train_system(data_dir, epochs=20, batch_size=32)`

Orchestrates the full two-phase training pipeline.

**Phase 1 — Head training (base frozen):**
- Optimizer: Adam, lr = 0.001
- Loss: Categorical Cross-Entropy
- Max epochs: 20
- Callbacks: EarlyStopping (patience=6), ModelCheckpoint, ReduceLROnPlateau (factor=0.5, patience=3)
- Class weights applied

**Phase 2 — Fine-tuning (top 30 MobileNetV2 layers unfrozen):**
- Optimizer: Adam, lr = 0.00005
- Same loss and callbacks
- Class weights applied
- Fine-tune epochs: 15

**Returns:** `(history, history_fine)`

---

### `plot_history(history, history_fine)`

Concatenates Phase 1 and Phase 2 history and plots accuracy and loss curves side by side. Saves output to `training_plots.png`.

| Plot | X-axis | Y-axis |
|---|---|---|
| Left | Epoch | Accuracy (train + val) |
| Right | Epoch | Loss (train + val) |

---

## predict.py

### `predict_image(image_path, model_path, class_indices_path)`

Runs inference on a single image file and returns the predicted class with confidence scores.

| Parameter | Default | Description |
|---|---|---|
| `image_path` | — | Path to input image |
| `model_path` | `soil_classifier_final.keras` | Trained model file |
| `class_indices_path` | `class_indices.json` | JSON mapping class names to indices |

**What it does:**
1. Loads the model
2. Inverts the class index map: `{index: class_name}`
3. Loads and resizes image to 224×224
4. Converts to numpy array, adds batch dimension: `shape (1, 224, 224, 3)`
5. Normalises: `array / 255.0`
6. Runs `model.predict()` → softmax probability vector of length 5
7. Takes `argmax` for predicted class, reads confidence from that index
8. Returns full probability dict for all 5 classes

**Returns:**
```python
{
  'class': str,           # predicted soil type
  'confidence': float,    # probability of top class
  'probabilities': dict   # {class_name: probability} for all 5
}
```

---

## evaluate.py

### `evaluate_model(model_path, data_dir)`

Evaluates the trained model on the test set and generates a full performance report.

| Parameter | Default |
|---|---|
| `model_path` | `soil_classifier_final.keras` |
| `data_dir` | `soil_dataset` |

**What it does:**
1. Loads model and test dataset
2. Runs `model.predict()` on all test batches
3. Converts one-hot labels and predictions to class indices via `argmax`
4. Calls `model.evaluate()` for loss and accuracy
5. Generates `classification_report` (precision, recall, F1 per class)
6. Builds and plots confusion matrix, saves to `confusion_matrix.png`

---

## preprocess_dataset.py

### `preprocess_image(input_path, output_path)`

Applies image enhancement to a single file and saves the result.

**Steps:**
1. Open image, convert to RGB
2. Apply Histogram Equalization via `ImageOps.equalize()`
3. Apply Gaussian Blur via `ImageFilter.GaussianBlur(radius=1.0)`
4. Save to output path (creates directories if needed)

**Returns:** `True` on success, `False` on error

---

### `main()`

Walks the entire `soil_dataset/` directory tree (train, validation, test × all classes) and calls `preprocess_image()` on every `.jpg`, `.jpeg`, `.png`, `.bmp` file. Outputs to `preprocessed_soil_dataset/` preserving the same folder structure.

---

## split_data.py

### `split_dataset(data_dir='soil_dataset', val_split=0.15, test_split=0.15)`

Splits images from `train/` into `validation/` and `test/` by physically moving files.

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `soil_dataset` | Root dataset directory |
| `val_split` | 0.15 | Fraction of train images to move to validation |
| `test_split` | 0.15 | Fraction of train images to move to test |

**What it does:**
1. Lists all class subdirectories in `train/`
2. For each class: shuffles image list randomly, computes `num_val = floor(total × 0.15)`, `num_test = floor(total × 0.15)`
3. Moves the first `num_val` images to `validation/<class>/`
4. Moves the next `num_test` images to `test/<class>/`
5. Remaining images stay in `train/<class>/`

> ⚠️ This operation is destructive — it moves files. Run only once on the original dataset.

---

## Mathematical Formulas

### 1. Pixel Normalisation

All images are scaled from `[0, 255]` to `[0, 1]` before being fed to the model:

```
x_norm = x / 255.0
```

---

### 2. ReLU Activation

Applied in Dense(256) and Dense(128) layers:

```
ReLU(x) = max(0, x)
```

Introduces non-linearity. Outputs zero for negative inputs, identity for positive — computationally efficient and avoids vanishing gradients.

---

### 3. Softmax Activation (Output Layer)

Converts raw logits from the final Dense(5) into a probability distribution over 5 classes:

```
Softmax(z_i) = exp(z_i) / Σ exp(z_j)   for j = 1..5
```

All outputs sum to 1.0. The class with the highest value is the prediction.

---

### 4. Categorical Cross-Entropy Loss

The training loss function. Measures how far the predicted probability distribution is from the true one-hot label:

```
L = -Σ y_i × log(ŷ_i)   for i = 1..5
```

Where:
- `y_i` = true label (1 for correct class, 0 for others)
- `ŷ_i` = predicted probability for class i

With class weights applied:

```
L_weighted = -Σ w_i × y_i × log(ŷ_i)
```

Where `w_i = N_total / (N_classes × N_i)` — minority classes get higher weight.

---

### 5. Inverse-Frequency Class Weights

Used to counter class imbalance (arid: 200 samples vs yellow: 980):

```
w_i = N_total / (N_classes × N_i)
```

Example with actual counts:

| Class | N_i | Weight |
|---|---|---|
| alluvial | 487 | ~0.72 |
| arid | 200 | ~1.75 |
| black | 823 | ~0.43 |
| red | 790 | ~0.44 |
| yellow | 980 | ~0.36 |

Arid gets ~4× more weight than yellow — misclassifying it costs more in the loss.

---

### 6. Adam Optimiser

Adaptive Moment Estimation. Updates weights using first and second moment estimates of gradients:

```
m_t = β1 × m_(t-1) + (1 - β1) × g_t          # 1st moment (mean)
v_t = β2 × v_(t-1) + (1 - β2) × g_t²          # 2nd moment (variance)

m̂_t = m_t / (1 - β1^t)                         # bias-corrected
v̂_t = v_t / (1 - β2^t)

θ_t = θ_(t-1) - α × m̂_t / (√v̂_t + ε)
```

Default values used: `β1=0.9`, `β2=0.999`, `ε=1e-7`
- Phase 1 learning rate `α = 0.001`
- Phase 2 learning rate `α = 0.00005`

---

### 7. ReduceLROnPlateau

Halves the learning rate when `val_loss` does not improve for 3 consecutive epochs:

```
α_new = α × factor   (factor = 0.5)
```

Minimum LR floor: `1e-6` (Phase 1), `1e-7` (Phase 2).

---

### 8. Dropout Regularisation

Randomly sets a fraction of neuron outputs to zero during training to prevent co-adaptation:

```
y_i = x_i × Bernoulli(1 - p) / (1 - p)
```

Where `p` is the dropout rate. Used at:
- `p = 0.4` after Dense(256)
- `p = 0.3` after Dense(128)

At inference time, dropout is disabled and all neurons are active.

---

### 9. Batch Normalisation

Applied after Dense(256) to stabilise training, especially important with imbalanced classes:

```
x̂ = (x - μ_B) / √(σ²_B + ε)
y  = γ × x̂ + β
```

Where:
- `μ_B`, `σ²_B` = batch mean and variance
- `γ`, `β` = learnable scale and shift parameters
- `ε = 1e-3` (small constant for numerical stability)

---

### 10. GlobalAveragePooling2D

Reduces the MobileNetV2 feature map from shape `(7, 7, 1280)` to `(1280,)` by averaging spatially:

```
GAP(F)_k = (1 / H×W) × Σ_i Σ_j F_ijk
```

Where `H=7`, `W=7`, `k` indexes the 1280 feature channels. More parameter-efficient than Flatten and reduces overfitting.

---

### 11. Evaluation Metrics

**Accuracy:**
```
Accuracy = (TP + TN) / N_total
```

**Precision (per class):**
```
Precision_i = TP_i / (TP_i + FP_i)
```

**Recall (per class):**
```
Recall_i = TP_i / (TP_i + FN_i)
```

**F1-Score (per class):**
```
F1_i = 2 × (Precision_i × Recall_i) / (Precision_i + Recall_i)
```

**Macro F1** — unweighted mean across all classes (treats arid equally to yellow):
```
F1_macro = (1/C) × Σ F1_i
```

**Weighted F1** — weighted by class support (number of true instances):
```
F1_weighted = Σ (support_i / N_total) × F1_i
```

---

### 12. Histogram Equalization

Applied in `preprocess_dataset.py` to improve contrast. Redistributes pixel intensities so the histogram is approximately uniform:

```
h(v) = round( (CDF(v) - CDF_min) / (N_pixels - CDF_min) × (L - 1) )
```

Where:
- `CDF(v)` = cumulative distribution function of pixel values up to intensity `v`
- `L` = number of intensity levels (256)
- `N_pixels` = total number of pixels

---

### 13. Gaussian Blur

Applied in `preprocess_dataset.py` to reduce noise. Convolves the image with a 2D Gaussian kernel:

```
G(x, y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)
```

Used with `radius=1.0` (σ ≈ 1.0). Each output pixel is a weighted average of its neighbourhood, with closer pixels weighted more.

---

### 14. Oversampling (Repeat Factor)

Used in `data_loader.py` to balance minority classes before training:

```
repeat_factor_i = ceil(N_max / N_i)
```

Where `N_max` is the count of the largest class (yellow: 980). For arid (200):

```
repeat_factor_arid = ceil(980 / 200) = 5
```

The arid dataset is repeated 5× so it contributes equally to training batches.
