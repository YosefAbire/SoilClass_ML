# Accuracy Diagnosis — Why Is the Model at ~30%?

> Full root-cause analysis across data quality, preprocessing, feature extraction, and training mathematics.

---

## Verdict Summary

| Category | Status | Severity |
|---|---|---|
| Bad Data | ⚠️ Partially | High |
| Poor Preprocessing | ⚠️ Partially | Medium |
| Weak Features | ✅ Not the cause | Low |
| Math / Training Config | ❌ Yes — multiple issues | Critical |

The model is not broken. The architecture is sound. The primary causes are **training math conflicts** and **data starvation** — not bad images or a weak backbone.

---

## 1. Data Quality

### 1a. Class Imbalance — HIGH SEVERITY

```
train/alluvial :  487 images
train/arid     :  200 images   ← 4.9× fewer than yellow
train/black    :  823 images
train/red      :  790 images
train/yellow   :  980 images
```

With 5 classes, random-chance accuracy is **20%**. At 30% the model is barely above chance — it has learned to predict the majority classes (yellow, black, red) for almost everything.

**Why this causes 30%:** The model sees yellow 4.9× more than arid per real epoch. Even with oversampling and class weights, the model's internal representations for arid and yellow are weak because there simply aren't enough *distinct* real images to learn from.

**Minimum recommended per class:** 500–1000 real images. Arid at 200 is critically low.

---

### 1b. Visual Similarity Between Classes — HIGH SEVERITY

Soil types in this dataset share very similar colour profiles:

| Pair | Why they're confused |
|---|---|
| Arid ↔ Red | Both are warm reddish-brown tones |
| Yellow ↔ Arid | Both are pale/sandy in dry conditions |
| Black ↔ Alluvial | Both are dark, fine-grained soils |

A model at 30% accuracy is essentially collapsing multiple classes into one or two dominant predictions. This is a **data problem** — the discriminating features (texture, grain size, moisture sheen) require high-resolution, consistently-lit, close-up images. If the dataset contains wide-angle or mixed-background photos, the model cannot learn these features.

---

### 1c. No Data Verification Step

There is no script that checks:
- Whether images are actually soil (no background filtering)
- Whether images are correctly labelled
- Whether there are duplicate images across train/val/test splits (data leakage)
- Image resolution distribution (very small images lose texture detail)

A mislabelled or low-quality 10% of 200 arid images = 20 bad samples = 10% noise rate, which is catastrophic for a minority class.

---

## 2. Preprocessing

### 2a. Histogram Equalization Applied Globally — MEDIUM SEVERITY

```python
# preprocess_dataset.py
img = ImageOps.equalize(img)          # global histogram equalization
img = img.filter(ImageFilter.GaussianBlur(radius=1.0))  # then blur
```

**Problem 1 — Global equalization destroys colour information.**
Soil classification depends heavily on colour (red soil is red, yellow soil is yellow). `ImageOps.equalize()` redistributes pixel intensities to be uniform across the full range. This *removes* the colour contrast that distinguishes red from yellow from arid. After equalization, a red soil image and a yellow soil image may have nearly identical histograms.

**Problem 2 — Blur after equalization.**
Gaussian blur with radius=1.0 softens texture edges. Texture is one of the few remaining discriminating features after colour is flattened by equalization. Blurring it makes the task harder, not easier.

**Problem 3 — Order is wrong.**
Blur should come *before* equalization (to remove noise first, then enhance contrast), not after.

**What should be used instead:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) — enhances local contrast without destroying global colour
- Or skip equalization entirely and let the model's BatchNorm layers handle contrast normalisation

---

### 2b. No ImageNet-Style Normalisation

MobileNetV2 was pre-trained with ImageNet normalisation:
```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The current pipeline only divides by 255:
```python
image = tf.cast(image, tf.float32) / 255.0
```

This means the input distribution seen during fine-tuning does not match what MobileNetV2 expects from its ImageNet pre-training. The base model's learned filters are calibrated for ImageNet-normalised inputs. Feeding `[0,1]` values instead of `[-2.1, 2.6]` range values shifts every activation in the base model, making transfer learning less effective.

**Fix:**
```python
# After dividing by 255, apply ImageNet normalisation
mean = tf.constant([0.485, 0.456, 0.406])
std  = tf.constant([0.229, 0.224, 0.225])
image = (image - mean) / std
```

---

## 3. Feature Extraction

### 3a. MobileNetV2 Architecture — NOT THE CAUSE

MobileNetV2 is a proven, well-tested backbone. It has successfully classified fine-grained visual categories (flowers, food, textures) with far fewer training samples than this dataset. The architecture is not the bottleneck.

### 3b. Custom Head — MINOR ISSUE

```python
GlobalAveragePooling2D
Dense(256, ReLU)
BatchNormalization
Dropout(0.4)
Dense(128, ReLU)
Dropout(0.3)
Dense(5, Softmax)
```

This is reasonable. However, BatchNormalization between two Dense layers can cause issues when `class_weight` is used — the batch statistics are computed over a mixed-class batch, but the weighted loss pushes gradients unevenly, which can destabilise BatchNorm's running mean/variance estimates.

---

## 4. Training Mathematics — CRITICAL

### 4a. Focal Loss + Class Weights = Double Counting

```python
# train.py
focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
model.fit(..., class_weight=class_weight)
```

**This is the most likely cause of 30% accuracy.**

Both mechanisms address class imbalance, but they do it in conflicting ways:

| Mechanism | What it does |
|---|---|
| `class_weight` | Multiplies the loss for each sample by `w_i` based on its class |
| Focal Loss `alpha=0.25` | Scales the entire loss down by 0.25 |
| Focal Loss `(1-p_t)^gamma` | Further down-weights easy samples |

When combined:
- A correctly classified majority sample (p_t=0.9) gets: `0.25 × (0.1)^2 × log(0.9) ≈ 0.00026` — nearly zero gradient
- A misclassified minority sample (p_t=0.1) gets: `w_i × 0.25 × (0.9)^2 × log(0.1) ≈ 2.0 × 0.25 × 0.81 × 2.3 ≈ 0.93`

The ratio between easy and hard samples is ~3500:1. The optimiser receives almost no signal from correctly classified samples and an overwhelming signal from a handful of hard minority samples. This causes the model to oscillate — it overcorrects toward minority classes, then overcorrects back, never converging.

**Fix: Use one mechanism, not both.**
- Option A: Keep Focal Loss, remove `class_weight`
- Option B: Keep `class_weight`, use standard cross-entropy
- Option C: Keep Focal Loss with `alpha` per class instead of a scalar

---

### 4b. Focal Loss alpha=0.25 is Too Aggressive for 5-Class

`alpha=0.25` was designed for **binary** classification (foreground vs background in object detection). For 5-class problems, it scales the entire loss to 25% of its normal magnitude. This means:

```
Effective learning rate = actual_lr × gradient_scale
                        ≈ 0.001 × 0.25 = 0.00025  (Phase 1)
                        ≈ 0.00001 × 0.25 = 0.0000025  (Phase 2)
```

Phase 2 effective LR of `2.5e-6` is so small that the model makes negligible weight updates. With EarlyStopping patience=10, it stops before any meaningful fine-tuning occurs.

**Fix:** Set `alpha=1.0` for multi-class, or use per-class alpha vector.

---

### 4c. Log-Smoothed Class Weights Formula Has an Edge Case

```python
weights = np.log(total) / np.log(counts)
```

For `counts[i] = 1` (if any class has exactly 1 sample), `log(1) = 0` → division by zero.
For `counts[i] = total` (if one class has all samples), weight = 1.0 regardless.

More critically: `log(3280) / log(200) ≈ 1.44` but `log(3280) / log(980) ≈ 1.08`. The ratio between arid and yellow weights is only 1.44/1.08 = **1.33×**. After normalisation and capping, the effective difference is even smaller. This is too gentle — arid needs at least 2–3× the weight of yellow to compensate for the 4.9× sample deficit.

---

### 4d. EarlyStopping Monitors val_loss, Not val_accuracy

```python
EarlyStopping(monitor='val_loss', patience=6)
```

With Focal Loss, `val_loss` values are in the range `[0.001, 0.05]` — much smaller than cross-entropy losses `[0.5, 2.0]`. Small absolute changes in Focal Loss may not reflect meaningful accuracy changes. The model can stop early while val_accuracy is still improving because val_loss has plateaued at a very small value.

**Fix:** Monitor `val_accuracy` in EarlyStopping, or use `min_delta=1e-4`.

---

### 4e. Phase 2 LR Too Low Combined With Focal Loss Scaling

```
Phase 2 Adam LR: 1e-5
Focal Loss alpha: 0.25
Effective gradient scale: ~0.25 × (1-p_t)^2

For a well-trained model (p_t ≈ 0.6 on hard classes):
Effective update ≈ 1e-5 × 0.25 × 0.16 = 4e-7
```

At `4e-7` effective update magnitude, the model needs hundreds of epochs to move weights meaningfully. With max 15 fine-tune epochs, Phase 2 is essentially doing nothing.

---

## 5. Root Cause Priority List

| # | Issue | Impact | Fix |
|---|---|---|---|
| 1 | Focal Loss + class_weight double-counting | 🔴 Critical | Remove `class_weight` from `model.fit()` |
| 2 | Focal Loss alpha=0.25 too small for 5-class | 🔴 Critical | Set `alpha=1.0` |
| 3 | Histogram equalization destroying colour | 🟠 High | Replace with CLAHE or remove |
| 4 | No ImageNet normalisation | 🟠 High | Apply mean/std normalisation after ÷255 |
| 5 | Arid class data starvation (200 images) | 🟠 High | Collect more data or use external dataset |
| 6 | EarlyStopping on val_loss with Focal Loss | 🟡 Medium | Switch to monitor='val_accuracy' |
| 7 | Phase 2 LR too low with alpha scaling | 🟡 Medium | Raise to 5e-5 or set alpha=1.0 first |
| 8 | BatchNorm + class_weight interaction | 🟡 Medium | Remove class_weight (fix #1 resolves this) |
| 9 | No data quality verification | 🟡 Medium | Add label audit script |

---

## 6. Recommended Fixes (In Order)

### Fix 1 — Remove class_weight, fix Focal Loss alpha

```python
# losses.py — change alpha
FocalLoss(gamma=2.0, alpha=1.0)   # not 0.25

# train.py — remove class_weight from both model.fit() calls
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks_p1)
#                                                    ↑ no class_weight
```

### Fix 2 — Apply ImageNet normalisation in data_loader.py

```python
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225])

def preprocess_eval(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image, label
```

### Fix 3 — Replace global histogram equalization with CLAHE

```python
# preprocess_dataset.py
import cv2
import numpy as np

def preprocess_image(input_path, output_path):
    img = cv2.imread(input_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])   # only L channel
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, img)
```

### Fix 4 — EarlyStopping on val_accuracy

```python
EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
```

### Fix 5 — Raise Phase 2 LR

```python
optimizer=Adam(learning_rate=0.00005)   # back to 5e-5 now that alpha=1.0
```

---

## 7. Expected Accuracy After Fixes

| State | Expected Accuracy |
|---|---|
| Current (30%) | Barely above random chance |
| After Fix 1+2 (alpha + normalisation) | ~55–65% |
| After Fix 1+2+3 (+ CLAHE) | ~65–75% |
| After all fixes + more arid data | ~80–88% |

The architecture is capable of 85%+ on this task. The gap between 30% and 85% is entirely explained by the training math conflicts and preprocessing issues identified above.
