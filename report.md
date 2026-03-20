# CIFAR-10 Classification — Experimental Report

## Project Overview

This project presents a progressive deep learning study on the CIFAR-10 image classification dataset. Starting from a simple baseline CNN, a series of controlled experiments were conducted — each introducing a single focused architectural or training improvement — to systematically increase test accuracy from **71.21% to 90.08%**, a total absolute gain of **+18.87 percentage points**. The final model is a ResNet-style CNN with residual (skip) connections, deployed as a FastAPI web application.

---

## Dataset

**CIFAR-10** is a standard benchmark dataset in computer vision.

| Property           | Value                                                              |
|--------------------|--------------------------------------------------------------------|
| Total images       | 60,000 RGB images                                                  |
| Training samples   | 50,000                                                             |
| Test samples       | 10,000                                                             |
| Image dimensions   | 32 × 32 pixels, 3 channels (RGB)                                  |
| Number of classes  | 10                                                                 |
| Classes            | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Format             | Directory-based (one subfolder per class)                          |
| Normalization      | Pixel values scaled from [0, 255] to [0, 1] via `rescale=1./255`  |

---

## Experiment 0 — Data Preprocessing & EDA

**File:** `experiment/0-preprocessing.ipynb`

This notebook establishes the baseline data pipeline that is reused across all subsequent experiments.

### Data Pipeline

```python
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../data/cifar10/train',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    '../data/cifar10/test',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)
```

### Class Index Mapping

```
{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
```

### Key Preprocessing Decisions

| Parameter       | Value          | Notes                                      |
|-----------------|----------------|--------------------------------------------|
| rescale         | 1./255         | Normalizes pixel values to [0, 1]          |
| target_size     | (32, 32)       | Matches native CIFAR-10 resolution         |
| batch_size      | 32             | Used consistently across all experiments   |
| class_mode      | categorical    | Produces one-hot encoded label vectors     |
| Augmentation    | None (this notebook) | Introduced in Experiment 2           |

### Exploratory Data Analysis

A 3×3 grid of 9 sample training images was visualized using Matplotlib. Labels were shown as integer class indices. All images were already normalized to [0, 1] by the generator at display time.

### Hardware Context

All experiments up to Experiment 6 were run on CPU only (no CUDA GPU available). Experiment 7 (ResNet) was run on Kaggle with 2× Tesla T4 GPUs.

---

## Experiment 1 — Baseline CNN

**File:** `experiment/1-baseline_model.ipynb`
**Test Accuracy: 71.21%**

### Objective

Establish a simple, well-defined CNN baseline on CIFAR-10 with no regularization, no augmentation, and minimal configuration. This serves as the reference point for all subsequent comparisons.

### Model Architecture

| Layer              | Type            | Filters / Units | Kernel | Activation | Output Shape |
|--------------------|-----------------|-----------------|--------|------------|--------------|
| Conv2D (Block 1)   | Convolutional   | 32              | 3×3    | ReLU       | 30×30×32     |
| MaxPooling2D       | Pooling         | —               | 2×2    | —          | 15×15×32     |
| Conv2D (Block 2)   | Convolutional   | 64              | 3×3    | ReLU       | 13×13×64     |
| MaxPooling2D       | Pooling         | —               | 2×2    | —          | 6×6×64       |
| Conv2D (Block 3)   | Convolutional   | 128             | 3×3    | ReLU       | 4×4×128      |
| Flatten            | Flatten         | —               | —      | —          | 2048         |
| Dense              | Fully Connected | 128             | —      | ReLU       | 128          |
| Dense (output)     | Fully Connected | 10              | —      | Softmax    | 10           |

- 3 convolutional blocks with progressively increasing filters (32 → 64 → 128)
- No Batch Normalization, no Dropout, no weight decay
- `input_shape=(32, 32, 3)`

### Compilation

| Parameter  | Value                    |
|------------|--------------------------|
| Optimizer  | Adam (default lr=0.001)  |
| Loss       | categorical_crossentropy |
| Metric     | accuracy                 |

### Training Configuration

| Parameter        | Value |
|------------------|-------|
| Epochs           | 10    |
| Batch size       | 32    |
| Steps/epoch      | 1563  |
| Validation steps | 313   |
| Callbacks        | None  |
| Augmentation     | None  |

### Per-Epoch Training Log

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | Time/epoch |
|-------|-----------|------------|---------|----------|------------|
| 1     | 0.4530    | 1.5021     | 0.5418  | 1.2985   | 67s        |
| 2     | 0.6034    | 1.1277     | 0.6275  | 1.0370   | 40s        |
| 3     | 0.6638    | 0.9595     | 0.6388  | 1.0346   | 42s        |
| 4     | 0.7049    | 0.8452     | 0.6831  | 0.9256   | 41s        |
| 5     | 0.7321    | 0.7622     | 0.6656  | 0.9751   | 42s        |
| 6     | 0.7572    | 0.6895     | 0.6962  | 0.9002   | 42s        |
| 7     | 0.7832    | 0.6230     | 0.7122  | 0.8770   | 42s        |
| 8     | 0.8033    | 0.5612     | 0.7158  | 0.8696   | 38s        |
| 9     | 0.8219    | 0.5050     | 0.7042  | 0.9220   | 39s        |
| 10    | 0.8412    | 0.4505     | 0.7121  | 0.9473   | 41s        |

### Final Results

| Metric              | Value  |
|---------------------|--------|
| Train Accuracy      | 84.12% |
| Train Loss          | 0.4505 |
| Test Accuracy       | 71.21% |
| Test Loss           | 0.9473 |
| Train/Val Gap       | ~13%   |

### Analysis

**Accuracy curves:** Training accuracy rose steadily from 45% to ~84%. Validation accuracy peaked at ~71–72% and stopped improving after epoch 6–7.

**Loss curves:** Training loss decreased smoothly throughout all 10 epochs. Validation loss decreased early, then began rising slightly after epoch 6–7 — indicating overfitting onset.

**Diagnosis:**

| Observation                        | Meaning                        |
|------------------------------------|--------------------------------|
| Train accuracy rises continuously  | Model has sufficient capacity  |
| Val accuracy plateaus at epoch 6–7 | Generalization limit reached   |
| Val loss rises after epoch 6       | Overfitting begins             |
| ~13% train/val gap                 | Strong regularization needed   |

**Conclusion:** The baseline model demonstrates that the architecture has enough representational power to learn the training data. The primary failure is overfitting — the model memorizes training examples instead of learning generalizable features. The prescribed next step is data augmentation.

---

## Experiment 2 — Data Augmentation

**File:** `experiment/2-augmentation_model.ipynb`
**Test Accuracy: 72.89%**

### Objective

Evaluate the isolated effect of data augmentation on generalization. The CNN architecture is kept **completely identical** to the baseline — only the training data pipeline changes. This follows a controlled experimental design.

### Augmentation Configuration

| Parameter          | Value         | Applied To   |
|--------------------|---------------|--------------|
| rescale            | 1./255        | Train + Test |
| rotation_range     | 15 degrees    | Train only   |
| width_shift_range  | 0.1 (10%)     | Train only   |
| height_shift_range | 0.1 (10%)     | Train only   |
| horizontal_flip    | True          | Train only   |
| zoom_range         | Not used      | —            |
| vertical_flip      | Not used      | —            |

Test data had only rescaling applied — no geometric augmentation — ensuring evaluation on clean, unmodified images.

### Model Architecture

Identical to Experiment 1:
- Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → Flatten → Dense(128) → Dense(10, softmax)
- No BatchNorm, no Dropout

### Training Configuration

| Parameter   | Value |
|-------------|-------|
| Epochs      | 10    |
| Batch size  | 32    |
| Callbacks   | None  |

### Per-Epoch Training Log

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | Time/epoch |
|-------|-----------|------------|---------|----------|------------|
| 1     | 0.4269    | 1.5662     | 0.5506  | 1.2434   | 72s        |
| 2     | 0.5558    | 1.2466     | 0.6181  | 1.0641   | 71s        |
| 3     | 0.6063    | 1.1139     | 0.6451  | 1.0062   | 72s        |
| 4     | 0.6325    | 1.0301     | 0.6791  | 0.9362   | 72s        |
| 5     | 0.6596    | 0.9696     | 0.6995  | 0.8723   | 68s        |
| 6     | 0.6731    | 0.9259     | 0.6959  | 0.8926   | 67s        |
| 7     | 0.6849    | 0.8956     | 0.7255  | 0.7958   | 68s        |
| 8     | 0.6932    | 0.8671     | 0.7210  | 0.8061   | 68s        |
| 9     | 0.7036    | 0.8445     | 0.7364  | 0.7551   | 67s        |
| 10    | 0.7095    | 0.8222     | 0.7289  | 0.7917   | 70s        |

Peak validation accuracy: **73.64%** at epoch 9.

### Final Results

| Metric              | Value  |
|---------------------|--------|
| Train Accuracy      | 70.95% |
| Train Loss          | 0.8222 |
| Test Accuracy       | 72.89% |
| Test Loss           | 0.7917 |
| Peak Val Accuracy   | 73.64% (epoch 9) |
| Train/Val Gap       | ~2–3%  |

### Comparison with Baseline

| Metric        | Baseline | + Augmentation | Change    |
|---------------|----------|----------------|-----------|
| Train Acc     | 84.12%   | 70.95%         | −13.17%   |
| Test Acc      | 71.21%   | 72.89%         | **+1.68%**|
| Test Loss     | 0.9473   | 0.7917         | −0.1556   |
| Train/Val Gap | ~13%     | ~2–3%          | **−10%**  |

### Analysis

**Why training accuracy dropped:** Augmentation forces the model to see harder, distorted versions of images during training. It can no longer easily memorize examples, so training accuracy is naturally lower. This is expected and desirable behavior.

**Why validation accuracy improved:** The model was forced to learn more generalizable features instead of memorizing pixel-level patterns. With augmentation, validation accuracy slightly exceeded training accuracy in early epochs — a healthy sign indicating that the test data (clean images) was in some respects easier than the augmented training data.

**Overfitting is now controlled:** The train/val gap collapsed from ~13% to ~2–3%, representing a dramatic improvement in generalization. Validation loss no longer rises — instead it decreases steadily throughout training, showing no signs of the overfitting that plagued the baseline.

**Conclusion:** Augmentation is highly effective at reducing overfitting. The next bottleneck is now model stability and feature learning capacity. The prescribed next steps are Batch Normalization (for training stability) and Dropout (for further regularization).

---

## Experiment 3 — Batch Normalization + Dropout

**File:** `experiment/3-bn_dropout_model.ipynb`
**Test Accuracy: 74.51%**

### Objective

Add Batch Normalization and Dropout to the existing augmented model. This experiment addresses training stability and provides an additional regularization layer. The architecture pattern changes from `Conv → ReLU → Pool` to `Conv → BatchNorm → ReLU → Pool`.

### Data Pipeline

Same augmentation as Experiment 2: rescale + rotation(15°) + width/height shift(10%) + horizontal flip. Test data rescaled only.

### Model Architecture

**Design pattern:** `Conv → BatchNorm → ReLU` (per block), followed by MaxPooling.

| Block       | Layers                                                          |
|-------------|-----------------------------------------------------------------|
| Block 1     | Conv2D(32, 3×3, same) → BN → ReLU → MaxPool(2×2)              |
| Block 2     | Conv2D(64, 3×3, same) → BN → ReLU → MaxPool(2×2)              |
| Block 3     | Conv2D(128, 3×3, same) → BN → ReLU → MaxPool(2×2)             |
| Head        | Flatten → Dense(128) → BN → ReLU → Dropout(0.5) → Dense(10, softmax) |

Key changes from previous experiments:
- `padding='same'` added to all Conv2D layers (preserves spatial dimensions before pooling)
- BatchNormalization inserted after each Conv2D and after the Dense(128) layer
- Activation layer used explicitly as a separate layer (not inline in Conv2D)
- Dropout(0.5) added before the final output Dense layer

### Training Configuration

| Parameter   | Value |
|-------------|-------|
| Epochs      | 10    |
| Batch size  | 32    |
| Callbacks   | None  |

### Per-Epoch Training Log

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | Time/epoch |
|-------|-----------|------------|---------|----------|------------|
| 1     | 0.4712    | 1.4814     | 0.5024  | 1.3645   | 74s        |
| 2     | 0.5907    | 1.1642     | 0.4181  | 2.0517   | 53s        |
| 3     | 0.6331    | 1.0481     | 0.5629  | 1.3135   | 59s        |
| 4     | 0.6586    | 0.9719     | 0.5935  | 1.2003   | 57s        |
| 5     | 0.6804    | 0.9228     | 0.6962  | 0.8833   | 57s        |
| 6     | 0.6976    | 0.8744     | 0.7089  | 0.8259   | 55s        |
| 7     | 0.7061    | 0.8486     | 0.7174  | 0.8196   | 53s        |
| 8     | 0.7206    | 0.8129     | 0.6457  | 1.0472   | 53s        |
| 9     | 0.7286    | 0.7884     | 0.7384  | 0.7616   | 53s        |
| 10    | 0.7382    | 0.7578     | 0.7451  | 0.7424   | 53s        |

Notable: Large validation loss spike at epoch 2 (2.0517) — attributed to BatchNormalization's running statistics still stabilizing in early epochs. Documented as normal behavior, not a training error.

### Final Results

| Metric              | Value  |
|---------------------|--------|
| Train Accuracy      | 73.82% |
| Train Loss          | 0.7578 |
| Test Accuracy       | 74.51% |
| Test Loss           | 0.7424 |
| Train/Val Gap       | ~0–1%  |

### Cumulative Comparison

| Model              | Test Acc | Overfitting | Stability    |
|--------------------|----------|-------------|--------------|
| Baseline           | 71.21%   | Moderate    | Good         |
| + Augmentation     | 72.89%   | Low         | Good         |
| + BN + Dropout     | 74.51%   | Very Low    | Very Stable  |

### Analysis

**Accuracy gain is modest (+1.62%):** The model's regularization is now excellent (near-zero train/val gap), but the architecture itself lacks depth. The model is well-regularized but capacity-constrained.

**Strong spike at epoch 2:** The large jump in validation loss (1.36 → 2.05 → 1.31) in epochs 1–3 is expected when Batch Normalization is first introduced. The running mean and variance statistics are unstable in early epochs and stabilize by epoch 4–5.

**Slight underfitting observed:** With Dropout(0.5) and BatchNorm both active, training accuracy (~73%) barely exceeds validation accuracy (~74.5%), suggesting the model may be slightly over-regularized for its current depth.

**Identified bottleneck:** Regularization is no longer the primary constraint. The limiting factor is now model representational capacity — the architecture is too shallow to learn more complex feature hierarchies.

**Conclusion:** BN and Dropout have done their job. The next step must be increasing model depth.

---

## Experiment 4 — Deeper CNN Architecture

**File:** `experiment/4-deeper_cnn_model.ipynb`
**Test Accuracy: 79.56%**

### Objective

Address the model capacity bottleneck identified in Experiment 3 by increasing architecture depth. Each convolutional block is upgraded from a single Conv2D to two sequential Conv2D layers ("double convolution" pattern). The Dense head is also expanded from 128 to 256 units.

### Data Pipeline

Same augmentation as Experiments 2 and 3.

### Model Architecture

**Key change:** Each block goes from `Conv → BN → ReLU → Pool` to `Conv → BN → ReLU → Conv → BN → ReLU → Pool`.

| Block       | Layers                                                                                 |
|-------------|----------------------------------------------------------------------------------------|
| Block 1     | Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool(2×2)                      |
| Block 2     | Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool(2×2)                      |
| Block 3     | Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool(2×2)                    |
| Head        | Flatten → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)                |

Specific changes from Experiment 3:
- Each block now has **two** Conv2D layers instead of one (doubled convolution depth per block)
- Dense head increased from **128 to 256** units
- BatchNorm and Dropout(0.5) retained
- All Conv2D layers use `padding='same'`
- Filter progression per block: 32→32, 64→64, 128→128

### Training Configuration

| Parameter   | Value |
|-------------|-------|
| Epochs      | 10    |
| Batch size  | 32    |
| Callbacks   | None  |

Note: CPU training is noticeably slower — approximately 110–143 seconds per epoch vs ~53–72 seconds for shallower models, due to the doubled number of Conv layers.

### Per-Epoch Training Log

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | Time/epoch |
|-------|-----------|------------|---------|----------|------------|
| 1     | 0.4965    | 1.4314     | 0.5833  | 1.2360   | 110s       |
| 2     | 0.6491    | 1.0012     | 0.7128  | 0.8336   | 106s       |
| 3     | 0.7017    | 0.8594     | 0.7443  | 0.7395   | 103s       |
| 4     | 0.7364    | 0.7651     | 0.7478  | 0.7608   | 112s       |
| 5     | 0.7575    | 0.7035     | 0.7575  | 0.7278   | 114s       |
| 6     | 0.7731    | 0.6582     | 0.7380  | 0.8117   | 115s       |
| 7     | 0.7935    | 0.6092     | 0.7974  | 0.6002   | 111s       |
| 8     | 0.8014    | 0.5818     | 0.7743  | 0.6681   | 111s       |
| 9     | 0.8130    | 0.5540     | 0.8038  | 0.5758   | 131s       |
| 10    | 0.8169    | 0.5326     | 0.7956  | 0.6270   | 143s       |

### Final Results

| Metric              | Value  |
|---------------------|--------|
| Train Accuracy      | 81.69% |
| Train Loss          | 0.5326 |
| Test Accuracy       | 79.56% |
| Test Loss           | 0.6270 |
| Train/Val Gap       | ~2%    |

### Cumulative Progression

| Experiment            | Test Acc | Delta  |
|-----------------------|----------|--------|
| Baseline              | 71.21%   | —      |
| + Augmentation        | 72.89%   | +1.68% |
| + BN + Dropout        | 74.51%   | +1.62% |
| Deeper CNN (2 conv/block) | 79.56% | **+5.05%** |

### Analysis

**Largest single accuracy gain so far:** The +5.05% jump from 74.51% to 79.56% confirms that model capacity (depth) was the primary bottleneck after regularization was addressed. Doubling the convolution layers per block dramatically increased the model's ability to learn hierarchical features.

**Why double convolution helps:** Two stacked 3×3 convolutions have a receptive field equivalent to a single 5×5 convolution, but with fewer parameters and an additional non-linearity. This allows the model to capture more complex spatial patterns before downsampling.

**Validation oscillation is normal:** Small oscillations in validation loss (e.g., epoch 6 val_loss rises to 0.8117 before dropping again) are attributed to the combination of data augmentation (stochastic), batch normalization, and small dataset size.

**Convergence not complete:** Training accuracy at 81.69% and validation at ~79.56% suggests the model has not fully converged in 10 epochs — there is remaining room for improvement through longer training.

**Proposed next improvements:**
1. Train longer (20–30 epochs) with EarlyStopping — primary recommendation
2. Replace Flatten with GlobalAveragePooling2D
3. Add Learning Rate Scheduler
4. Evaluate per-class accuracy via Confusion Matrix

**Conclusion:** Depth matters. With the regularization framework established in Experiments 2 and 3, increasing architecture depth produced the largest accuracy jump observed up to this point. The model is now undertrained rather than overfitted.

---

## Experiment 5 — Deeper CNN with Training Callbacks

**File:** `experiment/5-deeper_cnn_with_callbacks.ipynb`
**Test Accuracy: 87.28%**

### Objective

Address the convergence limitation identified in Experiment 4 by training longer (30 epochs) and introducing smart training callbacks: EarlyStopping and ReduceLROnPlateau. The model architecture is kept **identical** to Experiment 4 — only the training procedure changes.

### Data Pipeline

Same augmentation as Experiments 2–4.

### Model Architecture

Identical to Experiment 4:
- 3 blocks × (Conv2D → BN → ReLU → Conv2D → BN → ReLU → MaxPool)
- Filters: 32, 64, 128
- Head: Flatten → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)

### Callbacks (New in this experiment)

**EarlyStopping:**

| Parameter             | Value    |
|-----------------------|----------|
| monitor               | val_loss |
| patience              | 5 epochs |
| restore_best_weights  | True     |

Stops training if validation loss does not improve for 5 consecutive epochs, and restores the model weights from the best-performing epoch.

**ReduceLROnPlateau:**

| Parameter  | Value    |
|------------|----------|
| monitor    | val_loss |
| factor     | 0.5      |
| patience   | 3 epochs |
| min_lr     | 1e-6     |

Halves the learning rate when validation loss stagnates for 3 consecutive epochs, allowing finer gradient steps during later training.

### Training Configuration

| Parameter     | Value |
|---------------|-------|
| Max epochs    | 30    |
| Epochs run    | 30 (EarlyStopping did not trigger — val_loss still improving) |
| Batch size    | 32    |
| Initial LR    | 0.001 |

### Per-Epoch Training Log (all 30 epochs)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss  | LR     |
|-------|-----------|------------|---------|-----------|--------|
| 1     | 0.4910    | 1.4471     | 0.5712  | 1.2716    | 0.0010 |
| 2     | 0.6460    | 1.0052     | 0.7112  | 0.8345    | 0.0010 |
| 3     | 0.7009    | 0.8585     | 0.6477  | 1.1685    | 0.0010 |
| 4     | 0.7345    | 0.7681     | 0.6820  | 0.9465    | 0.0010 |
| 5     | 0.7595    | 0.7016     | 0.7332  | 0.8147    | 0.0010 |
| 6     | 0.7744    | 0.6550     | 0.7374  | 0.7866    | 0.0010 |
| 7     | 0.7910    | 0.6085     | 0.7609  | 0.7184    | 0.0010 |
| 8     | 0.8038    | 0.5762     | 0.7499  | 0.8278    | 0.0010 |
| 9     | 0.8133    | 0.5483     | 0.8213  | 0.5247    | 0.0010 |
| 10    | 0.8204    | 0.5280     | 0.8090  | 0.5532    | 0.0010 |
| 11    | 0.8277    | 0.5062     | 0.7836  | 0.6763    | 0.0010 |
| 12    | 0.8339    | 0.4848     | 0.8247  | 0.5098    | 0.0010 |
| 13    | 0.8397    | 0.4717     | 0.7871  | 0.6515    | 0.0010 |
| 14    | 0.8442    | 0.4585     | 0.8266  | 0.5222    | 0.0010 |
| 15    | 0.8493    | 0.4410     | 0.8331  | 0.5064    | 0.0010 |
| 16    | 0.8523    | 0.4298     | 0.8205  | 0.5810    | 0.0010 |
| 17    | 0.8593    | 0.4124     | 0.8425  | 0.4705    | 0.0010 |
| 18    | 0.8622    | 0.4069     | 0.8448  | 0.4607    | 0.0010 |
| 19    | 0.8640    | 0.4017     | 0.8299  | 0.5140    | 0.0010 |
| 20    | 0.8657    | 0.3907     | 0.8540  | 0.4408    | 0.0010 |
| 21    | 0.8698    | 0.3806     | 0.8514  | 0.4680    | 0.0010 |
| 22    | 0.8732    | 0.3752     | 0.8499  | 0.4698    | 0.0010 |
| 23    | 0.8765    | 0.3624     | 0.8574  | 0.4336    | 0.0010 |
| 24    | 0.8762    | 0.3568     | 0.8510  | 0.4454    | 0.0010 |
| 25    | 0.8776    | 0.3587     | 0.8650  | 0.4079    | 0.0010 |
| 26    | 0.8816    | 0.3476     | 0.8725  | 0.3991    | 0.0010 |
| 27    | 0.8823    | 0.3407     | 0.8728  | **0.3943**| 0.0010 |
| 28    | 0.8851    | 0.3324     | 0.8585  | 0.4464    | 0.0010 |
| 29    | 0.8876    | 0.3247     | 0.8694  | 0.4026    | 0.0010 |
| 30    | 0.8890    | 0.3262     | 0.8593  | 0.4438    | 0.0010 |

**Best epoch:** Epoch 27 — val_loss = 0.3943, val_acc = 87.28%.
ReduceLROnPlateau: LR remained at 0.001 throughout all 30 epochs — the callback never triggered because val_loss was still improving across the entire training window.
EarlyStopping: Also did not trigger. Training ended at max epochs.
`restore_best_weights=True` returned the model to epoch 27 weights at the end.

### Final Results

| Metric              | Value  |
|---------------------|--------|
| Train Accuracy      | 88.90% |
| Train Loss          | 0.3262 |
| Test Accuracy       | 87.28% |
| Test Loss           | 0.3943 |
| Best Val Acc Epoch  | 27     |
| Train/Val Gap       | ~1.6%  |

### Per-Class Accuracy (from Confusion Matrix)

| Class      | Accuracy |
|------------|----------|
| airplane   | 83.60%   |
| automobile | 94.40%   |
| bird       | 76.60%   |
| cat        | 76.60%   |
| deer       | 88.20%   |
| dog        | 77.40%   |
| frog       | 95.70%   |
| horse      | 92.30%   |
| ship       | 93.10%   |
| truck      | 94.90%   |

**Strong classes (≥ 90%):** Frog (95.7%), Truck (94.9%), Automobile (94.4%), Ship (93.1%), Horse (92.3%) — classes with clear global structure, distinct backgrounds, and minimal inter-class confusion.

**Weak classes (~76–77%):** Bird (76.6%), Cat (76.6%), Dog (77.4%) — fine-grained animal categories with overlapping textures, similar poses, and frequently confused with one another (cat↔dog, bird↔deer).

**Moderate class:** Airplane (83.6%) — confusions with ship (sky/sea backgrounds) and bird.

### Analysis

**Massive accuracy jump of +7.72%:** Extended training from 10 to 30 epochs with `restore_best_weights=True` produced the largest single gain in this series using the same architecture from Experiment 4. This confirmed that the deeper CNN was strongly undertrained at 10 epochs.

**Key insight from notebook:** "The model was not underpowered before. It was undertrained. Optimization and training control unlocked performance." The architecture had the necessary capacity — it simply needed more gradient updates to converge.

**Why EarlyStopping + best weights mattered:** Although EarlyStopping did not trigger (val_loss was still improving at epoch 30), `restore_best_weights=True` ensured the final model used weights from epoch 27 (the actual best point), not epoch 30 (which had slightly degraded val_loss of 0.4438). Without this, reported test accuracy would likely be lower.

**Accuracy curve behavior:** Training accuracy rose smoothly from ~49% to ~89%. Validation accuracy climbed with larger oscillations (from ~57% to ~87%), which is expected behavior when training with data augmentation and BatchNorm. The overall trend was consistently upward.

**Loss curve behavior:** Training loss decreased monotonically from 1.45 to 0.33. Validation loss decreased from 1.27 to a minimum of 0.39 at epoch 27. No evidence of overfitting or divergence throughout training.

**Conclusion:** Training callbacks are a critical tool. Extended training with intelligent stopping and LR scheduling unlocked the full potential of the Experiment 4 architecture. Per-class analysis reveals that fine-grained animal discrimination (particularly cat, dog, bird) remains the primary bottleneck.

---

## Experiment 6 — GlobalAveragePooling (GAP) Replacement

**File:** `experiment/6-deeper_cnn_gap_model.ipynb`
**Test Accuracy: 88.60%**

### Objective

Replace the `Flatten → Dense(256)` classifier head with `GlobalAveragePooling2D → Dense(128)`. This reduces parameter count, improves spatial robustness, and acts as a stronger regularizer by forcing each feature map to develop semantically meaningful global representations rather than allowing the dense classifier to learn a potentially overfit positional mapping.

### Data Pipeline

Same augmentation as Experiments 2–5.

### Model Architecture

The convolutional backbone is retained from Experiments 4 and 5. Only the classifier head changes.

| Block          | Layers                                                                                    |
|----------------|-------------------------------------------------------------------------------------------|
| Block 1        | Conv2D(32, 3×3, same) → BN → ReLU → Conv2D(32, 3×3, same) → BN → ReLU → MaxPool(2×2)   |
| Block 2        | Conv2D(64, 3×3, same) → BN → ReLU → Conv2D(64, 3×3, same) → BN → ReLU → MaxPool(2×2)   |
| Block 3        | Conv2D(128, 3×3, same) → BN → ReLU → Conv2D(128, 3×3, same) → BN → ReLU → MaxPool(2×2) |
| **Head (new)** | **GlobalAveragePooling2D → Dense(128) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)** |

**Key architectural change:** `Flatten` (outputs a flat 4×4×128 = 2048-dimensional vector) is replaced with `GlobalAveragePooling2D` (computes the spatial average of each of the 128 feature maps, producing a 128-dimensional vector). The Dense head is also reduced from 256 to 128 units.

### Callbacks

Same as Experiment 5: EarlyStopping (patience=5, restore_best_weights=True) and ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6).

### Training Configuration

| Parameter     | Value                      |
|---------------|----------------------------|
| Max epochs    | 30                         |
| Epochs run    | 30 (no early stopping)     |
| Batch size    | 32                         |
| Initial LR    | 0.001                      |

### Per-Epoch Training Log

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | LR       |
|-------|-----------|------------|---------|----------|----------|
| 1     | 0.4548    | 1.5151     | 0.5509  | 1.2937   | 0.001000 |
| 2     | 0.6204    | 1.0917     | 0.5479  | 1.3604   | 0.001000 |
| 3     | 0.6778    | 0.9320     | 0.6131  | 1.2436   | 0.001000 |
| 4     | 0.7181    | 0.8297     | 0.6402  | 1.0873   | 0.001000 |
| 5     | 0.7426    | 0.7595     | 0.6905  | 0.9735   | 0.001000 |
| 6     | 0.7575    | 0.7103     | 0.7484  | 0.7401   | 0.001000 |
| 7     | 0.7748    | 0.6634     | 0.7355  | 0.7842   | 0.001000 |
| 8     | 0.7888    | 0.6260     | 0.7769  | 0.6635   | 0.001000 |
| 9     | 0.8002    | 0.5944     | 0.8009  | 0.5806   | 0.001000 |
| 10    | 0.8096    | 0.5723     | 0.7530  | 0.7701   | 0.001000 |
| 11    | 0.8128    | 0.5531     | 0.8153  | 0.5275   | 0.001000 |
| 12    | 0.8201    | 0.5358     | 0.7801  | 0.6740   | 0.001000 |
| 13    | 0.8262    | 0.5130     | 0.7654  | 0.7012   | 0.001000 |
| 14    | 0.8316    | 0.4974     | 0.8348  | 0.4867   | 0.001000 |
| 15    | 0.8380    | 0.4795     | 0.8221  | 0.5301   | 0.001000 |
| 16    | 0.8415    | 0.4758     | 0.8241  | 0.5123   | 0.001000 |
| 17    | 0.8461    | 0.4608     | 0.8566  | 0.4291   | 0.001000 |
| 18    | 0.8483    | 0.4497     | 0.8304  | 0.4974   | 0.001000 |
| 19    | 0.8538    | 0.4343     | 0.8171  | 0.5342   | 0.001000 |
| 20    | 0.8571    | 0.4286     | 0.8278  | 0.5055   | 0.001000 |
| 21    | 0.8734    | 0.3762     | 0.8475  | 0.4591   | **0.000500** |
| 22    | 0.8786    | 0.3607     | 0.8690  | 0.3888   | 0.000500 |
| 23    | 0.8803    | 0.3538     | 0.8600  | 0.4243   | 0.000500 |
| 24    | 0.8831    | 0.3449     | 0.8624  | 0.4068   | 0.000500 |
| 25    | 0.8831    | 0.3432     | 0.8656  | 0.4014   | 0.000500 |
| 26    | 0.8949    | 0.3120     | 0.8715  | 0.3919   | **0.000250** |
| 27    | 0.8952    | 0.3032     | 0.8800  | 0.3612   | 0.000250 |
| 28    | 0.8984    | 0.3006     | 0.8762  | 0.3799   | 0.000250 |
| 29    | 0.8996    | 0.2991     | 0.8850  | 0.3487   | 0.000250 |
| 30    | 0.8998    | 0.2931     | 0.8860  | **0.3484** | 0.000250 |

**LR reduction events:**
- Epoch 21: LR reduced 0.001 → 0.0005 (val_loss stagnated for 3 epochs: epochs 17–20)
- Epoch 26: LR reduced 0.0005 → 0.00025 (val_loss stagnated again for epochs 22–25)

**This is the first experiment where ReduceLROnPlateau actively triggered.** The two LR reductions visibly pushed validation accuracy from ~85.6% to ~88.6% in the final training phase.

### Final Results

| Metric              | Value  |
|---------------------|--------|
| Train Accuracy      | 89.98% |
| Train Loss          | 0.2931 |
| Test Accuracy       | 88.60% |
| Test Loss           | 0.3484 |
| Train/Val Gap       | ~1.4%  |

### Per-Class Accuracy (from Confusion Matrix)

| Class      | Accuracy | Change from Exp 5 |
|------------|----------|-------------------|
| airplane   | 89.00%   | +5.4%             |
| automobile | 96.80%   | +2.4%             |
| bird       | 85.00%   | +8.4%             |
| cat        | 81.10%   | +4.5%             |
| deer       | 89.30%   | +1.1%             |
| dog        | 70.20%   | **−7.2%**         |
| frog       | 94.00%   | −1.7%             |
| horse      | 91.40%   | −0.9%             |
| ship       | 94.90%   | +1.8%             |
| truck      | 94.30%   | −0.6%             |

### Analysis

**GAP improved most classes significantly:** Bird gained +8.4%, airplane +5.4%, cat +4.5%, automobile +2.4%. By forcing each feature map to represent globally meaningful patterns, GAP improved the model's ability to recognize classes with strong global structure.

**Dog class worsened (−7.2%):** Global averaging reduced the classifier's flexibility for fine-grained local texture discrimination. Dog ↔ cat confusions increased substantially (dog predicted as cat 162 times) because subtle local texture differences between dogs and cats require position-sensitive features that GAP averages away. This is a known and documented trade-off.

**Why GAP generally helps over Flatten:**
- Flatten produces a 2048-dimensional vector where the classifier can learn to exploit spatial position artifacts
- GAP produces a 128-dimensional vector where each element is the global average activation of one feature channel
- This forces the convolutional layers to encode class-relevant information across the entire spatial extent of each feature map
- Fewer parameters in the classifier head reduces overfitting risk in the head

**LR scheduling visibly helped:** The two LR reductions (at epochs 21 and 26) enabled finer gradient updates, driving validation accuracy from ~85.6% to ~88.6% in the final training phase.

**Conclusion:** GlobalAveragePooling is an effective architectural upgrade over Flatten for this task. It improves generalization across most classes by enforcing stronger spatial information compression. The remaining limitation is fine-grained inter-class discrimination, especially for visually similar animal categories.

---

## Experiment 7 — ResNet-Style Model with Residual Connections

**File:** `experiment/fork-of-resnet-style-model.ipynb`
**Test Accuracy: 90.08%**

### Objective

Implement residual (skip) connections to enable training of a deeper and more powerful network. Residual connections allow the gradient to flow directly through skip paths, mitigating the vanishing gradient problem and enabling the network to learn incremental refinements rather than full transformations at each block.

### Hardware

This experiment was run on **Kaggle** with **2× Tesla T4 GPUs** (Compute Capability 7.5), XLA JIT compilation enabled, and cuDNN 91002. All previous experiments were CPU-only. GPU acceleration made this experiment viable to iterate on quickly.

### Data Pipeline

Same augmentation as all previous experiments: rescale + rotation(15°) + width/height shift(10%) + horizontal flip. Test generator uses `shuffle=False` to ensure correct alignment with ground-truth labels.

### Residual Block Definition

```python
def residual_block(x, filters, downsample=False):
    shortcut = x
    stride = 2 if downsample else 1

    x = Conv2D(filters, (3,3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x
```

**Design decisions:**
- Shortcut path uses a 1×1 Conv + BN projection when either (a) downsampling with stride=2 or (b) the number of channels changes
- ReLU is applied *after* the residual addition, not on the shortcut path
- Downsampling is done via stride=2 in the first 3×3 Conv, not via a separate MaxPool layer
- The Functional API (`models.Model(inputs, outputs)`) is used instead of Sequential (required for branching skip connections)

### Full Model Architecture

**Stem:**

| Layer           | Config                        |
|-----------------|-------------------------------|
| Conv2D          | 32 filters, 3×3, same padding |
| BatchNorm       | —                             |
| Activation      | ReLU                          |

**Stage 1 — filters=32, no downsampling (2 residual blocks):**
- `residual_block(x, 32)` — identity shortcut
- `residual_block(x, 32)` — identity shortcut
- Output spatial size: 32×32

**Stage 2 — filters=64, first block downsamples (2 residual blocks):**
- `residual_block(x, 64, downsample=True)` — stride=2, shortcut projected via 1×1 Conv + BN
- `residual_block(x, 64)` — identity shortcut
- Output spatial size: 16×16

**Stage 3 — filters=128, first block downsamples (2 residual blocks):**
- `residual_block(x, 128, downsample=True)` — stride=2, shortcut projected via 1×1 Conv + BN
- `residual_block(x, 128)` — identity shortcut
- Output spatial size: 8×8

**Classifier Head:**

| Layer                    | Config                    |
|--------------------------|---------------------------|
| GlobalAveragePooling2D   | Reduces 8×8×128 → 128     |
| Dense                    | 128 units                 |
| BatchNormalization       | —                         |
| Activation               | ReLU                      |
| Dropout                  | 0.5                       |
| Dense (output)           | 10 units, Softmax         |

**Total design:** 6 residual blocks in 3 stages with progressively increasing channels (32 → 64 → 128) and decreasing spatial resolution (32×32 → 16×16 → 8×8), followed by GAP and a fully-connected classifier.

### Compilation

Adam optimizer (lr=0.001), categorical crossentropy loss, accuracy metric.

### Callbacks

Same as Experiments 5 and 6: EarlyStopping (patience=5, restore_best_weights=True) and ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6).

### Training Configuration

| Parameter     | Value                           |
|---------------|---------------------------------|
| Max epochs    | 30                              |
| Epochs run    | 30 (EarlyStopping did not trigger) |
| Batch size    | 32                              |
| Initial LR    | 0.001                           |
| Hardware      | 2× Tesla T4 GPU (Kaggle)        |

### Per-Epoch Training Log

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss | LR       |
|-------|-----------|------------|---------|----------|----------|
| 1     | 0.3153    | 1.9172     | 0.4137  | 1.8554   | 0.001000 |
| 2     | 0.5725    | 1.1993     | 0.5849  | 1.2310   | 0.001000 |
| 3     | 0.6484    | 1.0053     | 0.5731  | 1.3340   | 0.001000 |
| 4     | 0.6998    | 0.8740     | 0.6866  | 0.9727   | 0.001000 |
| 5     | 0.7358    | 0.7773     | 0.6755  | 1.0603   | 0.001000 |
| 6     | 0.7543    | 0.7168     | 0.7238  | 0.8700   | 0.001000 |
| 7     | 0.7826    | 0.6471     | 0.6547  | 1.1554   | 0.001000 |
| 8     | 0.7928    | 0.6124     | 0.6723  | 1.0198   | 0.001000 |
| 9     | 0.8037    | 0.5803     | 0.8064  | 0.5707   | 0.001000 |
| 10    | 0.8192    | 0.5372     | 0.8142  | 0.5555   | 0.001000 |
| 11    | 0.8257    | 0.5161     | 0.8013  | 0.5804   | 0.001000 |
| 12    | 0.8357    | 0.4945     | 0.8326  | 0.5011   | 0.001000 |
| 13    | 0.8431    | 0.4618     | 0.8168  | 0.5855   | 0.001000 |
| 14    | 0.8471    | 0.4557     | 0.8201  | 0.5522   | 0.001000 |
| 15    | 0.8578    | 0.4271     | 0.8487  | 0.4451   | 0.001000 |
| 16    | 0.8621    | 0.4147     | 0.8327  | 0.5201   | 0.001000 |
| 17    | 0.8646    | 0.4067     | 0.8198  | 0.5535   | 0.001000 |
| 18    | 0.8709    | 0.3849     | 0.8357  | 0.4770   | 0.001000 |
| 19    | 0.8882    | 0.3395     | 0.8759  | 0.3645   | **0.000500** |
| 20    | 0.8984    | 0.3051     | 0.8660  | 0.3980   | 0.000500 |
| 21    | 0.9009    | 0.2900     | 0.8563  | 0.4346   | 0.000500 |
| 22    | 0.9009    | 0.2942     | 0.8824  | 0.3576   | 0.000500 |
| 23    | 0.9054    | 0.2764     | 0.8658  | 0.4141   | 0.000500 |
| 24    | 0.9063    | 0.2731     | 0.8737  | 0.3987   | 0.000500 |
| 25    | 0.9101    | 0.2652     | 0.8865  | 0.3418   | 0.000500 |
| 26    | 0.9113    | 0.2649     | 0.8905  | 0.3340   | 0.000500 |
| 27    | 0.9149    | 0.2496     | 0.8905  | 0.3401   | 0.000500 |
| 28    | 0.9157    | 0.2485     | 0.8861  | 0.3533   | 0.000500 |
| 29    | 0.9175    | 0.2394     | 0.8810  | 0.3813   | 0.000500 |
| 30    | 0.9263    | 0.2202     | **0.9008** | **0.2988** | **0.000250** |

**LR reduction events:**
- Epoch 19: LR 0.001 → 0.0005 (val_loss stagnated: epochs 15–18)
- Epoch 30: LR 0.0005 → 0.00025 (triggered at the last epoch)

**Early-epoch instability (epochs 3–8):** Validation loss oscillated significantly (epoch 7 val_loss = 1.1554 even with train_acc = 78.26%). This is characteristic of residual networks — skip connections need several epochs before their gradient highway effect becomes fully beneficial.

**EarlyStopping did not trigger:** Validation loss continued to improve at epoch 30. Final model uses epoch 30 weights (val_loss = 0.2988, lowest of all epochs).

### Final Results

| Metric              | Value     |
|---------------------|-----------|
| Train Accuracy      | 92.63%    |
| Train Loss          | 0.2202    |
| Test Accuracy       | **90.08%**|
| Test Loss           | 0.2961    |
| Train/Val Gap       | ~2.5%     |

### Analysis

**Crossed the 90% barrier:** The ResNet-style architecture is the first model in this series to exceed 90% on the CIFAR-10 test set — a meaningful milestone for a custom CNN trained from scratch without pretrained weights.

**Why residual connections help:**
- In a plain deep CNN, gradients must propagate through every layer during backpropagation. As depth increases, gradients can vanish before reaching early layers.
- Skip connections create a direct gradient path from the loss back to early layers, bypassing intermediate transformations.
- This allows the network to function as an ensemble of shallower subnetworks, each learning different levels of abstraction.
- Instead of learning the full transformation H(x), each residual block only needs to learn the residual F(x) = H(x) − x, which is a simpler optimization problem.

**1×1 projection shortcuts:** When a downsampling block changes both spatial size (stride=2) and channel count, the shortcut path is projected via a 1×1 Conv + BN to match dimensions. This is a standard technique from the original ResNet paper (He et al., 2016).

**Early instability is not a bug:** Large val_loss oscillations in epochs 3–8 reflect the skip connections' learning dynamics — they become effective only after the model has learned some initial feature representations at the base learning rate.

---

## Complete Accuracy Progression

```
71.21%  [Exp 1] Baseline CNN
72.89%  [Exp 2] + Data Augmentation             (+1.68%)
74.51%  [Exp 3] + Batch Normalization + Dropout (+1.62%)
79.56%  [Exp 4] + Deeper Architecture (2×Conv)  (+5.05%)
87.28%  [Exp 5] + 30-Epoch Training + Callbacks (+7.72%)
88.60%  [Exp 6] + GlobalAveragePooling          (+1.32%)
90.08%  [Exp 7] + Residual Connections          (+1.48%)
──────────────────────────────────────────────────────────
Total improvement: +18.87 percentage points
```

**The two largest jumps:**
1. **Exp 4 → Exp 5 (+7.72%):** Same architecture, trained 3× longer with callbacks. The deeper CNN was massively undertrained at 10 epochs.
2. **Exp 3 → Exp 4 (+5.05%):** Once regularization was in place, doubling depth per block directly unlocked representational capacity.

---

## All Experiments Summary

| # | File | Architecture | Key Change | Epochs | Callbacks | Test Acc | Test Loss |
|---|------|-------------|------------|--------|-----------|----------|-----------|
| 0 | `0-preprocessing.ipynb` | — | EDA + data pipeline | — | — | — | — |
| 1 | `1-baseline_model.ipynb` | 3-block CNN, Flatten, Dense(128) | Baseline | 10 | None | 71.21% | 0.9473 |
| 2 | `2-augmentation_model.ipynb` | Same as #1 | + Augmentation | 10 | None | 72.89% | 0.7917 |
| 3 | `3-bn_dropout_model.ipynb` | + BN after each Conv, Dropout(0.5) | + BN + Dropout | 10 | None | 74.51% | 0.7424 |
| 4 | `4-deeper_cnn_model.ipynb` | 2×Conv/block, Dense(256) | + Depth | 10 | None | 79.56% | 0.6270 |
| 5 | `5-deeper_cnn_with_callbacks.ipynb` | Same as #4 | + EarlyStopping + ReduceLROnPlateau | 30 | ES + ReduceLR | 87.28% | 0.3943 |
| 6 | `6-deeper_cnn_gap_model.ipynb` | Same backbone, Flatten → GAP, Dense(128) | + GAP | 30 | ES + ReduceLR | 88.60% | 0.3484 |
| 7 | `fork-of-resnet-style-model.ipynb` | ResNet-style, 6 residual blocks, GAP | + Residual Connections | 30 | ES + ReduceLR | **90.08%** | 0.2961 |

---

## Key Lessons from the Experimental Series

### 1. Diagnose Before Fixing
Every experiment begins with analysis of learning curves (accuracy and loss). Without correctly identifying whether the current bottleneck is overfitting, underfitting, or insufficient convergence, changes can be ineffective or counterproductive.

### 2. Controlled Experimentation
Each notebook changes one variable at a time. Experiments 1–3 keep the same architecture while varying only the data pipeline and regularization. Experiments 4–7 progressively layer in architectural improvements. This makes it possible to measure the isolated impact of each change.

### 3. Higher Training Accuracy Does Not Mean a Better Model
In Experiment 2, training accuracy decreased from 84% to 71% after augmentation, but test accuracy improved. Generalization performance (test/validation accuracy) is the only meaningful evaluation metric.

### 4. Regularization Order Matters
- Augmentation should precede BatchNorm/Dropout (Experiments 2 → 3)
- Regularization must be in place before deepening the model (Experiments 3 → 4)
- Adding depth without prior regularization leads to rapid overfitting

### 5. Training Duration Is a Hyperparameter
The biggest single accuracy gain (+7.72%) came purely from training longer with smart callbacks — with no architecture change required. ReduceLROnPlateau helps the optimizer escape flat regions; EarlyStopping with `restore_best_weights` prevents late-epoch degradation.

### 6. GlobalAveragePooling vs Flatten
GAP is a structural regularizer for convolutional networks. It forces spatial features to be globally meaningful, reduces the parameter count in the classifier head, and improves generalization for shape- and structure-dominant classes. The trade-off is reduced flexibility for fine-grained texture discrimination.

### 7. Residual Connections Enable Deeper, More Powerful Networks
Skip connections are not just about adding depth — they fundamentally change how gradients flow and how the model conceptualizes learning. Each block learns a residual correction to the shortcut, which is a simpler optimization problem than learning a complete transformation.

### 8. Fine-Grained Animal Classes Are Consistently Hardest
Across every experiment, cat, dog, and bird achieve the lowest per-class accuracies. These classes share fine-grained textures, overlapping poses, and visually similar backgrounds. Cat/dog confusion is a fundamental challenge in CIFAR-10 that persists even at 90% overall accuracy.

---

## Final Model Specification

| Property             | Value                             |
|----------------------|-----------------------------------|
| Architecture         | ResNet-style CNN (6 residual blocks, 3 stages) |
| Stem                 | Conv2D(32) → BN → ReLU            |
| Stage 1              | 2× ResBlock(32), output 32×32×32  |
| Stage 2              | 2× ResBlock(64, downsample first), output 16×16×64 |
| Stage 3              | 2× ResBlock(128, downsample first), output 8×8×128 |
| Classifier head      | GAP → Dense(128) → BN → ReLU → Dropout(0.5) → Dense(10, softmax) |
| Optimizer            | Adam (lr: 0.001 → 0.0005 → 0.00025 via ReduceLROnPlateau) |
| Augmentation         | rotation(15°), shift(10%), horizontal flip |
| Epochs               | 30                                |
| Batch size           | 32                                |
| Test Accuracy        | **90.08%**                        |
| Test Loss            | 0.2961                            |
| Training Hardware    | 2× Tesla T4 GPU (Kaggle)          |
| Saved formats        | `.keras`, `.h5`, SavedModel directory |

---

## Deployment — FastAPI Web Application

The final ResNet model is deployed as a full-stack web application using **FastAPI**.

### Stack

| Component    | Technology                                  |
|--------------|---------------------------------------------|
| Backend      | FastAPI (Python)                            |
| Model loading| Keras `load_model`                         |
| Inference    | `predictor.py` — preprocesses image, runs `.predict()` |
| Frontend     | HTML template (`index.html`) + CSS (`style.css`) |
| Serving      | Uvicorn (via `run.py`)                      |

### Application Flow

1. User visits `http://localhost:8000`
2. User uploads an image (any format or resolution)
3. Backend resizes image to 32×32, normalizes pixel values to [0, 1]
4. Image passed through the loaded ResNet model
5. Predicted class label returned and displayed in a popup UI on the page

### File Structure

```
cifar_classifier_api/
├── app/
│   ├── main.py          # FastAPI routes and endpoint definitions
│   ├── model_loader.py  # Keras model loading logic
│   ├── predictor.py     # Preprocessing + inference pipeline
│   ├── utils.py         # Image preprocessing utilities
│   └── schemas.py       # Request/response schemas
├── models/
│   └── resnet_cifar10_model.keras  # Saved final model (~8.8 MB)
├── templates/
│   └── index.html       # Frontend HTML
├── static/
│   └── style.css        # Styles
└── run.py               # Launches Uvicorn server
```
