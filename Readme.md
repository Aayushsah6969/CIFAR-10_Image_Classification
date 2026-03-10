# CIFAR-10 Image Classification using CNN

A progressive deep learning project that implements and compares multiple CNN architectures for image classification on the CIFAR-10 dataset, culminating in a ResNet-style model achieving **90.08% test accuracy**, deployed as a full-stack web application using FastAPI.

---

## Project Overview

The project is structured as a series of controlled experiments, each building on the previous to improve accuracy and generalization:

```
Image → Convolution → Activation (ReLU) → Pooling → Feature Extraction → Fully Connected Layers → Softmax Classification
```

Experiments conducted:

- **Experiment 1:** Baseline CNN (3 conv blocks, no regularization)
- **Experiment 2:** + Data Augmentation
- **Experiment 3:** + Batch Normalization + Dropout
- **Experiment 4:** Deeper CNN (double conv per block, larger Dense)
- **Experiment 5:** + EarlyStopping + ReduceLROnPlateau callbacks (30 epochs)
- **Experiment 6:** Flatten replaced with GlobalAveragePooling2D
- **Experiment 7 (Final):** ResNet-style model with residual connections

The final model is deployed as a web application — users can upload any image via a browser interface and receive a real-time prediction.

---

## Dataset

**CIFAR-10** — stored as PNG images organized in class folders.

- 60,000 color images at 32×32 resolution
- 50,000 training / 10,000 test images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Loaded using `ImageDataGenerator` with `rescale=1./255`

---

## Experiments

### Experiments Summary Table

| # | Experiment | Key Change | BN | Dropout | Augmentation | Callbacks | Epochs | Test Accuracy |
|---|------------|------------|----|---------|--------------|-----------|--------|---------------|
| 1 | Baseline CNN | 3 conv blocks, no regularization | No | No | No | None | 10 | 71.21% |
| 2 | + Augmentation | rotation, shift, h-flip | No | No | Yes | None | 10 | 72.89% |
| 3 | + BN + Dropout | BatchNorm + Dropout(0.5) | Yes | 0.5 | Yes | None | 10 | 74.51% |
| 4 | Deeper CNN | 2 conv layers per block, Dense(256) | Yes | 0.5 | Yes | None | 10 | 79.56% |
| 5 | + Callbacks | EarlyStopping + ReduceLROnPlateau | Yes | 0.5 | Yes | ES + ReduceLR | 30 | 87.28% |
| 6 | GAP Model | GlobalAveragePooling2D replaces Flatten | Yes | 0.5 | Yes | ES + ReduceLR | 30 | 88.60% |
| 7 | ResNet-style | Residual (skip) connections | Yes | 0.5 | Yes | ES + ReduceLR | 30 | **90.08%** |

**Total improvement: +18.87 percentage points (71.21% → 90.08%)**

---

### Experiment Details

**Exp 1 — Baseline CNN**
- Architecture: Conv2D(32) → Pool → Conv2D(64) → Pool → Conv2D(128) → Flatten → Dense(128) → Softmax
- Result: 71.21% | Train/val gap ~13% — clear overfitting

**Exp 2 — Data Augmentation**
- Added: rotation 15°, width/height shift 0.1, horizontal flip
- Result: 72.89% | Gap reduced to ~2–3%

**Exp 3 — BN + Dropout**
- Added: BatchNormalization after each conv, Dropout(0.5) before Dense
- Result: 74.51% | Very stable training, minimal overfitting; model capacity now the bottleneck

**Exp 4 — Deeper CNN**
- Changed: 2 conv layers per block (VGG-style), Dense widened to 256
- Result: 79.56% | Largest single jump (+5%), confirms capacity was limiting factor

**Exp 5 — Callbacks**
- Added: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=3), trained for 30 epochs
- Result: 87.28% | +7.7% — the model at 10 epochs was severely undertrained

**Exp 6 — GlobalAveragePooling2D**
- Changed: Flatten → GlobalAveragePooling2D, Dense reduced to 128
- Result: 88.60% | GAP forces spatial feature maps to encode global patterns, improving generalization; LR was automatically reduced (0.001 → 0.0005 → 0.00025)

**Exp 7 — ResNet-style (Final Model)**
- Added: True residual (skip) connections with identity and 1×1 projection shortcuts; downsampling via stride-2 convolutions; trained on Kaggle GPU (Tesla T4)
- Result: **90.08%** | Residual connections improve gradient flow; crossed the 90% milestone

---

## Final Model Architecture (ResNet-style, 90.08%)

Built using the Keras Functional API.

```
Input(32×32×3)
└── Conv2D(32, 3×3, same) → BN → ReLU

Residual Block 1  Conv(32) → BN → ReLU → Conv(32) → BN  ─┐ shortcut
                                                           + → ReLU
Residual Block 2  Conv(32) → BN → ReLU → Conv(32) → BN  ─┐ shortcut
                                                           + → ReLU

Residual Block 3  Conv(64, stride=2) → BN → ReLU → Conv(64) → BN  ─┐ 1×1 projection
                                                                     + → ReLU
Residual Block 4  Conv(64) → BN → ReLU → Conv(64) → BN  ─┐ shortcut
                                                           + → ReLU

Residual Block 5  Conv(128, stride=2) → BN → ReLU → Conv(128) → BN  ─┐ 1×1 projection
                                                                       + → ReLU
Residual Block 6  Conv(128) → BN → ReLU → Conv(128) → BN  ─┐ 1×1 projection
                                                             + → ReLU

GlobalAveragePooling2D
Dense(128) → BN → ReLU → Dropout(0.5)
Dense(10, Softmax)
```

---

## Training Details (Final Model)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial LR | 0.001 |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=3, min=1e-6) |
| Loss | Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 32 |
| Early Stopping | patience=5, restore best weights |
| Hardware | Kaggle GPU (Tesla T4) |

**Data Augmentation:**
- Rotation range: 15°
- Width/height shift: 0.1
- Horizontal flip: Yes
- Rescaling: 1/255

---

## Per-Class Accuracy (Final Model)

| Class | Accuracy |
|-------|----------|
| Airplane | 89.0% |
| Automobile | 96.8% |
| Bird | 85.0% |
| Cat | 81.1% |
| Deer | 89.3% |
| Dog | 70.2% |
| Frog | 94.0% |
| Horse | 91.4% |
| Ship | 94.9% |
| Truck | 94.3% |
| **Overall** | **90.08%** |

---

## FastAPI Web Application

The final ResNet model is deployed as a full-stack web application.

**Pipeline:**
```
User selects image
      ↓
Browser preview + automatic upload
      ↓
FastAPI backend receives image
      ↓
Preprocessing (resize 32×32, normalize, expand dims)
      ↓
ResNet model inference
      ↓
Prediction returned as JSON
      ↓
Popup UI displays result
```

**API Endpoint:**

`POST /predict` — accepts an image file, returns predicted class.

```json
{ "prediction": "cat" }
```

**Run the server:**

```bash
cd cifar_classifier_api
python run.py
# or
uvicorn app.main:app --reload
```

Access the web app at `http://localhost:8000`

---

## Project Structure

```
CIFAR_Classification/
│
├── experiment/
│   ├── 0-preprocessing.ipynb          # Data pipeline and EDA
│   ├── 1-baseline_model.ipynb         # Exp 1 — 71.21%
│   ├── 2-augmentation_model.ipynb     # Exp 2 — 72.89%
│   ├── 3-bn_dropout_model.ipynb       # Exp 3 — 74.51%
│   ├── 4-deeper_cnn_model.ipynb       # Exp 4 — 79.56%
│   ├── 5-deeper_cnn_with_callbacks.ipynb  # Exp 5 — 87.28%
│   ├── 6-deeper_cnn_gap_model.ipynb   # Exp 6 — 88.60%
│   └── fork-of-resnet-style-model.ipynb   # Exp 7 — 90.08% (Final)
│
├── cifar_classifier_api/              # FastAPI deployment
│   ├── app/
│   │   ├── main.py                    # FastAPI routes
│   │   ├── model_loader.py            # Keras model loading
│   │   ├── predictor.py               # Inference logic
│   │   ├── utils.py                   # Preprocessing utilities
│   │   └── schemas.py                 # Request/response schemas
│   ├── models/
│   │   └── resnet_cifar10_model.keras # Saved final model
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   └── index.html
│   ├── requirements.txt
│   └── run.py
│
├── models/
│   ├── resnet_cifar10_model.h5        # Saved model (HDF5)
│   └── resnet_cifar10_model/          # Saved model (SavedModel format)
│
├── data/
│   └── cifar10/
│       ├── train/                     # 50,000 training images (class subfolders)
│       └── test/                      # 10,000 test images (class subfolders)
│
├── results/
│   ├── results_till_4_experiments.md
│   ├── CIFAR10_CNN_Experiment_Report.docx
│   └── CNN Code Explanation till 4 exp.pdf
│
├── requirements.txt
├── LICENSE
└── Readme.md
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/CIFAR_Classification.git
cd CIFAR_Classification
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run experiments

```bash
jupyter notebook
```

Open any notebook in `experiment/` and run all cells.

### 5. Run the API (optional)

```bash
cd cifar_classifier_api
pip install -r requirements.txt
python run.py
```

---

## Dependencies

**Training:**
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

**API deployment:**
- FastAPI
- Uvicorn
- Pillow
- Jinja2
- python-multipart

---

## Key Takeaways

1. Higher training accuracy does not mean a better model — diagnose using train/val learning curves.
2. Data augmentation is the single most impactful regularization technique at low cost.
3. Batch Normalization stabilizes training; Dropout reduces overfitting.
4. Model depth controls representational power — doubling conv layers per block gave the largest single accuracy jump.
5. Training duration matters — the model at 10 epochs was severely undertrained; 30 epochs with callbacks gave +7.7%.
6. GlobalAveragePooling2D improves generalization over Flatten by reducing the parameter count and spatial overfitting.
7. Residual connections allow gradients to flow through deeper networks, enabling the model to cross the 90% threshold.

---

## Future Improvements

- Wider ResNet variants (ResNet-34, ResNet-50)
- Learning rate warmup + cosine annealing
- MixUp / CutMix data augmentation
- Test-time augmentation (TTA)
- Docker containerization of the API
- Cloud deployment (AWS / GCP / Render)
- Top-3 predictions with confidence scores in the UI

---

## Conclusion

Through 7 controlled experiments, each targeting a specific bottleneck — overfitting, underfitting, training instability, insufficient capacity, or lack of gradient flow — the final ResNet-style model achieved **90.08% test accuracy** on CIFAR-10.

The project covers the full deep learning lifecycle: data preprocessing, model design, regularization, architecture search, callbacks-based training, model serialization, and production deployment via a FastAPI web application.

---

## Author

**Aayush**
Deep Learning & Computer Vision
