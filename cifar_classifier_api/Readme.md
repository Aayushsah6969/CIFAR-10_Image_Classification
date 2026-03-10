# CIFAR-10 Image Classification System (ResNet + FastAPI)

A full-stack deep learning application that performs **real-time image classification using a ResNet-based Convolutional Neural Network trained on the CIFAR-10 dataset**.

The system allows users to upload an image through a web interface, automatically sends it to a backend API, performs preprocessing, runs the trained neural network model, and returns the predicted class.

This project demonstrates **machine learning model deployment**, integrating a trained TensorFlow model with a backend API and a browser interface.

---

# Project Overview

The goal of this project is to build a **complete ML inference pipeline**, moving beyond notebook experimentation to a deployable system.

Pipeline:

```
User selects image
      ↓
Browser preview + automatic upload
      ↓
FastAPI backend receives image
      ↓
Image preprocessing
      ↓
ResNet CNN model inference
      ↓
Prediction returned
      ↓
Popup UI displays result
```

---

# Features

* ResNet-based CNN trained on CIFAR-10
* FastAPI backend for model inference
* Automatic image upload on selection
* Image preview before prediction
* Popup UI displaying prediction result
* Clean modular backend structure
* Production-style ML inference pipeline

---

# CIFAR-10 Classes

The model predicts one of the following classes:

```
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

---

# Project Structure

```
cifar_classifier_api
│
├── app
│   ├── main.py
│   ├── model_loader.py
│   ├── predictor.py
│   └── utils.py
│
├── models
│   └── resnet_cifar10_model.keras
│
├── static
│   └── style.css
│
├── templates
│   └── index.html
│
├── requirements.txt
└── run.py
```

---

# Technology Stack

Backend:

* FastAPI
* TensorFlow / Keras
* Python

Frontend:

* HTML
* CSS
* JavaScript

ML Model:

* ResNet-style Convolutional Neural Network
* CIFAR-10 dataset

---

# System Architecture

```
Browser (HTML + JS)
        │
        ▼
FastAPI Web Server
        │
        ▼
Image Preprocessing
        │
        ▼
TensorFlow Model
        │
        ▼
Prediction Response
```

---

# Installation Guide

## 1. Clone the Repository

```
git clone https://github.com/yourusername/cifar-classifier.git

cd cifar-classifier
```

---

## 2. Create Virtual Environment

Linux / Mac

```
python -m venv venv
source venv/bin/activate
```

Windows

```
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```

---

# Required Python Packages

```
fastapi
uvicorn
tensorflow
pillow
jinja2
python-multipart
```

---

# Running the Application

Start the server using:

```
python run.py
```

or directly with uvicorn:

```
uvicorn app.main:app --reload
```

---

# Access the Web Application

Open your browser and go to:

```
http://localhost:8000
```

---

# How to Use

1. Open the web application.
2. Click **Choose Image**.
3. Select an image.
4. Image preview appears automatically.
5. The image is uploaded to the backend.
6. The model predicts the class.
7. A popup displays the predicted category.

---

# Image Preprocessing

Before inference, images go through the following steps:

```
1. Resize to 32x32 pixels
2. Convert to RGB
3. Normalize pixel values (divide by 255)
4. Expand dimensions for batch input
```

These steps ensure the input format matches the training pipeline.

---

# Model Details

Architecture:

* ResNet-style Convolutional Neural Network

Framework:

* TensorFlow / Keras

Dataset:

* CIFAR-10

Input Shape:

```
32 × 32 × 3
```

Output:

```
10 class probabilities
```

Prediction uses the class with the highest probability.

---

# API Endpoint

## POST /predict

Accepts an image file and returns the predicted class.

Example response:

```
{
  "prediction": "cat"
}
```

---

# Example Workflow

```
Upload Image → Model Prediction
```

Example:

```
Input Image: Dog
Prediction: dog
```

---

# Future Improvements

Possible extensions for this project:

* Top-3 prediction probabilities
* Confidence scores
* Drag-and-drop image upload
* Docker containerization
* GPU inference support
* Cloud deployment

---

# Educational Value

This project demonstrates:

* Deep learning model training
* Model serialization
* Backend API development
* ML model deployment
* Full-stack ML system integration

---

