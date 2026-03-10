import numpy as np
from .model_loader import model

CIFAR_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def predict(image_array):

    preds = model.predict(image_array)

    class_index = np.argmax(preds)

    return CIFAR_CLASSES[class_index]