import numpy as np
from PIL import Image

IMG_SIZE = 32

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image