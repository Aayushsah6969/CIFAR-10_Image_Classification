import tensorflow as tf

MODEL_PATH = "models/resnet_cifar10_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)