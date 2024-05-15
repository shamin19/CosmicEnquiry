import numpy as np
from PIL import Image
import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = image.resize((150, 150))  # Resize the image to match the input size of your model
    image = np.array(image)
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = image[..., :3]
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction, axis=1)[0]
    class_labels = {0: 'Comets', 1: 'Galaxies', 2: 'Moons', 3: 'Nebulas', 4: 'Planets', 5: 'Stars', 6: 'Sun'}  # Update this dictionary with your actual class labels
    return class_labels[class_index]
