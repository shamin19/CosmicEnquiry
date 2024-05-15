import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Adjust based on your model's input size
    image = np.array(image)
    if image.shape[-1] == 4:  # Convert RGBA to RGB if necessary
        image = image[..., :3]
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def predict_image(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction, axis=1)[0]
    class_labels = {0: 'Comets', 1: 'Galaxies', 2: 'Moons', 3: 'Nebulas', 4: 'Planets', 5: 'Stars', 6: 'Sun'}  # Update based on your model
    return class_labels[class_index]

# Load the trained model
model = load_model()

# Streamlit app interface
st.title("Astronomical Image Classifier")
st.write("Upload an image to classify it into one of the categories.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict_image(image, model)
    st.write(f"Prediction: {label}")
