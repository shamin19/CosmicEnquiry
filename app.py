import streamlit as st
import numpy as np
from PIL import Image
import joblib
from utils import load_model, predict_image

# Load the trained model
model = load_model('model.joblib')

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