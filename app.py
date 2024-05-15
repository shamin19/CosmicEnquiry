import requests
import streamlit as st
import os
import zipfile
from utils import load_model, predict_image
from PIL import Image

# URLs for the dataset and model on Google Drive
dataset_url = 'https://drive.google.com/uc?id=1e23T43mfIKI-_qnZ6IwkU4bpl6tu0N6v'
model_url = 'https://drive.google.com/uc?id=1g_QYE3DVhZPHQKavpqMJ-asaG3lvW6D9'

# Paths where the dataset and model will be saved
dataset_zip_path = 'NASA APOD Dataset.zip'
dataset_dir = 'NASA APOD Dataset'
model_path = 'model.h5'

# Function to download files using requests
def download_file(url, output):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        st.error(f"Error downloading file: {e}")

def extract_zip_file(zip_path, extract_to='.'):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        st.error(f"The file {zip_path} is not a valid zip file.")

# Download and extract the dataset if not already present
if not os.path.exists(dataset_dir):
    st.write("Downloading dataset...")
    download_file(dataset_url, dataset_zip_path)
    extract_zip_file(dataset_zip_path)
    if zipfile.is_zipfile(dataset_zip_path):
        os.remove(dataset_zip_path)

# Download the model if not already present
if not os.path.exists(model_path):
    st.write("Downloading model...")
    download_file(model_url, model_path)

# Load the trained model
model = load_model(model_path)

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
