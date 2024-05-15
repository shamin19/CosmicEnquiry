import streamlit as st
import gdown
import os
import zipfile
from utils import load_model, predict_image

# URLs for the dataset and model on Google Drive
dataset_url = 'https://drive.google.com/file/d/1e23T43mfIKI-_qnZ6IwkU4bpl6tu0N6v'
model_url = 'https://drive.google.com/file/d/1g_QYE3DVhZPHQKavpqMJ-asaG3lvW6D9'

# Paths where the dataset and model will be saved
dataset_zip_path = 'NASA_APOD_Dataset.zip'
dataset_dir = 'NASA_APOD_Dataset'
model_path = 'model.h5'

# Function to download files from Google Drive
def download_file_from_gdrive(url, output):
    gdown.download(url, output, quiet=False)

def extract_zip_file(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Download and extract the dataset if not already present
if not os.path.exists(dataset_dir):
    download_file_from_gdrive(dataset_url, dataset_zip_path)
    extract_zip_file(dataset_zip_path)
    os.remove(dataset_zip_path)

# Download the model if not already present
if not os.path.exists(model_path):
    download_file_from_gdrive(model_url, model_path)

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
