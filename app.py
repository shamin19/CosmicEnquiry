import streamlit as st
import os
import zipfile
from utils import load_model, predict_image
from PIL import Image
from auth import authenticate_and_create_drive

# IDs for the dataset and model on Google Drive
dataset_file_id = '1e23T43mfIKI-_qnZ6IwkU4bpl6tu0N6v'
model_file_id = '1g_QYE3DVhZPHQKavpqMJ-asaG3lvW6D9'

# Paths where the dataset and model will be saved
dataset_zip_path = 'NASA APOD Dataset.zip'
dataset_dir = 'NASA APOD Dataset'
model_path = 'model.h5'

# Function to download files using PyDrive
def download_file_from_gdrive(drive, file_id, output):
    try:
        file = drive.CreateFile({'id': file_id})
        file.GetContentFile(output)
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return False
    return True

def extract_zip_file(zip_path, extract_to='.'):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        st.error(f"The file {zip_path} is not a valid zip file.")

# Authenticate and create PyDrive client
drive = authenticate_and_create_drive()

# Download and extract the dataset if not already present
if not os.path.exists(dataset_dir):
    st.write("Downloading dataset...")
    if download_file_from_gdrive(drive, dataset_file_id, dataset_zip_path):
        extract_zip_file(dataset_zip_path)
        if zipfile.is_zipfile(dataset_zip_path):
            os.remove(dataset_zip_path)

# Download the model if not already present
if not os.path.exists(model_path):
    st.write("Downloading model...")
    if download_file_from_gdrive(drive, model_file_id, model_path):
        # Check if the model file is valid
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1:
            st.error("The downloaded model file is not valid.")
            os.remove(model_path)

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
