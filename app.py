import streamlit as st
import os
import requests
import pandas as pd
from utils import load_model, predict_image
from PIL import Image

# Folder ID for the dataset in Google Drive
folder_id = '1gbU1BcxFsQmhzOt5BzU8DTasRcAjsO9Y'

# Model file ID and path
model_file_id = '1g_QYE3DVhZPHQKavpqMJ-asaG3lvW6D9'
model_path = 'model.h5'

# Function to list files in a Google Drive folder
def list_files_in_folder(folder_id):
    url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    gdown_output = "file_list.csv"
    os.system(f"gdown --folder {url} -O {gdown_output}")
    files_df = pd.read_csv(gdown_output)
    return files_df

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
        return False
    return True

# Function to download Google Drive file by ID
def download_gdrive_file(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return download_file(url, dest_path)

# List files in the folder
files_df = list_files_in_folder(folder_id)

# Download and save dataset files if not already present
for index, row in files_df.iterrows():
    file_id = row["id"]
    file_name = row["name"]
    file_path = os.path.join("NASA APOD Dataset", file_name)
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        st.write(f"Downloading {file_path}...")
        download_gdrive_file(file_id, file_path)

# Download the model if not already present
if not os.path.exists(model_path):
    st.write("Downloading model...")
    if download_gdrive_file(model_file_id, model_path):
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
