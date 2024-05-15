import os
import requests
import streamlit as st
from utils import load_model, predict_image
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# Set up the necessary API and authentication
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_gdrive():
    """Authenticate and create the PyDrive client."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(service, folder_id):
    """List all files in a Google Drive folder."""
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, pageSize=1000, fields="files(id, name)").execute()
    items = results.get('files', [])
    if not items:
        print('No files found.')
    else:
        return items

# Provide your folder ID here
FOLDER_ID = '1gbU1BcxFsQmhzOt5BzU8DTasRcAjsO9Y'

# Authenticate and create the service
service = authenticate_gdrive()

# List all files in the folder
files = list_files_in_folder(service, FOLDER_ID)

# Create the list of files to download
files_to_download = [{"id": file['id'], "path": f"NASA_APOD_Dataset/{file['name']}"} for file in files]

# Model file ID
model_file_id = '1g_QYE3DVhZPHQKavpqMJ-asaG3lvW6D9'

# Paths where the model will be saved
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
        return False
    return True

# Function to download Google Drive file by ID
def download_gdrive_file(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return download_file(url, dest_path)

# Download and save dataset files if not already present
for file_info in files_to_download:
    file_id = file_info["id"]
    file_path = file_info["path"]
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        st.write(f"Downloading {file_path}...")
        download_gdrive_file(file_id, file_path)

# Download the model if not already present
if not os.path.exists(model_path):
    st.write("Downloading model...")
    download_gdrive_file(model_file_id, model_path)
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
