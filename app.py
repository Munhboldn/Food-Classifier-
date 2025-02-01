import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pathlib
import platform
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import requests

# 🔧 Streamlit Page Config
st.set_page_config(page_title="Mongolian Food Classifier", page_icon="🍲", layout="centered")

# 🔧 Fix for Windows Paths
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# 🔥 Google Drive API Setup
SERVICE_ACCOUNT_FILE = 'food-classifier-449613-f4606430ec12.json'  # UPLOAD THIS FILE
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FILE_ID = '1AmQcU0FoqwZvgTNHMhHGHI-XwH1u03eN'  # Your Model File ID from Google Drive
DESTINATION = 'mongolian_food_classifier.pkl'  # Local Model Path

# 📥 Function to Download Model from Google Drive
def download_file_from_google_drive(file_id, destination):
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    
    with open(destination, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

# 📌 Download Model if Not Exists
if not os.path.exists(DESTINATION):
    st.info("Downloading model from Google Drive...")
    download_file_from_google_drive(FILE_ID, DESTINATION)
    st.success("Model downloaded successfully!")

# 🎯 Load Model
learn = load_learner(DESTINATION)

# 🖼️ Load Example Images (From GitHub)
EXAMPLE_IMAGES = {
    "Buuz": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier/main/Example_Images/Buuz.jpg",
    "Khuushuur": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier/main/Example_Images/Khuushuur.jpg",
    "Tsuivan": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier/main/Example_Images/Tsuivan.jpg",
    "Olivier Salad": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier/main/Example_Images/Olivier%20Salad.jpg"
}

# 📌 Show Example Images in Sidebar
st.sidebar.title("Example Images")
st.sidebar.write("Drag & drop these images to test!")

for name, url in EXAMPLE_IMAGES.items():
    response = requests.get(url)
    if response.status_code == 200:
        st.sidebar.image(url, caption=name, use_column_width=True)
    else:
        st.sidebar.warning(f"Failed to load {name}.")

# 📢 App Header
st.title("🍲 Mongolian Food Classifier")
st.write("Upload an image or drag one from the examples to predict!")

# 🖼️ File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# 🗑️ Clear Button
if st.button("Clear Image"):
    st.session_state.uploader = None
    st.experimental_rerun()

# 🏆 Prediction Logic
if uploaded_file is not None:
    try:
        # Show Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make Prediction
        img = PILImage.create(uploaded_file)
        pred, pred_idx, probs = learn.predict(img)

        # Show Results
        st.success(f"**Prediction:** {pred}")
        st.write("**Confidence Scores:**")
        for i, category in enumerate(learn.dls.vocab):
            st.write(f"- {category}: {probs[i]:.4f}")

    except Exception as e:
        st.error(f"Error: {e}. Please upload a valid image.")

else:
    st.info("Upload an image to get started.")
