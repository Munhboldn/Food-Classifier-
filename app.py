import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pathlib
import platform
import os
import requests
import gdown  # A library to download files from Google Drive

# 🔧 Streamlit Page Config
st.set_page_config(page_title="Mongolian Food Classifier", page_icon="🍲", layout="centered")

# 🔧 Fix for Windows Paths
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# 🔥 Google Drive File ID and Local Path
FILE_ID = '1AmQcU0FoqwZvgTNHMhHGHI-XwH1u03eN'  # Replace with your Google Drive file ID
DESTINATION = 'mongolian_food_classifier.pkl'  # Local path to save the model

# 📥 Function to Download Model from Google Drive
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

# 📌 Download Model if Not Exists
if not os.path.exists(DESTINATION):
    st.info("Downloading model from Google Drive...")
    download_file_from_google_drive(FILE_ID, DESTINATION)
    st.success("Model downloaded successfully!")

# 🎯 Load Model
learn = load_learner(DESTINATION)

# 🖼️ Load Example Images (From GitHub)
EXAMPLE_IMAGES = {
    "Buuz": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Buuz.jpg",
    "Khuushuur": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Khuushuur.jpg",
    "Tsuivan": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Tsuivan.jpg",
    "Olivier Salad": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Olivier%20Salad.jpg"
}

# 📌 Show Example Images in Sidebar
st.sidebar.title("Example Images")
st.sidebar.write("Drag & drop these images to test!")

# Try to fetch and display each example image
for name, url in EXAMPLE_IMAGES.items():
    try:
        # Check if the image is accessible
        response = requests.get(url)
        response.raise_for_status()  # Will raise an exception for non-2xx responses
        st.sidebar.image(url, caption=name, use_container_width=True)  # Use the new parameter
    except requests.exceptions.RequestException as e:
        st.sidebar.warning(f"Failed to load {name}. Error: {str(e)}")

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
        st.image(image, caption='Uploaded Image', use_container_width=True)  # Use the new parameter

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
