import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pathlib
import platform
import os
import requests
import gdown  # A library to download files from Google Drive

#  Streamlit Page Config
st.set_page_config(page_title="Mongolian Food Classifier", page_icon="🍲", layout="centered")

#  Fix for Windows Paths
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

#  Google Drive File ID and Local Path
FILE_ID = '1AmQcU0FoqwZvgTNHMhHGHI-XwH1u03eN'  # Replace with your Google Drive file ID
DESTINATION = 'mongolian_food_classifier.pkl'  # Local path to save the model

#  Function to Download Model from Google Drive
def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

#  Download Model if Not Exists
if not os.path.exists(DESTINATION):
    st.info("Downloading model from Google Drive...")
    download_file_from_google_drive(FILE_ID, DESTINATION)
    st.success("Model downloaded successfully!")

#  Load Model
learn = load_learner(DESTINATION)

#  Load Example Images (From GitHub)
EXAMPLE_IMAGES = {
    "Buuz": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Buuz.jpg",
    "Khuushuur": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Khuushuur.jpg",
    "Tsuivan": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Tsuivan.jpg",
    "Olivier Salad": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Olivier%20Salad.jpg"
}

#  Show Example Images in Sidebar
st.sidebar.title("Example Images")
st.sidebar.write("Click on an image to predict!")

# Create buttons for each example image
for name, url in EXAMPLE_IMAGES.items():
    if st.sidebar.button(name):
        # When a button is clicked, load the image for prediction
        image = Image.open(requests.get(url, stream=True).raw)
        st.session_state.image = image  # Store image in session_state
        st.sidebar.success(f"Image for {name} loaded!")
        break

#  App Header
st.title("🍲 Mongolian Food Classifier")
st.write("Upload an image or click on the examples to predict!")

#  File Uploader (Alternative for manually uploading images)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Use the session-stored image if clicked from the sidebar
if 'image' in st.session_state:
    st.image(st.session_state.image, caption='Example Image', use_container_width=True)
    uploaded_file = None  # Disable manual upload when an image is selected

#  Prediction Logic
if uploaded_file is not None or 'image' in st.session_state:
    try:
        # Use the manually uploaded image or the image selected from the sidebar
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = st.session_state.image

        # Display the selected image
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Make Prediction
        img = PILImage.create(image)
        pred, pred_idx, probs = learn.predict(img)

        # Show Results
        st.success(f"**Prediction:** {pred}")
        st.write("**Confidence Scores:**")
        for i, category in enumerate(learn.dls.vocab):
            st.write(f"- {category}: {probs[i]:.4f}")

    except Exception as e:
        st.error(f"Error: {e}. Please upload a valid image.")

else:
    st.info("Upload an image or click on one of the examples to get started.")
