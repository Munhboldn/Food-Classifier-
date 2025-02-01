import streamlit as st
from fastai.vision.all import *
from PIL import Image
import requests
import gdown
import time
import os

# Constants
MODEL_FILE = "mongolian_food_classifier.pkl"
MODEL_ID = "1AmQcU0FoqwZvgTNHMhHGHI-XwH1u03eN"
EXAMPLE_IMAGES = {
    "Buuz": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Buuz.jpg",
    "Khuushuur": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Khuushuur.jpg",
    "Tsuivan": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Tsuivan.jpg",
    "Olivier Salad": "https://raw.githubusercontent.com/Munhboldn/Food-Classifier-/main/Example_Images/Olivier%20Salad.jpg"
}
FOOD_DESCRIPTIONS = {
    "Buuz": "A traditional Mongolian steamed dumpling filled with minced meat.",
    "Khuushuur": "Deep-fried meat pastries, similar to empanadas.",
    "Tsuivan": "Stir-fried noodles with meat and vegetables.",
    "Olivier Salad": "A popular potato salad with vegetables and meat."
}

# Load model
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.spinner("Downloading model...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_FILE, quiet=False)
    return load_learner(MODEL_FILE)

# Predict food from image
def predict(model, image):
    img = PILImage.create(image)
    pred, _, probs = model.predict(img)
    return pred, float(probs.max())

# Main app
def main():
    st.set_page_config(page_title="Mongolian Food Classifier", layout="centered")

    # Load model
    model = load_model()

    # Title and description
    st.title("Mongolian Food Classifier")
    st.markdown("Upload an image of Mongolian food to classify it.")

    # Sidebar for example images
    st.sidebar.title("Example Images")
    for name, url in EXAMPLE_IMAGES.items():
        if st.sidebar.button(f"Try {name}"):
            image = Image.open(requests.get(url, stream=True).raw)
            st.image(image, caption=name)
            pred, confidence = predict(model, image)
            st.success(f"Predicted: {pred} with {confidence*100:.2f}% confidence")
            st.info(FOOD_DESCRIPTIONS.get(pred, "No description available."))
    
    # File upload for user image
    uploaded_file = st.file_uploader("Upload your own image:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')
        with st.spinner("Analyzing..."):
            time.sleep(0.5)  # Delay for user experience
            pred, confidence = predict(model, image)
            st.success(f"Predicted: {pred} with {confidence*100:.2f}% confidence")
            st.info(FOOD_DESCRIPTIONS.get(pred, "No description available."))

if __name__ == "__main__":
    main()
