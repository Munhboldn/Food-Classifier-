import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pathlib
import platform
import os
import requests
import gdown
from typing import Dict, Tuple
import time

# Constants
MODEL_INFO = {
    "FILE_ID": "1AmQcU0FoqwZvgTNHMhHGHI-XwH1u03eN",
    "DESTINATION": "mongolian_food_classifier.pkl"
}

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

class FoodClassifier:
    def __init__(self):
        self.setup_page_config()
        self.fix_windows_paths()
        self.customize_theme()
        self.model = self.load_model()
        
    @staticmethod
    def setup_page_config():
        st.set_page_config(
            page_title="Mongolian Food Classifier",
            page_icon=None,
            layout="centered",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def customize_theme():
        st.markdown("""
            <style>
            .stButton>button {
                width: 100%;
                border-radius: 5px;
            }
            .stProgress > div > div > div {
                background-color: #4CAF50;
            }
            </style>
        """, unsafe_allow_html=True)
        
    @staticmethod
    def fix_windows_paths():
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
            
    def load_model(self) -> object:
        if not os.path.exists(MODEL_INFO["DESTINATION"]):
            with st.spinner("Downloading model from Google Drive..."):
                self.download_model()
        return load_learner(MODEL_INFO["DESTINATION"])
    
    @staticmethod
    def download_model():
        try:
            url = f"https://drive.google.com/uc?id={MODEL_INFO['FILE_ID']}"
            gdown.download(url, MODEL_INFO["DESTINATION"], quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.stop()
    
    def predict(self, image: Image) -> Tuple[str, float, Dict[str, float]]:
        img = PILImage.create(image)
        pred, pred_idx, probs = self.model.predict(img)
        confidence_scores = {
            category: float(prob) 
            for category, prob in zip(self.model.dls.vocab, probs)
        }
        return str(pred), float(probs[pred_idx]), confidence_scores
    
    def create_ui(self):
        self.create_header()
        self.create_sidebar()
        self.handle_image_upload()
        self.create_footer()
        
    def create_header(self):
        st.title("Mongolian Food Classifier")
        st.markdown("""
        Welcome to the Mongolian Food Classifier! This app uses machine learning to identify 
        traditional Mongolian dishes. Upload your own image or try our example images to get started.
        """)
        
    def create_sidebar(self):
        st.sidebar.title("Example Images")
        st.sidebar.write("Click on an image to try it out!")
        
        for name, url in EXAMPLE_IMAGES.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"**{name}**")
                st.write(FOOD_DESCRIPTIONS[name])
            with col2:
                if st.button("Try", key=f"btn_{name}"):
                    with st.spinner("Loading image..."):
                        image = Image.open(requests.get(url, stream=True).raw)
                        st.session_state.image = image
                        st.session_state.source = "example"
                        
    def handle_image_upload(self):
        upload_col, preview_col = st.columns([2, 1])
        
        with upload_col:
            uploaded_file = st.file_uploader(
                "Upload your own image:",
                type=["jpg", "png", "jpeg"],
                help="Supported formats: JPG, JPEG, PNG"
            )
            
            if uploaded_file:
                st.session_state.image = Image.open(uploaded_file)
                st.session_state.source = "upload"
                
        if 'image' in st.session_state:
            self.process_image(st.session_state.image)
            
    def process_image(self, image: Image):
        st.image(image, caption='Selected Image', use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            time.sleep(0.5)  # Add slight delay for better UX
            try:
                pred, confidence, all_scores = self.predict(image)
                
                # Display results
                st.success(f"**Predicted Dish:** {pred}")
                st.info(f"**Description:** {FOOD_DESCRIPTIONS.get(pred, 'No description available.')}")
                
                # Display confidence scores with a progress bar
                st.subheader("Confidence Scores")
                for category, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{category}")
                    st.progress(score)
                    st.write(f"Confidence: {score:.2%}")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.write("Please try uploading a different image.")

    def create_footer(self):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                """
                <div style='text-align: center'>
                    <p>Created by <a href="https://github.com/Munhboldn">Munkhbold Nyamdorj</a></p>
                    <p>
                        <a href="https://github.com/Munhboldn/Food-Classifier-" target="_blank">
                            <img src="https://img.shields.io/github/stars/Munhboldn/Food-Classifier-?style=social" alt="GitHub stars">
                        </a>
                    </p>
                    <p style='font-size: 0.875em; color: #666;'>
                        Powered by FastAI & Streamlit | Version 1.0.0
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

def main():
    classifier = FoodClassifier()
    classifier.create_ui()

if __name__ == "__main__":
    main()
