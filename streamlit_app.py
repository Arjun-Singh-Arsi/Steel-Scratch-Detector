import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from utils import load_model, predict

# --- Page Configuration ---
st.set_page_config(
    page_title="Steel Scratch Detector | precision AI",
    page_icon="⚡",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        border: none;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .image-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown("<div style='text-align: center; padding: 2rem 0;'>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #3b82f6; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;'>Industrial Quality Control</h4>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 3rem; font-weight: 800; margin-bottom: 1rem;'>Steel Scratch Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>High-precision surface defect segmentation using ResUNet++ with Attention</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Device & Model Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'steel_defect_resunet_final (1).pth'

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH, device)
    return None

model = get_model()

# --- Main Interface ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Surface Image")
    uploaded_file = st.file_uploader("Select a steel surface image to analyze...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Surface", use_container_width=True)
        
        analyze_btn = st.button("🚀 Analyze Surface")
        
        if analyze_btn:
            if model is None:
                st.error("Model weights not found. Please ensure the model file exists in the 'model/' directory.")
            else:
                with st.spinner("Analyzing surface for defects..."):
                    # Save temporarily to disk for processing (utils.predict expects a path)
                    temp_path = "temp_upload.png"
                    input_image.save(temp_path)
                    
                    # Run prediction
                    overlay_image, defect_info = predict(temp_path, model, device)
                    
                    # Store results in session state
                    st.session_state.overlay = overlay_image
                    st.session_state.defects = defect_info
                    
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if "overlay" in st.session_state:
        st.markdown("<p class='image-label'>Detection Overlay</p>", unsafe_allow_html=True)
        st.image(st.session_state.overlay, caption="Defect Segmentation Overlay", use_container_width=True)
        
        st.markdown("### 📝 Defect Report")
        for defect in st.session_state.defects:
            with st.container():
                st.markdown(f"""
                    <div class='metric-card'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 1.1rem; font-weight: 600;'>{defect['class']}</span>
                            <span style='padding: 4px 12px; border-radius: 20px; background-color: {"#10b98122" if defect['detected'] else "#ef444422"}; color: {"#10b981" if defect['detected'] else "#ef4444"}; font-size: 0.8rem; font-weight: bold;'>
                                {"DETECTED" if defect['detected'] else "NONE"}
                            </span>
                        </div>
                        <div style='margin-top: 0.5rem; color: #94a3b8; font-size: 0.9rem;'>
                            Confidence: {defect['confidence']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Upload an image and click 'Analyze Surface' to see the results.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Powered by ResUNet++ | precision AI Systems</p>", unsafe_allow_html=True)
