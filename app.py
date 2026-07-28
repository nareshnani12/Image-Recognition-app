import os
import streamlit as st
from PIL import Image
import numpy as np
import json
from model_pipeline import ImageRecognitionPipeline

# Set Page Config
st.set_page_config(
    page_title="Image Recognition & Visual Attention Captioning",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1F2937;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .caption-box {
        background-color: #111827;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 6px;
        font-size: 1.15rem;
        font-family: 'Inter', sans-serif;
        color: #F3F4F6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Pipeline Cache
@st.cache_resource
def load_pipeline():
    return ImageRecognitionPipeline()

pipeline = load_pipeline()

# Sidebar Setup
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("---")

vocab_size = st.sidebar.slider("Vocabulary Size", 1000, 20000, 20000, step=1000)
attention_dim = st.sidebar.select_slider("Attention Dimension", options=[128, 256, 512, 1024], value=512)
top_k = st.sidebar.slider("Top-K Sampling", 1, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Select Input Image")

input_option = st.sidebar.radio("Choose source:", ["Preset Samples", "Upload Image"])

sample_dir = r"C:\Users\next\.gemini\antigravity-ide\scratch\image_recognition_project\sample_images"
selected_img_path = None
uploaded_file = None

if input_option == "Preset Samples":
    samples = {
        "Baseball Game": os.path.join(sample_dir, "baseball.jpg"),
        "Surfing Ocean": os.path.join(sample_dir, "surfing.jpg")
    }
    choice = st.sidebar.selectbox("Pick a sample:", list(samples.keys()))
    selected_img_path = samples[choice]
else:
    uploaded_file = st.sidebar.file_uploader("Upload image (JPG/PNG/WEBP)", type=["jpg", "jpeg", "png", "webp"])

# Header UI
st.markdown('<div class="main-title">📷 Image Recognition & Visual Attention Captioning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Deep Learning Pipeline using InceptionResNetV2 + GRU Decoder with Bahdanau Attention</div>', unsafe_allow_html=True)

# Load Selected Image
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif selected_img_path and os.path.exists(selected_img_path):
    image = Image.open(selected_img_path).convert("RGB")

if image is None:
    st.info("👈 Please select a sample image or upload one from the sidebar to get started.")
    st.stop()

# Layout: Main Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Recognition & Captioning", "🔍 Visual Attention Maps", "📜 Notebook & Code Viewer"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Input Image Preview")
        st.image(image, use_container_width=True)
        st.caption(f"Image Resolution: {image.width} x {image.height} px")
        
    with col2:
        st.subheader("Recognized Objects & Scene Features")
        with st.spinner("Analyzing image features..."):
            class_results = pipeline.classify_image(image)
            
        for res in class_results:
            col_lbl, col_bar = st.columns([1.5, 3])
            with col_lbl:
                st.write(f"**{res['label']}**")
            with col_bar:
                st.progress(min(res['confidence'], 1.0), text=f"{res['confidence']*100:.1f}%")
                
        st.markdown("---")
        st.subheader("Generated Captions (Encoder-Decoder Attention)")
        
        if st.button("🚀 Generate New Captions", type="primary"):
            with st.spinner("Running GRU attention decoder..."):
                captions = pipeline.generate_caption(image, num_captions=3)
                for idx, cap in enumerate(captions, 1):
                    st.markdown(f'<div class="caption-box"><b>Candidate {idx}:</b> "{cap}"</div>', unsafe_allow_html=True)
        else:
            captions = pipeline.generate_caption(image, num_captions=3)
            for idx, cap in enumerate(captions, 1):
                st.markdown(f'<div class="caption-box"><b>Candidate {idx}:</b> "{cap}"</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Visual Attention Spatial Heatmaps")
    st.write("Visual attention allows the decoder model to focus on specific 8x8 spatial grid regions of the InceptionResNetV2 feature map when generating each word in the sequence.")
    
    selected_caption = captions[0] if 'captions' in locals() and captions else "a baseball player swinging at a pitch on a field."
    st.info(f"Target Caption: **\"{selected_caption}\"**")
    
    with st.spinner("Generating spatial attention maps per token..."):
        fig = pipeline.generate_attention_visualization(image, selected_caption)
        st.pyplot(fig)

with tab3:
    st.subheader("Original Notebook Code & Architecture")
    
    code_tab1, code_tab2 = st.subplots = st.tabs(["Python Script (image_captioning.py)", "Jupyter Notebook (image_captioning.ipynb)"])
    
    with code_tab1:
        py_file_path = r"C:\Users\next\.gemini\antigravity-ide\scratch\image_recognition_project\image_captioning.py"
        if os.path.exists(py_file_path):
            with open(py_file_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
                
    with code_tab2:
        nb_file_path = r"C:\Users\next\.gemini\antigravity-ide\scratch\image_recognition_project\image_captioning.ipynb"
        if os.path.exists(nb_file_path):
            with open(nb_file_path, "r", encoding="utf-8") as f:
                nb_json = json.load(f)
                st.json(nb_json, expanded=False)
