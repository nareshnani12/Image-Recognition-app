import os
import json
import streamlit as st
from PIL import Image
import numpy as np
from model_pipeline import ImageRecognitionPipeline

# Base directory for relative file paths on both local and cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set Page Config
st.set_page_config(
    page_title="Image Recognition & Visual Attention Captioning",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .caption-box {
        background-color: #F3F4F6;
        border-left: 4px solid #2563EB;
        padding: 1rem;
        border-radius: 6px;
        font-size: 1.1rem;
        color: #1F2937;
        margin: 0.6rem 0;
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

sample_dir = os.path.join(BASE_DIR, "sample_images")
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
st.markdown('<div class="sub-title">End-to-End Deep Learning Pipeline using InceptionResNetV2 + GRU Decoder with Visual Attention</div>', unsafe_allow_html=True)

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
    
    sub_tab1, sub_tab2 = st.tabs(["Python Script (image_captioning.py)", "Jupyter Notebook (image_captioning.ipynb)"])
    
    with sub_tab1:
        py_file_path = os.path.join(BASE_DIR, "image_captioning.py")
        if os.path.exists(py_file_path):
            with open(py_file_path, "r", encoding="utf-8") as f:
                code_text = f.read()
                st.code(code_text, language="python")
        else:
            st.warning("image_captioning.py file not found.")

    with sub_tab2:
        nb_file_path = os.path.join(BASE_DIR, "image_captioning.ipynb")
        if os.path.exists(nb_file_path):
            with open(nb_file_path, "r", encoding="utf-8") as f:
                nb_data = json.load(f)
                
            st.caption(f"Notebook loaded: {len(nb_data.get('cells', []))} cells")
            
            # Display notebook cells nicely formatted
            for idx, cell in enumerate(nb_data.get("cells", [])):
                cell_type = cell.get("cell_type")
                source = "".join(cell.get("source", []))
                if cell_type == "markdown":
                    st.markdown(source)
                elif cell_type == "code":
                    st.code(source, language="python")
        else:
            st.warning("image_captioning.ipynb file not found.")
