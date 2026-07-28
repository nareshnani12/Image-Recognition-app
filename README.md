# Image Recognition & Visual Attention Captioning App

An end-to-end Deep Learning Web Application and Jupyter Notebook project for **Image Recognition and Image Captioning with Visual Attention**.

Built using **InceptionResNetV2**, **MobileNetV2**, **TensorFlow/Keras**, **GRU Decoder with Bahdanau/Luong Attention**, and **Streamlit**.

---

## 🌟 Key Features

- **Object Recognition**: Identifies objects and scenes in images with top-5 confidence ratings.
- **Attention-Based Image Captioning**: Generates descriptive text captions using an encoder-decoder architecture.
- **Spatial Attention Heatmaps**: Visualizes $8 \times 8$ feature map attention weights for each predicted word.
- **Interactive Streamlit Web UI**: Easy drag-and-drop image upload, sample gallery, parameter sliders, and interactive tabs.
- **Jupyter Notebook & Script**: Includes full `.ipynb` notebook and standalone `.py` script extracted from the Google Cloud ASL lab dataset.

---

## 📁 Repository Structure

```text
├── app.py                   # Streamlit Web Application
├── model_pipeline.py        # Object Recognition & Visual Attention Pipeline
├── image_captioning.ipynb   # Jupyter Notebook (InceptionResNetV2 + GRU Attention)
├── image_captioning.py      # Standalone Python training & prediction script
├── sample_images/           # Sample test images (baseball, surfing)
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install tensorflow tensorflow-datasets tensorflow-hub streamlit matplotlib pillow numpy
```

### 2. Run the Web Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🧠 Model Architecture

1. **Image Encoder**: Pre-trained InceptionResNetV2 extracts $8 \times 8 \times 1536$ feature maps, reshaped and projected to $64 \times 512$.
2. **Text Decoder**: Embedding layer + GRU + Luong Attention layer.
3. **Loss Function**: Sparse Categorical Crossentropy masked over valid sentence lengths.
