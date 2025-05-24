# streamlit_app.py — Classify real vs fake face using ViT + LoRA (Enhanced UI + Colored Labels)

import os
import streamlit as st
st.set_page_config(page_title="Real vs Fake Face", layout="centered")

from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel

# ------------------------- Config ------------------------- #
BASE_MODEL_ID = "google/vit-base-patch16-224"
LORA_PATH = "model"
NUM_LABELS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load model & transforms ------------------ #
@st.cache_resource(show_spinner="🔄 Loading model...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID, use_fast=True)

    base_model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )

    try:
        model = PeftModel.from_pretrained(base_model, LORA_PATH, is_trainable=False)
    except Exception as e:
        st.error(f"❌ Failed to load LoRA model from '{LORA_PATH}'.\n\n{e}")
        st.stop()

    model.eval().to(DEVICE)

    transforms_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)),
    ])

    return model, transforms_pipeline

model, vit_transforms = load_model()

# -------------------- Inference helper -------------------- #
def predict(img: Image.Image):
    tensor = vit_transforms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor).logits
    probs = torch.softmax(logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    label = "Real" if idx == 1 else "Fake"
    conf = probs[idx].item()
    return label, conf

# ------------------------ UI ------------------------ #
st.markdown("""
<style>
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100%;
    }
    .centered-image img {
        max-width: 30vw;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .result-box {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    .label-real {
        color: #28a745;
    }
    .label-fake {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Real vs Fake Face Classifier (ViT + LoRA)")

st.markdown("Upload a **face image**. The model will predict whether it's a **real** photo or an **AI-generated** fake.")

uploaded = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    label, confidence = predict(img)

    label_class = "label-real" if label == "Real" else "label-fake"
    result_html = f"<div class='result-box {label_class}'>🧾 <b>Prediction:</b> {label} — Confidence: {confidence:.2%}</div>"

    st.markdown("<div class='centered-container'>" + result_html, unsafe_allow_html=True)
    st.markdown("<div class='centered-image'>", unsafe_allow_html=True)
    st.image(img, use_container_width=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

st.caption("Model: ViT-B/16 + Low-Rank Adaptation (LoRA)")
