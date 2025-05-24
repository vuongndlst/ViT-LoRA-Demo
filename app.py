# app.py
from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel, PeftConfig

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load processor
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Load base ViT model
vit_base = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Load LoRA checkpoint from local
vit_lora = PeftModel.from_pretrained(
    vit_base,
    "Trained_Model/vit/lora/checkpoint-62500"
)
vit_lora.eval()

# Transform
vit_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: vit_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0))
])

# Predict function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = vit_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = vit_lora(img_tensor)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "real" if pred == 1 else "fake"

@app.route("/", methods=["GET", "POST"])
def upload_and_classify():
    if request.method == "POST":
        if "image" not in request.files:
            return "No image uploaded", 400
        image = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        result = predict(image_path)
        return render_template("result.html", image=image.filename, result=result)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400
    image = request.files["image"]
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)
    result = predict(path)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
