import base64
import json
import os
from io import BytesIO

import flask
from flask import request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


app = flask.Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# --- Local PyTorch model (the one you asked to use) ---
#MODEL_PATH = os.environ.get("SIGN_PTH_PATH", "backend/backend/models/best_model.pth")
#CLASS_MAPPING_PATH = os.environ.get("SIGN_CLASS_MAPPING", "backend/models/class_mapping.json")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

CLASS_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_mapping.json"
)


_model = None
_idx_to_class = None
_device = torch.device("cpu")


class SignResNet18(nn.Module):
    """
    Matches checkpoint keys like:
      model.conv1.weight, ..., model.fc.1.weight, model.fc.4.weight
    """

    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _load_class_mapping():
    global _idx_to_class
    if _idx_to_class is not None:
        return _idx_to_class
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx_to_class_raw = data.get("idx_to_class", {})
    _idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
    return _idx_to_class


def _load_model():
    global _model
    if _model is not None:
        return _model

    ckpt = torch.load(MODEL_PATH, map_location=_device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    m = SignResNet18(num_classes=26).to(_device)
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    _model = m
    return _model


def _decode_data_url_image(data_url: str):
    if not isinstance(data_url, str) or "base64," not in data_url:
        return None
    try:
        b64 = data_url.split("base64,", 1)[1]
        raw = base64.b64decode(b64)
        return Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        return None


@app.route("/health", methods=["GET"])
def health():
    try:
        _load_class_mapping()
        _load_model()
        return jsonify({"ok": True, "model_path": MODEL_PATH, "classes": 26})
    except Exception as e:
        return jsonify({"ok": False, "model_path": MODEL_PATH, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": 'Expected JSON body with key "image"'}), 400

    img = _decode_data_url_image(data.get("image"))
    if img is None:
        return jsonify({"error": "Invalid image payload"}), 400

    try:
        idx_to_class = _load_class_mapping()
        model = _load_model()

        x = _preprocess(img).unsqueeze(0).to(_device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            score = float(probs[idx].item())

        pred = str(idx_to_class.get(idx, "")).strip()
        return jsonify({"prediction": pred, "probability": score * 100.0})
    except Exception as e:
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
