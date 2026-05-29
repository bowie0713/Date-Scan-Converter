# ============================================================
# server.py — Datestamp inference server
# Wraps the trained CRNN+CTC model behind a single POST /predict endpoint.
# ============================================================
import io
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ---- path to the trained weights (keep the .pt file next to this script) ----
CHECKPOINT_PATH = "best_crnn_avery_30k.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL — must match Avery's trained architecture exactly.
# This is the 4-Conv-block CRNN (the Colab notebook had only 3).
# ============================================================
class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1, cnn_out_channels=256,
                 lstm_hidden_size=256, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),

            nn.Conv2d(128, cnn_out_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_out_channels), nn.ReLU(inplace=True),

            # --- 4th block: present in Avery's model, absent in the Colab code ---
            nn.Conv2d(cnn_out_channels, cnn_out_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_out_channels), nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        lstm_input_size = cnn_out_channels * 8
        self.rnn = nn.LSTM(
            input_size=lstm_input_size, hidden_size=lstm_hidden_size,
            num_layers=lstm_layers, bidirectional=True, batch_first=False,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.shape
        features = features.permute(3, 0, 1, 2).contiguous().view(w, b, c * h)
        recurrent, _ = self.rnn(features)
        logits = self.classifier(recurrent)
        return F.log_softmax(logits, dim=2)


def greedy_decode(log_probs, blank_idx=0):
    """CTC greedy decode: argmax -> collapse repeats -> drop blanks."""
    best = torch.argmax(log_probs, dim=2).permute(1, 0).cpu().tolist()
    decoded = []
    for seq in best:
        collapsed, prev = [], None
        for idx in seq:
            if idx != blank_idx and idx != prev:
                collapsed.append(idx)
            prev = idx
        decoded.append(collapsed)
    return decoded


# ============================================================
# DATE NORMALIZATION — turn raw model text into YYYY-MM-DD.
# ============================================================
DATE_PATTERNS = [
    "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%d/%m/%y",
    "%B %d, %Y", "%d %B %Y", "%d %b %Y", "%b %d, %Y",
    "%m-%d-%Y", "%d-%m-%Y", "%m.%d.%Y", "%Y-%m-%d", "%d %b '%y",
]


def normalize_date_string(text):
    text = " ".join(text.strip().split())
    for pattern in DATE_PATTERNS:
        try:
            return datetime.strptime(text, pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ============================================================
# LOAD MODEL ONCE AT STARTUP (not per-request — that would be slow).
# ============================================================
print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
VOCAB_CHARS = checkpoint["vocab_chars"]
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(VOCAB_CHARS)}
IMG_H = checkpoint["image_height"]
IMG_W = checkpoint["image_width"]
NUM_CLASSES = checkpoint["model_state_dict"]["classifier.weight"].shape[0]

MODEL = CRNN(num_classes=NUM_CLASSES).to(DEVICE)
MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.eval()
print(f"Model ready. device={DEVICE}  classes={NUM_CLASSES}  input={IMG_H}x{IMG_W}")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Datestamp API")

# Allow the browser frontend (Live Server, file://, etc.) to call this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # fine for local dev; restrict for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "message": "Datestamp API is running"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Receive an image, return {date, raw, normalized}."""
    try:
        raw_bytes = await image.read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    except Exception:
        return {"error": "Could not read the uploaded file as an image."}

    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        log_probs = MODEL(x)
        pred_ids = greedy_decode(log_probs, blank_idx=0)[0]

    raw_text = "".join(IDX_TO_CHAR[i] for i in pred_ids if i in IDX_TO_CHAR)
    normalized = normalize_date_string(raw_text)

    return {
        "raw": raw_text,                       # exactly what the model output
        "date": normalized or raw_text,        # YYYY-MM-DD if parseable, else raw
        "normalized": normalized is not None,  # False = couldn't parse a clean date
    }
