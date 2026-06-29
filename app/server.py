# ============================================================
# server.py — Datestamp inference server (Model 2: 75-epoch word-level CRNN)
# ============================================================
# Key differences from model 1's server:
#   - Word-level vocabulary (338 tokens: full month names, days, years)
#     instead of character-level (42 chars).
#   - CRNN architecture includes Dropout2d + pre-classifier Dropout
#     (regularization fixes that produced the 75-epoch model).
# ============================================================
import io
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", "models/best_crnn_30k_75ep_dylan.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# WORD-LEVEL VOCABULARY — must match the training notebook exactly.
# ============================================================
class OCRWordVocabulary:
    """Maps word-level tokens <-> integer IDs."""

    TOKENS = [
        # Full month names
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        # Month abbreviations
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec",
        # Separators
        "/", "-", ".", ",", "'", " ",
        # Zero-padded day/month "01".."31"
        *[f"{i:02d}" for i in range(1, 32)],
        # Bare day/month "1".."31"
        *[str(i) for i in range(1, 32)],
        # Four-digit years 1900..2099
        *[str(y) for y in range(1900, 2100)],
        # Two-digit years "00".."99"
        *[f"{y:02d}" for y in range(0, 100)],
    ]

    def __init__(self):
        self.blank_idx = 0
        seen, unique = set(), []
        for t in self.TOKENS:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        self.tokens = unique
        self.idx_to_token: Dict[int, str] = {i + 1: t for i, t in enumerate(self.tokens)}

    @property
    def num_classes(self) -> int:
        return len(self.tokens) + 1   # +1 for CTC blank

    def decode(self, indices: List[int]) -> str:
        return "".join(self.idx_to_token[i] for i in indices if i in self.idx_to_token)


# ============================================================
# CRNN — matches the notebook's architecture exactly:
#   4 conv blocks, BatchNorm everywhere, Dropout2d after blocks 1-2,
#   LSTM dropout 0.3, pre-classifier dropout 0.3.
# ============================================================
class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=1, cnn_out_channels=256,
                 lstm_hidden_size=256, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1), nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1), nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(cnn_out_channels), nn.ReLU(inplace=True),

            nn.Conv2d(256, cnn_out_channels, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        lstm_input_size = cnn_out_channels * 8
        self.rnn = nn.LSTM(
            input_size=lstm_input_size, hidden_size=lstm_hidden_size,
            num_layers=lstm_layers, bidirectional=True, batch_first=False,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )
        self.pre_classifier_dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.shape
        features = features.permute(3, 0, 1, 2).contiguous().view(w, b, c * h)
        recurrent, _ = self.rnn(features)
        recurrent = self.pre_classifier_dropout(recurrent)
        logits = self.classifier(recurrent)
        return F.log_softmax(logits, dim=2)


def greedy_decode(log_probs, blank_idx=0):
    """CTC greedy: argmax -> collapse repeats -> drop blanks."""
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
# DATE NORMALIZATION — same patterns as the training notebook.
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
# LOAD MODEL ONCE AT STARTUP
# ============================================================
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found at {CHECKPOINT_PATH}. "
        "Set MODEL_CHECKPOINT=/path/to/best_crnn_30k_75ep_dylan.pt or place the file in models/."
    )

print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
VOCAB = OCRWordVocabulary()
IMG_H = checkpoint["image_height"]
IMG_W = checkpoint["image_width"]

MODEL = CRNN(num_classes=VOCAB.num_classes).to(DEVICE)
MODEL.load_state_dict(checkpoint["model_state_dict"])
MODEL.eval()
print(f"Model ready. device={DEVICE}  classes={VOCAB.num_classes}  input={IMG_H}x{IMG_W}")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Datestamp API (Model 2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "model": "75-epoch word-level CRNN", "classes": VOCAB.num_classes}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        raw_bytes = await image.read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    except Exception:
        return {"error": "Could not read the uploaded file as an image."}

    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        log_probs = MODEL(x)
        pred_ids = greedy_decode(log_probs, blank_idx=0)[0]

    raw_text = VOCAB.decode(pred_ids)
    normalized = normalize_date_string(raw_text)

    return {
        "raw": raw_text,
        "date": normalized or raw_text,
        "normalized": normalized is not None,
    }
