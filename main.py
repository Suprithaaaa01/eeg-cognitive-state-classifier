import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# --- Re-define the EEGNet architecture (must match training exactly) ---
class EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=321, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.separableConv(self.depthwiseConv(self.firstconv(dummy)))
            flat_size = out.view(1, -1).shape[1]
        self.classify = nn.Linear(flat_size, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)

# --- Load the trained model once at startup ---
device = torch.device('cpu')  # inference only, CPU is fine
model = EEGNet(n_channels=64, n_samples=321, n_classes=2)
model.load_state_dict(torch.load('eegnet_best_model.pt', map_location=device))
model.eval()

LABELS = {0: 'left_hand_imagery', 1: 'right_hand_imagery'}

# --- FastAPI app ---
app = FastAPI(title="EEG Motor Imagery Classifier API")

class EEGInput(BaseModel):
    data: list  # expected shape: 64 channels x 321 timepoints, nested list

@app.get("/")
def root():
    return {"status": "EEG classifier API is running"}

@app.post("/predict")
def predict(input: EEGInput):
    x = np.array(input.data, dtype=np.float32)  # shape (64, 321)

    # Basic validation
    if x.shape != (64, 321):
        return {"error": f"Expected shape (64, 321), got {x.shape}"}

    # Normalize the same way we did during training (per-sample z-score here as a simple approach)
    x = (x - x.mean()) / (x.std() + 1e-8)

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_class = int(torch.argmax(logits, dim=1).item())

    return {
        "prediction": LABELS[pred_class],
        "confidence": round(max(probs), 3),
        "probabilities": {
            LABELS[0]: round(probs[0], 3),
            LABELS[1]: round(probs[1], 3)
        }
    }