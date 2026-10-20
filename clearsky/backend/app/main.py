"""
ClearSky inference API.

A small FastAPI service that loads the trained U-Net once at startup and exposes
a single endpoint: upload a telescope image, get back the streak mask overlay
and the cleaned image, base64-encoded, in real time.

Run:
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_SIZE = int(os.getenv("CLEARSKY_SIZE", "128"))
WEIGHTS = os.getenv("CLEARSKY_WEIGHTS", "ml/models/clearsky.weights.h5")

app = FastAPI(title="ClearSky API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    """Lazy-load the model so the API imports fast and tests can stub it."""
    global _model
    if _model is None:
        import sys
        sys.path.insert(0, os.getcwd())
        from ml.models.unet import build_unet
        m = build_unet(MODEL_SIZE, base=16)
        if os.path.exists(WEIGHTS):
            m.load_weights(WEIGHTS)
        _model = m
    return _model


def _to_b64_png(arr: np.ndarray) -> str:
    arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok", "model_size": MODEL_SIZE}


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/tiff"}:
        raise HTTPException(415, "Upload a PNG, JPEG, or TIFF image.")
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        raise HTTPException(400, "Could not read the image.")

    orig = img.size
    x = np.asarray(img.resize((MODEL_SIZE, MODEL_SIZE)), dtype=np.float32) / 255.0
    pred = get_model().predict(x[None, ..., None], verbose=0)
    mask = pred["mask"][0, ..., 0]
    clean = pred["clean"][0, ..., 0]

    # Calibrated thresholding: a fixed 0.5 cut is brittle when the sigmoid head
    # is not fully saturated. We threshold relative to the mask's own dynamic
    # range, which keeps the overlay meaningful across training maturities.
    thr = float(mask.mean() + 2.0 * mask.std())
    thr = min(max(thr, 0.15), 0.6)
    mask = (mask >= thr).astype(np.float32) * mask

    # Red overlay of the detected streak on the input, for the UI.
    rgb = np.stack([x, x, x], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] + mask * 0.8, 0, 1)
    rgb[..., 1] *= (1 - mask * 0.6)
    rgb[..., 2] *= (1 - mask * 0.6)

    coverage = float((mask > 0.5).mean())
    return {
        "original_size": {"width": orig[0], "height": orig[1]},
        "streak_coverage": round(coverage, 4),
        "detected": bool(coverage > 0.002),
        "input": _to_b64_png(x),
        "overlay": _to_b64_png(rgb),
        "cleaned": _to_b64_png(clean),
        "mask": _to_b64_png(mask),
    }
