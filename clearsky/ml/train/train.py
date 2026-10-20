"""
Train the ClearSky U-Net on synthetic star-field / satellite-streak pairs.

Usage:
    python -m ml.train.train --n 2000 --size 128 --epochs 15

For a quick smoke test (used in CI and the README figures):
    python -m ml.train.train --n 64 --size 128 --epochs 1 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.data.synthesize import build_dataset          # noqa: E402
from ml.models.unet import build_unet, bce_dice        # noqa: E402


def iou_metric(y_true, y_pred, thr: float = 0.5):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > thr, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return inter / (union + 1e-6)


def detection_rate(mask_true, mask_pred, thr: float = 0.5):
    """Fraction of true streak pixels recovered (recall) — the headline metric."""
    mt = (mask_true > 0.5).astype(np.float32)
    mp = (mask_pred > thr).astype(np.float32)
    tp = float((mt * mp).sum())
    total = float(mt.sum())
    return tp / (total + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--out", type=str, default="ml/models/clearsky.weights.h5")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"Building {args.n} synthetic samples @ {args.size}px ...")
    X, C, M = build_dataset(args.n, size=args.size, seed=0)
    n_val = max(1, int(0.15 * args.n))
    Xtr, Xval = X[:-n_val], X[-n_val:]
    Ctr, Cval = C[:-n_val], C[-n_val:]
    Mtr, Mval = M[:-n_val], M[-n_val:]

    model = build_unet(args.size, base=args.base)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"mask": bce_dice, "clean": "mae"},
        loss_weights={"mask": 1.0, "clean": 0.5},
    )

    model.fit(
        Xtr, {"mask": Mtr, "clean": Ctr},
        validation_data=(Xval, {"mask": Mval, "clean": Cval}),
        epochs=args.epochs, batch_size=args.batch, verbose=2,
    )

    pred = model.predict(Xval, verbose=0)
    det = detection_rate(Mval, pred["mask"])
    iou = float(iou_metric(Mval, pred["mask"]).numpy())
    psnr = float(tf.reduce_mean(tf.image.psnr(Cval, pred["clean"], max_val=1.0)).numpy())
    metrics = {"detection_rate": round(det, 4), "iou": round(iou, 4), "psnr_db": round(psnr, 2)}
    print("VALIDATION METRICS:", json.dumps(metrics, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save_weights(args.out)
    with open("ml/models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved weights -> {args.out}")


if __name__ == "__main__":
    main()
