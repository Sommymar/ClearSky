"""
ClearSky U-Net.

A compact U-Net with two output heads:

  * ``mask``  — sigmoid segmentation map: P(pixel belongs to a satellite streak)
  * ``clean`` — the reconstructed streak-free image (inpainting head)

A shared encoder/decoder backbone feeds both heads, so detection and removal are
learned jointly. This mirrors the U-Net-based segmentation approach that has
become standard for streak removal, extended with a reconstruction head so the
same forward pass both *finds* and *erases* the trail.

Kept intentionally small (~1.9M params at base=16) so it trains on CPU for a
demo while remaining a faithful U-Net.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_r1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_r2")(x)
    return x


def build_unet(size: int = 256, base: int = 16) -> Model:
    inp = layers.Input((size, size, 1), name="image")

    # Encoder
    c1 = _conv_block(inp, base, "enc1")
    p1 = layers.MaxPool2D()(c1)
    c2 = _conv_block(p1, base * 2, "enc2")
    p2 = layers.MaxPool2D()(c2)
    c3 = _conv_block(p2, base * 4, "enc3")
    p3 = layers.MaxPool2D()(c3)

    # Bottleneck
    bn = _conv_block(p3, base * 8, "bottleneck")

    # Decoder
    u3 = layers.UpSampling2D()(bn)
    u3 = layers.Concatenate()([u3, c3])
    d3 = _conv_block(u3, base * 4, "dec3")
    u2 = layers.UpSampling2D()(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = _conv_block(u2, base * 2, "dec2")
    u1 = layers.UpSampling2D()(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = _conv_block(u1, base, "dec1")

    mask = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(d1)
    # Reconstruction head: predict the clean image directly.
    clean = layers.Conv2D(1, 1, activation="sigmoid", name="clean")(d1)

    return Model(inp, {"mask": mask, "clean": clean}, name="clearsky_unet")


def dice_loss(y_true, y_pred, eps: float = 1e-6):
    y_true = tf.cast(y_true, tf.float32)
    num = 2.0 * tf.reduce_sum(y_true * y_pred) + eps
    den = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + eps
    return 1.0 - num / den


def bce_dice(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


if __name__ == "__main__":
    m = build_unet(128, base=16)
    m.summary()
    print("total params:", m.count_params())
