"""
ClearSky — synthetic astronomical image generator.

Real annotated satellite-streak datasets are scarce, so the standard approach in
the literature (e.g. the U-Net streak-removal work of Betgen et al. and the
mixed real/synthetic training used by ZTF) is to inject synthetic satellite
trails onto realistic star fields. This module builds paired samples:

    (streaked_image, clean_image, streak_mask)

Each star field is generated with a Poisson-distributed star count, Gaussian PSF
stars with a realistic magnitude/flux distribution, sky background, and read
noise. Satellite trails are drawn as anti-aliased lines with a transverse PSF,
brightness taper toward the ends, and optional gaps (matching the broken
appearance of tumbling/blinking satellites).

Everything is float32 in [0, 1]. No external downloads required.
"""

from __future__ import annotations

import numpy as np


def _gaussian_psf_star(canvas: np.ndarray, x: float, y: float, flux: float, sigma: float) -> None:
    """Add a single Gaussian point-spread-function star into canvas in place."""
    h, w = canvas.shape
    radius = int(np.ceil(3 * sigma))
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2)))
    canvas[y0:y1, x0:x1] += flux * g


def make_star_field(size: int = 256, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a realistic clean star field in [0, 1]."""
    rng = rng or np.random.default_rng()
    img = np.zeros((size, size), dtype=np.float32)

    # Sky background gradient (light pollution / moon glow), subtle.
    gy, gx = np.mgrid[0:size, 0:size] / size
    background = 0.02 + 0.03 * (gx * rng.uniform(-1, 1) + gy * rng.uniform(-1, 1) + 1) / 2
    img += background.astype(np.float32)

    # Stars: count scales with area; fluxes follow a steep power law (many faint,
    # few bright), which mimics a real magnitude distribution.
    n_stars = rng.poisson(size * size / 900)
    for _ in range(n_stars):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        flux = float(rng.pareto(2.5) * 0.15)
        flux = min(flux, 1.0)
        sigma = rng.uniform(0.8, 1.8)
        _gaussian_psf_star(img, x, y, flux, sigma)

    # Photon (shot) noise + Gaussian read noise.
    img = rng.poisson(np.clip(img, 0, None) * 255.0) / 255.0
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def add_satellite_streak(
    clean: np.ndarray,
    rng: np.random.Generator | None = None,
    n_streaks: tuple[int, int] = (1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Inject 1+ satellite trails. Returns (streaked_image, binary_mask)."""
    rng = rng or np.random.default_rng()
    size = clean.shape[0]
    streaked = clean.copy()
    mask = np.zeros_like(clean)

    for _ in range(rng.integers(n_streaks[0], n_streaks[1] + 1)):
        # A trail enters and exits the frame: pick two points on the border.
        edge = rng.integers(0, 4)
        if edge == 0:
            p0 = np.array([rng.uniform(0, size), 0.0])
        elif edge == 1:
            p0 = np.array([rng.uniform(0, size), size - 1.0])
        elif edge == 2:
            p0 = np.array([0.0, rng.uniform(0, size)])
        else:
            p0 = np.array([size - 1.0, rng.uniform(0, size)])
        angle = rng.uniform(0, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        length = size * 1.6
        p1 = p0 + direction * length

        brightness = rng.uniform(0.35, 0.95)
        width = rng.uniform(0.7, 2.0)          # transverse PSF sigma
        n_samples = int(length * 2)
        ts = np.linspace(0, 1, n_samples)
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]

        # Brightness taper toward the ends + optional intensity flicker.
        taper = np.sin(np.pi * ts) ** 0.4
        flicker = 1.0 if rng.random() < 0.6 else (0.5 + 0.5 * np.sin(ts * rng.uniform(20, 60)))

        for (px, py), t, fl in zip(pts, taper, np.atleast_1d(flicker) * np.ones_like(ts)):
            if 0 <= px < size and 0 <= py < size:
                _gaussian_psf_star(streaked, px, py, brightness * t * fl, width)
                _gaussian_psf_star(mask, px, py, 1.0, width)

    streaked = np.clip(streaked, 0.0, 1.0).astype(np.float32)
    mask = (mask > 0.15).astype(np.float32)
    return streaked, mask


def make_pair(size: int = 256, rng: np.random.Generator | None = None):
    """Convenience: one (streaked, clean, mask) triple."""
    rng = rng or np.random.default_rng()
    clean = make_star_field(size, rng)
    streaked, mask = add_satellite_streak(clean, rng)
    return streaked, clean, mask


def build_dataset(n: int, size: int = 256, seed: int = 0):
    """Build n paired samples. Returns arrays shaped (n, size, size, 1)."""
    rng = np.random.default_rng(seed)
    X = np.empty((n, size, size, 1), dtype=np.float32)
    Yclean = np.empty((n, size, size, 1), dtype=np.float32)
    Ymask = np.empty((n, size, size, 1), dtype=np.float32)
    for i in range(n):
        s, c, m = make_pair(size, rng)
        X[i, ..., 0] = s
        Yclean[i, ..., 0] = c
        Ymask[i, ..., 0] = m
    return X, Yclean, Ymask


if __name__ == "__main__":
    X, C, M = build_dataset(4, size=128, seed=42)
    print("streaked:", X.shape, "clean:", C.shape, "mask:", M.shape)
    print("streak pixel fraction:", float(M.mean()))
