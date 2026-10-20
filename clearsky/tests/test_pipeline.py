"""Smoke tests for the ClearSky data engine and inference API."""

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("CLEARSKY_SIZE", "128")

from ml.data.synthesize import make_pair, build_dataset  # noqa: E402


def test_star_field_range():
    s, c, m = make_pair(128, np.random.default_rng(0))
    assert s.shape == (128, 128)
    assert 0.0 <= s.min() and s.max() <= 1.0
    assert 0.0 <= c.min() and c.max() <= 1.0


def test_mask_is_binary_and_nonempty():
    _, _, m = make_pair(128, np.random.default_rng(1))
    uniq = set(np.unique(m).tolist())
    assert uniq.issubset({0.0, 1.0})
    assert m.sum() > 0, "a satellite streak should always be injected"


def test_dataset_shapes():
    X, C, M = build_dataset(5, size=64, seed=2)
    assert X.shape == C.shape == M.shape == (5, 64, 64, 1)


def test_api_health_and_process():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    s, _, _ = make_pair(128, np.random.default_rng(3))
    buf = io.BytesIO()
    Image.fromarray((s * 255).astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    res = client.post("/api/process", files={"file": ("t.png", buf, "image/png")})
    assert res.status_code == 200
    body = res.json()
    for key in ("input", "overlay", "cleaned", "mask", "streak_coverage"):
        assert key in body


def test_api_rejects_non_image():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    res = client.post("/api/process", files={"file": ("t.txt", b"hello", "text/plain")})
    assert res.status_code == 415
