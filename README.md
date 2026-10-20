# ClearSky — Satellite Streak Detection & Removal

Low-Earth-orbit satellite constellations leave bright trails across long-exposure
telescope images, corrupting the data astronomers depend on. **ClearSky** is a
full-stack application that detects those trails with a U-Net and reconstructs a
clean image — upload a frame, get back a streak mask and a cleaned result in real
time.

<p align="center">
  <img src="assets/pipeline_demo.png" width="640" alt="Streaked input, streak mask, and clean target"><br>
  <em>The data engine, left to right: synthetic streaked frame · streak mask · clean ground truth.</em>
</p>

---

## Why synthetic data

Large, pixel-annotated datasets of real satellite streaks barely exist. The
research community works around this by **injecting synthetic trails onto
realistic star fields** — the same U-Net segmentation approaches for streak
removal are routinely trained on thousands of synthetic 256×256 night-sky frames,
and current state-of-the-art pipelines train on a mix of real and synthetic data.

ClearSky follows that convention with a self-contained generator
(`ml/data/synthesize.py`) that builds realistic frames from scratch:

- **Star fields** — Poisson-distributed star counts with Gaussian PSF stars whose
  fluxes follow a steep power law (many faint, few bright), plus a sky-background
  gradient, photon shot noise, and read noise.
- **Satellite trails** — anti-aliased lines entering and exiting the frame at a
  random angle, with a transverse PSF, brightness taper toward the ends, and
  optional flicker to mimic tumbling satellites.

Every sample is a `(streaked, clean, mask)` triple, so the same data trains both
the detection and the removal heads. No downloads, runs anywhere.

## Model

`ml/models/unet.py` is a compact U-Net with a shared encoder/decoder backbone and
**two output heads**:

| Head    | Activation | Purpose                                   |
|---------|------------|-------------------------------------------|
| `mask`  | sigmoid    | per-pixel probability of a streak         |
| `clean` | sigmoid    | reconstructed streak-free image           |

Detection and removal are learned jointly in one forward pass. The mask head is
trained with a combined **binary-crossentropy + Dice** loss (robust to the heavy
class imbalance of thin streaks on a large sky), and the reconstruction head with
an L1 (MAE) loss.

## Results

Train the reference model:

```bash
pip install -r requirements-ml.txt
python -m ml.train.train --n 2000 --size 128 --epochs 15
```

This writes `ml/models/clearsky.weights.h5` and `ml/models/metrics.json`
(pixel-wise streak recall, IoU, and reconstruction PSNR on a held-out synthetic
validation split). The training script is deterministic given `--seed`.

> **Note on the committed demo weights.** The weights checked in here come from a
> short CPU run so the app is runnable out of the box; they under-detect at a
> fixed threshold. Run the command above (ideally on a GPU) to reproduce
> full-strength detection. The generator, model, losses, metrics, and API are all
> exercised by the test suite regardless of training budget.

## Full stack

```
┌─────────────┐   image    ┌──────────────┐   tensor   ┌─────────────┐
│  React + TS │ ─────────▶ │   FastAPI    │ ─────────▶ │  U-Net (TF) │
│  dropzone   │ ◀───────── │  /api/process│ ◀───────── │ mask + clean│
└─────────────┘  overlay,  └──────────────┘  arrays    └─────────────┘
                 cleaned
```

**Backend** (`backend/app/main.py`) — FastAPI service that loads the model once,
accepts a PNG/JPEG/TIFF upload, and returns base64 PNGs of the input, the
red streak overlay, the cleaned frame, and the raw mask, plus a coverage score.

**Frontend** (`frontend/`) — React + TypeScript (Vite). A drag-and-drop dropzone
posts the image and renders the input / detected-streak / cleaned panels. The UI
was refactored to TypeScript for type safety across the API boundary.

## Run it

```bash
# 1. backend
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload            # http://localhost:8000

# 2. frontend (separate shell)
cd frontend && npm install && npm run dev         # http://localhost:5173
```

## Tests

```bash
python -m pytest tests/ -q
```

Covers the data generator's output ranges and mask validity, the dataset builder
shapes, the API happy path, and rejection of non-image uploads.

## Layout

```
clearsky/
├── ml/
│   ├── data/synthesize.py     # synthetic star-field + streak generator
│   ├── models/unet.py         # dual-head U-Net + BCE/Dice loss
│   └── train/train.py         # training loop, metrics, checkpointing
├── backend/app/main.py        # FastAPI inference service
├── frontend/                  # React + TypeScript (Vite) dashboard
├── tests/test_pipeline.py     # pytest smoke suite
└── assets/                    # generated figures
```

## License

MIT
