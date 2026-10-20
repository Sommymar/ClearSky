# ClearSky — design notes

## Problem
Satellite trails corrupt long-exposure astronomical frames. Two tasks matter:
*finding* the affected pixels (so their scientific impact can be quantified) and
*removing* them (so the frame is usable). ClearSky does both from one model.

## Key decisions

**Synthetic data over scraping archives.** Annotated real-streak data is scarce
and inconsistent. Synthetic injection onto procedurally generated star fields is
the established workaround in the literature, gives perfect ground-truth masks
for free, and makes the repo reproducible with no external dependencies.

**One U-Net, two heads.** Detection and inpainting share features, so a single
encoder/decoder with a segmentation head and a reconstruction head is cheaper and
lets the removal head condition on the same representation that locates the streak.

**BCE + Dice for the mask.** Streaks are a few percent of pixels; plain BCE
collapses to predicting "no streak." Dice counteracts the imbalance.

**Calibrated thresholding at inference.** A fixed 0.5 cut is brittle when the
sigmoid head is not fully saturated, so the API thresholds relative to the mask's
own statistics.

## If this were productionized
- Train on real survey cutouts (SDSS/ZTF) with synthetic streaks mixed in.
- Swap the reconstruction head for an inpainting objective masked to streak pixels
  only, so clean regions pass through untouched.
- Serve the model with batching and a queue; stream tiles for large FITS frames.
