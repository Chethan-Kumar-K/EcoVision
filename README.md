# EcoVision

Forest detection from Sentinel-2 satellite imagery. A UNet (ResNet34 encoder) segments
forest vs non-forest on 4-band images (R, G, B, NIR), served through a FastAPI backend
with a React frontend.

## Results

Trained on the Amazon and Atlantic Forest datasets combined (984 training images),
evaluated per biome on the held-out test sets (20 images each).

| Biome | Test IoU | Test F1 |
|---|---|---|
| Amazon | 0.958 | 0.978 |
| Atlantic Forest | 0.882 | 0.938 |

Metrics are for the forest class. Atlantic Forest is harder: forest is only about 33% of
its pixels and occurs in small fragmented patches.

## Method

- **Data:** Bragagnolo, da Silva & Grzybowski (2021), *Amazon and Atlantic Forest image
  datasets for semantic segmentation*, Zenodo, DOI 10.5281/zenodo.4498086 (CC BY 4.0).
  512x512 Sentinel-2 L2A tiles, bands B4, B3, B2, B8, stored as uint16 reflectance x 10000.
- **Preprocessing:** divide by 10,000, clip to [0, 1], standardize each band with
  training-set mean and std (`data/band_stats.json`). One shared function
  (`src/dataset.py: preprocess`) is used for training and inference.
- **Model:** `segmentation_models_pytorch` UNet, ResNet34 encoder with ImageNet weights,
  first convolution adapted to 4 input bands.
- **Training:** 256x256 random crops with flips and 90-degree rotations, BCE + Dice loss,
  AdamW (lr 3e-4, cosine schedule), batch size 8, 20 epochs on CPU. The checkpoint with
  the best mean validation IoU (epoch 19) was kept.

## Setup

```bash
uv sync
```

Download `AMAZON.rar` and `ATLANTIC FOREST.rar` from the Zenodo record into `data/raw/`
and extract them into `data/extracted/amazon/` and `data/extracted/atlantic/`.
Model weights are not stored in the repository: retrain with the notebooks in
`notebooks/`, which saves `models/unet_resnet34_best.pt`.

## Run

```bash
# backend (terminal 1)
uv run uvicorn api:app --reload --port 8000

# frontend (terminal 2)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and upload a 4-band GeoTIFF.

## Input requirements

4-band GeoTIFF in the order B4, B3, B2, B8 (R, G, B, NIR) with raw Sentinel-2 L2A
reflectance values (about 0 to 10,000). Other inputs are rejected. Area figures assume
10 m pixels.

## Limitations

- The model detects forest vs non-forest in a single image. It is not a deforestation
  detector by itself.
- Clouds and cloud shadows are labelled non-forest in the training data, so the model
  predicts them as non-forest. Over forest, they can look like forest loss.
- Test sets are small (20 images per biome), so scores are approximate.
- Trained on two Brazilian biomes only. Other regions are untested.