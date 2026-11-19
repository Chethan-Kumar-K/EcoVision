import base64
import io
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from rasterio import mask
from rasterio.io import MemoryFile

from src.inference import load_model, predict_prob

PIXEL_HA = 0.01   # 10 m Sentinel-2 pixels

app = FastAPI(title="EcoVision API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # the React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)
model = load_model()


def png_data_url(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/predict")
def predict(file: UploadFile = File(...), threshold: float = 0.5):
    try:
        with MemoryFile(file.file.read()) as mem:
            with mem.open() as src:
                img = src.read()
    except Exception:
        raise HTTPException(400, "Could not read the file as a GeoTIFF.")

    if img.shape[0] != 4:
        raise HTTPException(400, f"Expected 4 bands (R, G, B, NIR), got {img.shape[0]}.")
    if img.max() <= 255:
        raise HTTPException(400, "Values look 8-bit. Expected raw Sentinel-2 L2A reflectance x 10000.")

    t = time.time()
    mask = predict_prob(model, img) > threshold
    print(f"image {img.shape}: inference took {time.time() - t:.1f}s", flush=True)
    rgb = (np.clip(np.transpose(img[:3], (1, 2, 0)) / 1500.0, 0, 1) * 255).astype(np.uint8)
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask] = [34, 197, 94, 255]   # green where forest, transparent elsewhere

    return {
        "forest_percent": round(float(100 * mask.mean()), 2),
        "forest_ha": round(float(mask.sum() * PIXEL_HA), 1),
        "width": int(img.shape[2]),
        "height": int(img.shape[1]),
        "image_png": png_data_url(rgb),
        "mask_png": png_data_url(mask.astype(np.uint8) * 255),
        "overlay_png": png_data_url(overlay),
    }