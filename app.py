import numpy as np
import streamlit as st
from rasterio.io import MemoryFile

from src.inference import load_model, predict_prob

PIXEL_HA = 0.01   # Sentinel-2 bands 2,3,4,8 are 10 m pixels -> 100 m2 = 0.01 ha

st.set_page_config(page_title="EcoVision", layout="wide")
st.title("EcoVision: forest detection from Sentinel-2")


@st.cache_resource
def get_model():
    return load_model()


def read_tif(file):
    with MemoryFile(file.getvalue()) as mem:
        with mem.open() as src:
            return src.read()


def check(img):
    """Return an error message if the input doesn't match the training data, else None."""
    if img.shape[0] != 4:
        return f"Expected 4 bands (R, G, B, NIR), got {img.shape[0]}."
    if img.max() <= 255:
        return "Values look 8-bit. Expected raw Sentinel-2 L2A reflectance x 10000 (values up to several thousand)."
    return None


def rgb_view(img):
    return np.clip(np.transpose(img[:3], (1, 2, 0)) / 1500.0, 0, 1)


file = st.file_uploader("Upload a 4-band GeoTIFF (B4, B3, B2, B8 in that order)", type=["tif", "tiff"])
threshold = st.slider("Forest threshold", 0.1, 0.9, 0.5, 0.05)

if file:
    img = read_tif(file)
    error = check(img)
    if error:
        st.error(error)
    else:
        with st.spinner("Running model..."):
            prob = predict_prob(get_model(), img)
        mask = prob > threshold
        forest_ha = mask.sum() * PIXEL_HA

        c1, c2, c3 = st.columns(3)
        c1.metric("Forest cover", f"{100 * mask.mean():.1f}%")
        c2.metric("Forest area", f"{forest_ha:,.1f} ha")
        c3.metric("Image size", f"{img.shape[2]} x {img.shape[1]} px")

        left, right = st.columns(2)
        left.image(rgb_view(img), caption="Image (RGB)", use_container_width=True)
        right.image(mask.astype(np.uint8) * 255, caption="Predicted forest (white)", use_container_width=True)

        st.caption("Note: clouds and cloud shadows are classed as non-forest, as in the training data.")