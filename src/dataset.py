import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "extracted"
BASES = {
    "amazon": DATA / "amazon" / "AMAZON",
    "atlantic": DATA / "atlantic" / "ATLANTIC FOREST",
}
SPLIT_DIRS = {
    "Training": ("image", "label"),
    "Validation": ("images", "masks"),
    "Test": ("image", "mask"),
}
STATS = json.loads((ROOT / "data" / "band_stats.json").read_text())
MEAN = np.array(STATS["mean"], dtype=np.float32).reshape(4, 1, 1)
STD = np.array(STATS["std"], dtype=np.float32).reshape(4, 1, 1)


def preprocess(img):
    """uint16 array (4, H, W) in R,G,B,NIR order -> normalized float32.
    The one preprocessing function used for training AND inference."""
    img = np.clip(img.astype(np.float32) / STATS["scale"], 0.0, 1.0)
    return (img - MEAN) / STD


class ForestDataset(Dataset):
    """One biome, one split. train=True -> random crop + flips/rotations."""

    def __init__(self, biome, split, train=False, crop_size=256):
        img_dir, msk_dir = SPLIT_DIRS[split]
        base = BASES[biome] / split
        self.biome = biome
        self.img_paths = sorted((base / img_dir).glob("*.tif"))
        self.msk_dir = base / msk_dir
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        p = self.img_paths[i]
        with rasterio.open(p) as src:
            img = src.read()
        with rasterio.open(self.msk_dir / p.name) as src:
            msk = src.read(1)

        img = preprocess(img)

        if self.train:
            c = self.crop_size
            top = np.random.randint(0, img.shape[1] - c + 1)
            left = np.random.randint(0, img.shape[2] - c + 1)
            img = img[:, top:top + c, left:left + c]
            msk = msk[top:top + c, left:left + c]
            if np.random.rand() < 0.5:
                img, msk = img[:, :, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1, :], msk[::-1, :]
            k = np.random.randint(0, 4)
            img, msk = np.rot90(img, k, axes=(1, 2)), np.rot90(msk, k)

        x = torch.from_numpy(np.ascontiguousarray(img))
        y = torch.from_numpy(np.ascontiguousarray(msk)).float().unsqueeze(0)
        return x, y