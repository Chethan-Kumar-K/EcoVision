import numpy as np
import torch
import segmentation_models_pytorch as smp

from src.dataset import preprocess, ROOT


def load_model(path=ROOT / "models" / "unet_resnet34_best.pt"):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


@torch.no_grad()
def predict_prob(model, img, tile=512):
    """img: uint16 array (4, H, W), bands R,G,B,NIR (Sentinel-2 L2A B4,B3,B2,B8),
    reflectance x 10000. Any size. Returns forest probability (H, W) in [0, 1]."""
    x = preprocess(img)
    _, H, W = x.shape
    ph, pw = (-H) % tile, (-W) % tile
    x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="edge")
    out = np.zeros(x.shape[1:], dtype=np.float32)
    for i in range(0, x.shape[1], tile):
        for j in range(0, x.shape[2], tile):
            t = torch.from_numpy(np.ascontiguousarray(x[:, i:i + tile, j:j + tile]))[None]
            out[i:i + tile, j:j + tile] = torch.sigmoid(model(t))[0, 0].numpy()
    return out[:H, :W]