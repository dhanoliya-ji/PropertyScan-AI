"""Monocular metric depth for the video and photo tiers.

Model: Depth Anything V2, Metric-Indoor (Hypersim fine-tune), via Hugging Face transformers.
Disclosed in docs/MODELS.md. Weights are fetched on first use into the HF cache
(scripts/fetch_weights.py pre-fetches them). Preprocessing is done here (resize to a
multiple of 14 with the short side 518, ImageNet normalisation) to avoid a torchvision
dependency.

Predictions are cached on disk keyed by (model, image bytes hash) so benchmark runs replay
deterministically; the live path recomputes anything not in the cache.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

MODEL_ID = os.environ.get("PROPSCAN_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
CACHE = Path(os.environ.get("PROPSCAN_CACHE", Path(__file__).resolve().parents[2] / ".cache" / "depth"))

_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForDepthEstimation
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        m = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = (m.to(dev), dev)
    return _model


def _prep(rgb, short=518):
    import cv2
    h, w = rgb.shape[:2]
    s = short / min(h, w)
    nh, nw = int(round(h * s / 14)) * 14, int(round(w * s / 14)) * 14
    x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return x.transpose(2, 0, 1)[None]


def predict(rgb: np.ndarray, out_hw=(192, 256)) -> np.ndarray:
    """rgb uint8 HxWx3 (RGB order) -> metric depth (metres) resized to out_hw."""
    import cv2
    key = hashlib.sha1(MODEL_ID.encode() + rgb.tobytes()[::7] + str(rgb.shape).encode()).hexdigest()
    f = CACHE / f"{key}.npy"
    if f.exists():
        d = np.load(f)
    else:
        import torch
        m, dev = _load()
        with torch.no_grad():
            d = m(pixel_values=torch.from_numpy(_prep(rgb)).to(dev)).predicted_depth[0].float().cpu().numpy()
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, d.astype(np.float16))
    d = d.astype(np.float32)
    return cv2.resize(d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
