"""Inference helpers for JetSeg.

Provides small utilities to load ONNX sessions, preprocess images,
postprocess prediction maps and save composite RGB|GT|PRED images.
"""
from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple

import onnxruntime as ort


def preprocess_image(img: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Resize and normalize an image to [0,1] float32 with shape (1,H,W,3)."""
    h, w = input_size[1], input_size[0]
    resized = cv2.resize(img, (w, h))
    arr = resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def postprocess_mask(prob_map: np.ndarray, threshold: float = 0.5, to_uint8: bool = True) -> np.ndarray:
    """Turn probability map (H,W) or (1,H,W) into a binary mask.

    Returns mask with values 0/255 (uint8) when `to_uint8` is True.
    """
    if prob_map.ndim == 3 and prob_map.shape[0] == 1:
        prob_map = prob_map[0]
    mask = (prob_map > threshold).astype(np.uint8)
    if to_uint8:
        return mask * 255
    return mask


def load_onnx_session(onnx_path: str, providers=None):
    """Create an ONNX Runtime session for a given path.

    Providers default to available providers with preference for CUDA/TensorRT.
    """
    if providers is None:
        providers = ort.get_available_providers()
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def run_onnx(session, input_name: str, input_tensor: np.ndarray):
    outputs = session.run(None, {input_name: input_tensor})
    return outputs[0]


def save_composite(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, out_path: str, height: int = 480):
    """Create and save a side-by-side composite: RGB | GT | PRED scaled to `height`."""
    # scale each column to requested height
    h = height
    # ensure single-channel masks -> 3-channel for visualization
    def to_rgb(img_or_mask):
        if img_or_mask.ndim == 2:
            return cv2.cvtColor(img_or_mask, cv2.COLOR_GRAY2BGR)
        return img_or_mask

    cols = [image, to_rgb(gt_mask), to_rgb(pred_mask)]
    resized_cols = []
    for c in cols:
        hh = int(h)
        ww = int(c.shape[1] * (hh / c.shape[0]))
        resized_cols.append(cv2.resize(c, (ww, hh)))

    comp = np.hstack(resized_cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, comp)
