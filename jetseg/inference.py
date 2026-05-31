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


def save_composite(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, out_path: str, height: int = 480, mode: str = "side_by_side"):
    """Save composite visualization.

    mode: 'side_by_side' (default) produces RGB | GT | PRED horizontally stacked.
          'overlay' produces a single image where prediction is blended onto the RGB.
    """
    h = int(height)

    def to_rgb(img_or_mask):
        if img_or_mask.ndim == 2:
            return cv2.cvtColor(img_or_mask, cv2.COLOR_GRAY2BGR)
        return img_or_mask

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if mode == "overlay":
        # create colored overlay where pred_mask > 0
        img = image.copy()
        pm = pred_mask
        if pm.ndim == 3 and pm.shape[0] == 1:
            pm = pm[0]
        if pm.ndim == 3:
            pm = cv2.cvtColor(pm, cv2.COLOR_BGR2GRAY)
        mask_bool = (pm > 0).astype(np.uint8)
        if mask_bool.sum() == 0:
            # nothing predicted, just resize original
            ww = int(img.shape[1] * (h / img.shape[0]))
            comp = cv2.resize(img, (ww, h))
            cv2.imwrite(out_path, comp)
            return

        # colored mask (red)
        color_mask = np.zeros_like(img)
        color_mask[:] = (0, 0, 255)
        blended = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
        mask_3ch = np.repeat(mask_bool[:, :, None], 3, axis=2)
        overlay = np.where(mask_3ch.astype(bool), blended, img)
        ww = int(overlay.shape[1] * (h / overlay.shape[0]))
        comp = cv2.resize(overlay, (ww, h))
        cv2.imwrite(out_path, comp)
        return

    # default: side_by_side
    cols = [image, to_rgb(gt_mask), to_rgb(pred_mask)]
    resized_cols = []
    for c in cols:
        hh = h
        ww = int(c.shape[1] * (hh / c.shape[0]))
        resized_cols.append(cv2.resize(c, (ww, hh)))

    comp = np.hstack(resized_cols)
    cv2.imwrite(out_path, comp)
