#!/usr/bin/env python3
"""Run inference over images and save composite RGB|GT|PRED outputs.

The script uses `HumanSeg` to run prediction. By default it will
attempt on-device quantization (interactive) unless `--no-quantize`
is provided.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run_inference_and_save")
    parser.add_argument("--model", default="jetseg/jetseg/model_store/humanseg/human_seg_large/human_seg.onnx", help="Path to ONNX model")
    parser.add_argument("--input", default="images/", help="Input images folder")
    parser.add_argument("--output", default="images/results/", help="Output folder for composites")
    parser.add_argument("--no-quantize", action="store_true", help="Do not attempt on-device quantization; use model as-is")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "torch"], help="Backend to use")
    parser.add_argument("--use-fp16", action="store_true", help="Prefer FP16 variants when available")
    args = parser.parse_args(argv)

    # import heavy deps inside main so tests can import this module safely
    import cv2
    from jetseg.engine import HumanSeg
    from jetseg import inference as inference_mod

    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    hs = HumanSeg(use_fp16=args.use_fp16, cache_dir=None, backend=args.backend, model_path=args.model, input_size=None, no_quantize=args.no_quantize)

    for p in sorted(input_dir.iterdir() if input_dir.exists() else []):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"Skipping unreadable image: {p}")
            continue

        pred = hs.predict(img)
        if pred is None:
            print(f"Prediction returned None for {p}")
            continue

        # no ground-truth available in this demo: use empty mask
        gt_mask = np.zeros_like(pred)

        out_path = out_dir / f"{p.stem}_comp.png"
        inference_mod.save_composite(img, gt_mask, pred, str(out_path))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
